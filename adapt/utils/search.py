"""Methods for searching for optimal guides to use for a diagnostic.
"""

from collections import defaultdict
import logging
import math
import random

import numpy as np

from adapt import alignment
from adapt.utils import index_compress
from adapt.utils import lsh
from adapt.utils import predict_activity

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


class OligoSearcher:
    """Methods to search for oligos to use for a diagnostic.

    This is a base class, with subclasses depending on oligo and objective.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable oligos.
    """

    def __init__(self, aln, min_oligo_length, max_oligo_length,
            missing_data_params, is_suitable_fns=[], required_oligos={},
            ignored_ranges={}, allow_gu_pairs=False,
            required_flanking_seqs=(None, None), do_not_memoize=False,
            predictor=None):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            min_oligo_length: minimum length of the oligo to construct
            max_oligo_length: maximum length of the oligo to construct
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design oligos overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m)), where m is
                the median fraction of sequences with missing data over the
                alignment
            is_suitable_fns: if set, the value of this argument is a list
                of functions f(x) such that this will only construct a oligo x
                for which each f(x) is True
            required_oligos: dict that maps oligo sequences to their position
                in the alignment; all of these oligo sequences are immediately
                placed in the set of covering oligos for their appropriate
                windows before finding other oligos, so that they are
                guaranteed to be in the output (i.e., the set of covering
                oligos is initialized with these oligos)
            ignored_ranges: set of tuples (start, end) that provide
                ranges in the alignment from which oligos should not be
                constructed. No oligo that might overlap these ranges is
                constructed. Note that start is inclusive and end is
                exclusive.
            allow_gu_pairs: if True, tolerate G-U base pairs between a
                oligo and target when computing whether an oligo binds
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the oligo (in the target, not the oligo) that must be
                present for a oligo to bind; if either is None, no
                flanking sequence is required for that end
            do_not_memoize: if True, never memoize the results of
                computed oligos at a site and always compute the oligos (this
                can be useful if we know the memoized result will never be used
                and memoizing it may be slow, or if we want to benchmark
                performance with/without memoization)
            predictor: adapt.utils.predict_activity.Predictor object. If
                None, do not predict activities.
        """
        self.aln = aln
        assert min_oligo_length <= max_oligo_length
        self.min_oligo_length = min_oligo_length
        self.max_oligo_length = max_oligo_length

        # Because calls to compute oligos at a site can be expensive and
        # repeated very often, memoize the output
        self._memo = defaultdict(dict)
        self._memo_last_inner_dict = None
        self._memo_last_inner_dict_key = None
        self._memo_num_removed_since_last_resize = 0
        self.do_not_memoize = do_not_memoize

        # Save the positions of selected oligos in the alignment so these can
        # be easily revisited. In case a oligo sequence appears in multiple
        # places, store a set of positions
        self._selected_positions = defaultdict(set)

        # Determine a threshold at which to ignore sites with too much
        # missing data
        missing_max, missing_min, missing_coeff = missing_data_params
        self.missing_threshold = min(missing_max, max(missing_min,
            missing_coeff * self.aln.median_sequences_with_missing_data()))

        self.is_suitable_fns = is_suitable_fns

        self.required_oligos = required_oligos

        # Verify positions of the oligos are within the alignment
        highest_possible_pos = self.aln.seq_length - self.min_oligo_length
        for olg, olg_pos in self.required_oligos.items():
            if olg_pos < 0 or olg_pos > highest_possible_pos:
                raise Exception(("An oligo with sequence '%s' at position %d "
                    "is required to be in the output, but does not fall "
                    "within the alignment") % (olg, olg_pos))

        # Because determining which sequences are covered by each required
        # oligo is expensive and will be done repeatedly on the same oligo,
        # memoize them
        self._memoized_seqs_covered_by_required_oligos = {}

        self.ignored_ranges = ignored_ranges

        # Verify ignored ranges are within the alignment
        for start, end in ignored_ranges:
            if start < 0 or end <= start or end > self.aln.seq_length:
                raise Exception(("An ignored range [%d, %d) is invalid "
                    "for a given alignment; ranges must fall within the "
                    "alignment: [0, %d)") % (start, end, self.aln.seq_length))

        self.allow_gu_pairs = allow_gu_pairs

        self.required_flanking_seqs = required_flanking_seqs

        self.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(max_oligo_length),
            k=min(10, int(max_oligo_length/2)))

        self.predictor = predictor

    def _compute_memoized(self, start, call_fn, key_fn, use_last=False):
        """Make a memoized call to compute an oligo.

        The actual computation is defined in a subclass and passed as
        a function (call_fn) here -- it can be to construct a oligo or
        to compute information (e.g., expected activity) about different
        oligos in a ground set.

        Args:
            start: start position in alignment at which to target
            call_fn: function to call for compute oligo(s)
            key_fn: function call to construct a key from current state
            use_last: if set, check for a memoized result by using the last
                key constructed (it is assumed that key is identical to
                the last provided values)

        Returns:
            result of call_fn()
        """
        if use_last:
            # key (defined below) can be large and slow to hash; therefore,
            # assume that the last computed key is identical to the one that
            # would be computed here (i.e., call_state is the same), and use
            # the last inner dict to avoid having to hash key
            assert self._memo_last_inner_dict is not None
            inner_dict = self._memo_last_inner_dict
        else:
            key = key_fn()
            if key not in self._memo:
                self._memo[key] = {}
            inner_dict = self._memo[key]
            self._memo_last_inner_dict_key = key

        if start in inner_dict:
            # The result has been memoized

            p_memoized = inner_dict[start]

            # p was compressed before memoizing it; decompress
            # it before returning it
            if p_memoized is None:
                p = None
            else:
                p = self._decompress_result(p_memoized)
        else:
            # The result is not memoized; compute it and memoize it

            p = call_fn()

            if p is None:
                p_to_memoize = None
            else:
                # Compress p before memoizing it.
                p_to_memoize = self._compress_result(p)

            inner_dict[start] = p_to_memoize

        self._memo_last_inner_dict = inner_dict
        return p

    def _cleanup_memo(self, start, frac_removed_until_resize=0.1):
        """Remove a position that is stored in self._memo.

        This should be called when the position no longer needs to be stored.

        Python only resizes dicts on insertions (and, seemingly, only after
        reaching a sufficiently large size); see
        https://github.com/python/cpython/blob/master/Objects/dictnotes.txt
        In this case, it may never resize or resize too infrequently,
        especially for the inner dicts. It appears in many cases to never
        resize. Therefore, this "resizes" the self._memo dict at
        certain cleanups by copying all the content over to a new dict,
        effectively forcing it to shrink its memory usage. It does this
        by computing the number of elements that have been removed from the
        data structure relative to its current total number of elements,
        and resizing when this fraction exceeds `frac_removed_until_resize`.
        Since the total number of elements should stay roughly constant
        as we scan along the alignment (i.e., start increases), this fraction
        should grow over time. At each resizing, the fraction will drop back
        down to 0.

        This also cleans up memoizations in the predictor, if that was set.

        Args:
            start: start position that no longer needs to be memoized (i.e., where
                oligos covering at that position are no longer needed)
            frac_removed_until_resize: resize the self._memo
                data structure when the total number of elements removed
                due to cleanup exceeds this fraction of the total size
        """
        keys_to_rm = set()
        for key in self._memo.keys():
            if start in self._memo[key]:
                del self._memo[key][start]
                self._memo_num_removed_since_last_resize += 1
            if len(self._memo[key]) == 0:
                keys_to_rm.add(key)

        for key in keys_to_rm:
            del self._memo[key]

        # Decide whether to resize
        total_size = sum(len(self._memo[k])
                for k in self._memo.keys())
        if total_size > 0:
            frac_removed = float(self._memo_num_removed_since_last_resize) / total_size
            logger.debug(("Deciding to resize with a fraction %f removed "
                "(%d / %d)"), frac_removed,
                self._memo_num_removed_since_last_resize,
                total_size)
            if frac_removed >= frac_removed_until_resize:
                # Resize self._memo by copying all content to a new dict
                new_memo = defaultdict(dict)
                for key in self._memo.keys():
                    for i in self._memo[key].keys():
                        new_memo[key][i] = self._memo[key][i]
                    if key == self._memo_last_inner_dict_key:
                        self._memo_last_inner_dict = new_memo[key]
                self._memo = new_memo
                self._memo_num_removed_since_last_resize = 0

        # Cleanup the predictor's memoizations at this position
        if self.predictor is not None:
            self.predictor.cleanup_memoized(start)

    def _overlaps_ignored_range(self, olg_start, olg_len=None):
        """Determine whether a oligo would overlap an ignored range.

        The ignored ranges are given in self.ignored_ranges.

        Args:
            olg_start: start position of an oligo
            olg_len: length of an oligo (defaults to min oligo length)

        Returns:
            True iff the oligo overlaps a ignored range
        """
        if not olg_len:
            olg_len = self.min_oligo_length

        olg_end = olg_start + olg_len - 1

        for start, end in self.ignored_ranges:
            if ((olg_start >= start and olg_start < end) or
                    (olg_end >= start and olg_end < end)):
                return True
        return False

    def oligo_set_activities(self, window_start, window_end, oligo_set):
        """Compute activity across target sequences for oligo set in a window.

        Let S be the set of target sequences. Let G be oligo_set, and
        let the predicted activity of a oligo g in detecting s_i \in S
        be d(g, s_i). Then the activity of G in detecting s_i is
        max_{g \in G} d(g, s_i). We can compute an activity for G
        against all sequences s_i by repeatedly taking element-wise
        maxima; see
        OligoSearcherMaximizeActivity._activities_after_adding_oligo() for
        an explanation.

        Note that if oligo_set is a subset of the ground set, then
        we already have these computed in OligoSearcherMaximizeAcitivity.
        Re-implementing it here lets us use have them with
        OligoSearcherMinimizeoligos, and is also not a bad check
        to re-compute.

        This assumes that oligo positions for each oligo in oligo_set
        are stored in self._selected_positions.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            oligo_set: set of strings representing oligo sequences that
                have been selected to be in a oligo set

        Returns:
            list of activities across the target sequences (self.aln)
            yielded by oligo_set
        """
        if self.predictor is None:
            raise NoPredictorError(("Cannot compute activities when "
                "predictor is not set"))

        activities = np.zeros(self.aln.num_sequences)
        for seq in oligo_set:
            if seq not in self._selected_positions:
                raise Exception(("Oligo must be selected and its position "
                    "saved"))

            # The oligo could hit multiple places
            for start in self._selected_positions[seq]:
                if start < window_start or start > window_end - len(seq):
                    # oligo is outside window
                    continue
                try:
                    olg_activities = self.aln.compute_activity(start, seq,
                            self.predictor)
                except CannotConstructOligoError:
                    # Most likely this site is too close to an endpoint and
                    # does not have enough context_nt; skip it
                    continue

                # Update activities with olg_activities
                activities = np.maximum(activities, olg_activities)

        return activities

    def oligo_set_activities_percentile(self, window_start, window_end,
            oligo_set, q, activities=None):
        """Compute percentiles of activity across target sequences for
        a oligo set in a window.

        For example, when percentiles is 0.5, this returns the median
        activity across the target sequences that the oligo set provides.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            oligo_set: set of strings representing oligo sequences that
                have been selected to be in a oligo set
            q: list of percentiles to compute, each in [0,100]
                (0 is minimum, 100 is maximum)
            activities: output of self.oligo_set_activities(); if not set,
                this calls that function

        Returns:
            list of percentile values
        """
        if activities is None:
            activities = self.oligo_set_activities(window_start, window_end,
                    oligo_set)

        # Do not interpolate (choose lower value)
        p = np.percentile(activities, q, interpolation='lower')
        return list(p)

    def oligo_set_activities_expected_value(self, window_start, window_end,
            oligo_set, activities=None):
        """Compute expected activity across target sequences for
        a oligo set in a window.

        This assumes the distribution across target sequences is uniform,
        so it is equivalent to the mean.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            oligo_set: set of strings representing oligo sequences that
                have been selected to be in a oligo set
            activities: output of self.oligo_set_activities(); if not set,
                this calls that function

        Returns:
            expected (here, mean) activity
        """
        if activities is None:
            activities = self.oligo_set_activities(window_start, window_end,
                    oligo_set)

        return np.mean(activities)

    def oligo_activities_expected_value(self, window_start, window_end,
            olg_seq):
        """Compute expected activity across target sequences for a single
        oligo in a window.

        Let S be the set of target sequences. Then the activity of a oligo
        g in detecting s_i \in S be d(g, s_i). We assume a uniform
        distribution over the s_i, so the expected value for g is the
        mean of d(g, s_i) across the s_i.

        Note that if olg_seq is in the ground set, then we already have these
        computed in OligoSearcherMaximizeAcitivity. Re-implementing it here
        lets us use have them with OligoSearcherMinimizeNumber, and is also not
        a bad check to re-compute.

        This assumes that the oligo position for olg_seq is stored in
        self._selected_positions.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            olg_seq: string representing oligo sequence

        Returns:
            mean activity across the target sequences (self.aln) yielded
            by olg_seq
        """
        if self.predictor is None:
            raise NoPredictorError(("Cannot compute activities when "
                "predictor is not set"))

        if olg_seq not in self._selected_positions:
            raise Exception(("Oligo must be selected and its position "
                "saved"))

        # The oligo could hit multiple places; account for that just in case
        activities = np.zeros(self.aln.num_sequences)
        for start in self._selected_positions[olg_seq]:
            if start < window_start or start > window_end - len(olg_seq):
                # oligo is outside window
                continue
            try:
                olg_activities = self.aln.compute_activity(start, olg_seq,
                        self.predictor)
            except CannotConstructOligoError:
                # Most likely this site is too close to an endpoint and
                # does not have enough context_nt; skip it
                continue

            activities = np.maximum(activities, olg_activities)

        return np.mean(activities)

    def _find_oligos_for_each_window(self, window_size, window_step=1,
            hide_warnings=False):
        """Find a collection of oligos in each window.

        This runs a sliding window across the aligned sequences and, in each
        window, computes a oligo set by calling self._find_oligos_in_window().

        This returns oligos for each window.

        This does not return oligos for windows where it cannot design
        oligos in the window (e.g., due to indels or ambiguity).

        Args:
            window_size: length of the window to use when sliding across
                alignment
            window_step: amount by which to increase the window start for
                every search
            hide_warnings: when set, this does not provide log warnings
                when no more suitable oligos can be constructed

        Returns:
            yields x_i in which each x_i corresponds to a window;
            x_i is a tuple consisting of the following values, in order:
              1) start position of the window
              2) end position of the window
              3) set of oligos for the window
        """
        if window_size > self.aln.seq_length:
            raise ValueError(("window size must be < the length of the "
                              "alignment"))

        for start in range(0, self.aln.seq_length - window_size + 1,
                window_step):
            end = start + window_size
            logger.info("Searching for oligos within window [%d, %d)" %
                        (start, end))

            try:
                oligos = self._find_oligos_in_window(start, end)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                if not hide_warnings:
                    logger.warning(("No more suitable oligos could be constructed "
                        "in the window [%d, %d), but more are needed to "
                        "achieve the desired coverage") % (start, end))
                self._cleanup_memo(start)
                continue
            except CannotFindAnyOligosError:
                # Cannot find any oligos in this window; log and skip it
                if not hide_warnings:
                    logger.warning(("No suitable oligos could be constructed "
                        "in the window [%d, %d)") % (start, end))
                self._cleanup_memo(start)
                continue

            yield (start, end, oligos)

            # We no longer need to memoize results for oligos that start at
            # this position
            self._cleanup_memo(start)

    def _compress_result(self, p):
        """Compress the information to be stored in self._memo

        By default, this doesn't compress the information in self._memo.
        Override this in subclasses if compression is needed

        Args:
            p: information to be stored in self._memo

        Returns:
            compressed version of p
        """
        return p

    def _decompress_result(self, p_compressed):
        """Decompress the information stored in self._memo

        By default, this doesn't decompress the information in self._memo.
        Override this in subclasses if decompression is needed

        Args:
            p_compressed: compressed information stored in self._memo

        Returns:
            decompressed version of p_compressed
        """
        return p_compressed

    def obj_value(self, *args, **kwargs):
        """Compute objective value for an oligo set

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of OligoSearcher must implement "
                                  "obj_value")

    def best_obj_value(self, *args, **kwargs):
        """Estimate the best possible objective value of an oligo

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of OligoSearcher must implement "
                                  "best_obj_value")

    def _find_oligos_in_window(self, *args, **kwargs):
        """Find a collection of oligos with the best objective in a given window

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of OligoSearcher must implement "
                                  "_find_optimal_oligo_in_window")


class OligoSearcherMinimizeNumber(OligoSearcher):
    """Methods to minimize the number of oligos.

    This is a base class, with subclasses defining methods depending on the
    oligo. It should not be used without subclassing, as it does not define
    all the positional arguments necessary for search.OligoSearcher.
    """

    def __init__(self, cover_frac, mismatches, seq_groups=None, **kwargs):
        """
        Args:
            cover_frac: fraction in (0, 1] of sequences that must be 'captured'
                by an oligo; see seq_groups
            mismatches: threshold on number of mismatches for determining
                whether an oligo would hybridize to a target sequence
            seq_groups: dict that maps group ID to collection of sequences in
                that group. If set, cover_frac must also be a dict that maps
                group ID to the fraction of sequences in that group that
                must be 'captured' by a oligo. If None, then do not divide
                the sequences into groups.
            kwargs: see OligoSearcher.__init__()
        """
        self.mismatches = mismatches

        super().__init__(**kwargs)

        if seq_groups is None and (cover_frac <= 0 or cover_frac > 1):
            raise ValueError("cover_frac must be in (0,1]")

        if seq_groups is not None:
            # Check that each group has a valid cover fraction
            for group_id in seq_groups.keys():
                assert group_id in cover_frac
                if cover_frac[group_id] <= 0 or cover_frac[group_id] > 1:
                    raise ValueError(("cover_frac for group %d must be in "
                                      "(0,1]") % group_id)

            # Check that each sequence is only in one group
            seqs_seen = set()
            for group_id, seqs in seq_groups.items():
                for s in seqs:
                    assert s not in seqs_seen
                seqs_seen.update(seqs)

            # Check that each sequence is in a group
            for i in range(self.aln.num_sequences):
                assert i in seqs_seen

            self.seq_groups = seq_groups
            self.cover_frac = cover_frac
        else:
            # Setup a single dummy group containing all sequences, and make
            # cover_frac be the fraction of sequences that must be covered in
            # this group
            self.seq_groups = {0: set(range(self.aln.num_sequences))}
            self.cover_frac = {0: cover_frac}

    def _construct_memoized(self, start, seqs_to_consider,
            num_needed=None, use_last=False, memoize_threshold=0.1):
        """Make a memoized call to get the next oligo to add to an oligo set

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: dict mapping universe group ID to collection
                of indices to use when constructing the oligo
            num_needed: dict mapping universe group ID to the number of
                sequences from the group that are left to cover in order to
                achieve the desired partial cover
            use_last: if set, check for a memoized result by using the last
                key constructed (it is assumed that seqs_to_consider and
                num_needed are identical to the last provided values)
            memoize_threshold: only memoize results when the total fraction
                of sequences in seqs_to_consider (compared to the whole
                alignment) exceeds this threshold

        Returns:
            tuple (x, y, z) where:
                x is the sequence of the constructed oligo
                y is a set of indices of sequences (a subset of
                    values in seqs_to_consider) to which the guide x will
                    hybridize
                z is the marginal contribution of the oligo to the objective
            (Note that it is possible that x binds to no sequences and that
            y will be empty.)
        """
        def construct_p():
            p_best = (None, None, 0)
            for oligo_length in range(self.min_oligo_length, min(
                    self.max_oligo_length+1, self.aln.seq_length)):
                try:
                    p = self.construct_oligo(start, oligo_length, seqs_to_consider,
                                             num_needed=num_needed)
                    # Defaults to smallest oligo to break ties (make a setting?)
                    if p[2] > p_best[2]:
                        p_best = p
                except CannotConstructOligoError:
                    continue
            if p_best[0] is None:
                return None
            return p_best

        # Only memoize results if this is being computed for a sufficiently
        # high fraction of sequences; for cases where seqs_to_consider
        # represents a small fraction of all sequences, we are less likely
        # to re-encounter the same seqs_to_consider in the future (so
        # there is little need to memoize the result) and the call to
        # construct_oligo() should be relatively quick (so it is ok to have
        # to repeat the call if we do re-encounter the same seqs_to_consider)
        num_seqs_to_consider = sum(len(v) for k, v in seqs_to_consider.items())
        frac_seqs_to_consider = float(num_seqs_to_consider) / self.aln.num_sequences
        should_memoize = frac_seqs_to_consider >= memoize_threshold

        if not should_memoize or self.do_not_memoize:
            # Construct the oligos and return them; do not memoize the result
            p = construct_p()
            return p

        def make_key():
            # Make a key for hashing
            # Make frozen version of both dicts; note that values in
            # seqs_to_consider may be sets that need to be frozen
            seqs_to_consider_frozen = set()
            for k, v in seqs_to_consider.items():
                # Compress the sequence indices, which, in many cases, are
                # mostly contiguous
                v_compressed = index_compress.compress_mostly_contiguous(v)
                seqs_to_consider_frozen.add((k, frozenset(v_compressed)))
            seqs_to_consider_frozen = frozenset(seqs_to_consider_frozen)
            if num_needed is None:
                num_needed_frozen = None
            else:
                num_needed_frozen = frozenset(num_needed.items())

            key = (seqs_to_consider_frozen, num_needed_frozen)
            return key

        p = super()._compute_memoized(start, construct_p, make_key,
                                      use_last=use_last)
        return p

    def obj_value(self, oligo_set):
        """Compute objective value for a oligo set.

        This is just the number of oligos, which we seek to minimize.

        Args:
            oligo_set: set of oligo sequences

        Returns:
            number of oligos
        """
        return float(len(oligo_set))

    def best_obj_value(self):
        """Return the best possible objective value (or a rough estimate).

        Returns:
            float
        """
        # The best objective value occurs when there is 1 oligo.
        return 1.0

    def _find_optimal_oligo_in_window(self, start, end, seqs_to_consider,
            num_needed):
        """Find the oligo that hybridizes to the most sequences in a given window.

        This considers each position within the specified window at which a oligo
        can start. At each, it determines the optimal oligo (i.e., attempting to cover
        the most number of sequences) as well as the number of sequences that the
        oligo covers (hybridizes to). It selects the oligo that covers the most. This
        breaks ties arbitrarily.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            seqs_to_consider: dict mapping universe group ID to collection of
                indices of sequences to use when selecting a oligo
            num_needed: dict mapping universe group ID to the number of
                sequences from the group that are left to cover in order to
                achieve the desired (partial) cover

        Returns:
            tuple (w, x, y, z) where:
                w is the sequence of the selected oligo
                x is a collection of indices of sequences (a subset of
                    sequence IDs in seqs_to_consider) to which oligo w will
                    hybridize
                y is the starting position of w in the alignment
                z is a score representing the amount of the remaining
                    universe that w covers
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        # Calculate the end of the search (exclusive), which is the last
        # position in the window at which a oligo can start; a oligo needs to
        # fit completely within the window
        search_end = end - self.min_oligo_length + 1

        called_construct_oligo = False
        max_oligo_cover = (None, set(), None, 0)
        for pos in range(start, search_end):
            if self._overlaps_ignored_range(pos):
                # oligo starting at pos would overlap an ignored range,
                # so skip this oligo
                p = None
            else:
                # After the first call to self._construct_memoized in
                # this window, seqs_to_consider and num_needed will all be
                # the same as the first call, so tell the function to
                # take advantage of this (by avoiding hashing these
                # dicts)
                use_last = pos > start and called_construct_oligo

                p = self._construct_memoized(pos, seqs_to_consider, num_needed,
                    use_last=use_last)
                called_construct_oligo = True

            if p is not None:
                # There is a suitable oligo at pos
                olg, covered_seqs, score = p

                if max_oligo_cover is None:
                    max_oligo_cover = (olg, covered_seqs, pos, score)
                else:
                    if score > max_oligo_cover[3]:
                        # olg has the highest score
                        max_oligo_cover = (olg, covered_seqs, pos, score)
        return max_oligo_cover

    def _find_oligos_in_window(self, start, end, only_consider=None):
        """Find a collection of oligos that cover sequences in a given window.

        This attempts to find the smallest number of oligos such that, within
        the specified window, at least the fraction self.cover_frac of
        sequences have a oligo that hybridizes to it.

        The solution is based on approximating the solution to an instance
        of the set cover problem by using the canonical greedy algorithm.
        The following is a description of the problem and solution to the
        most basic case, in which sets are unweighted and we seek to cover
        the entire universe:
        We are given are universe U of (hashable) objects and a collection
        of m subsets S_1, S_2, ..., S_m of U whose union equals U. We wish
        to approximate the smallest number of these subets whose union is U
        (i.e., "covers" the universe U). Pseudocode of the greedy algorithm
        is as follows:
          C <- {}
          while the universe is not covered (U =/= {}):
            Pick the set S_i that covers the most of U (i.e., maximizes
              | S_i _intersection_ U |)
            C <- C _union_ { S_i }
            U <- U - S_i
          return C
        The collection of subsets C is the approximate set cover and a
        valid set cover is always returned. The loop goes through min(|U|,m)
        iterations and picking S_i at each iteration takes O(m*D) time where
        D is the cardinality of the largest subset. The returned solution
        is a ceil(ln(D))-approximation. In the worst-case, where D=|U|, this
        is a ceil(ln(|U|))-approximation. Inapproximability results show
        that it is NP-hard to approximate the problem to within c*ln(|U|)
        for any 0 < c < 1. Thus, this is a very good approximation given
        what is possible.

        Here, we generalize the above case to support a partial set cover;
        rather than looping while the universe is not covered, we instead
        loop until we have covered as much as the universe as we desire to
        cover (determined by a parameter indicating the fraction of the
        universe to cover). The paper "Improved performance of the greedy
        algorithm for the minimum set cover and minimum partial cover
        problems" (Petr Slavik) provides guarantees on the approximation to
        the partial cover. We select a set S_i on each iteration as the one
        that maximizes min(r, S_i \intersect U) where r is the number of
        elements left to cover to achieve the desired partial cover.

        We also generalize this to support a "multi-universe" -- i.e.,
        where the universe is divided into groups and we want to achieve
        a particular partial cover for each group. For example, each group
        may encompass sequences from a particular year. We iterate until
        we have covered as much of each group in the universe as we desire
        to cover. We select a set S_i on each iteration as the one that
        maximizes as score, where the score is calculated by summing across
        each group g the value min(r_g, S_i \intersect U_g) where r_g
        is the number of elements left to cover in g to achieve the desired
        partial cover, and U_g is the universe left to cover for group g.

        In this problem, the universe U is the set of all sequences (i.e.,
        each element represents a sequence). Each subset S_i of U represents
        a possible oligo and the elements of S_i are integers representing
        the sequences to which the oligo hybridizes (i.e., the sequences that
        the oligo "covers"). When approximating the solution, we do not need
        to actually construct all possible oligos (or subsets S_i). Instead,
        on the first iteration we construct the oligo p_1 that hybridizes to
        (covers) the most number of sequences; S_1 is then the sequences that
        p_1 covers. We include p_1 in the output and subtract all of the
        sequences in S_1 from U. On the next iteration, we construct the
        oligo that covers the most number of sequences that remain in U, and
        so on.

        Although it is not done here, we could choose to assign each oligo
        a cost (e.g., based on its sequence composition) and then select the
        oligos that have the smallest total cost while achieving the desired
        coverage of sequences. This would be a weighted set cover problem.
        Without assigning costs, we can think of each oligo as having a cost
        of 1.0; in this case, we simply select the smallest number of oligos
        that achieve the desired coverage.

        The collection of covering oligos (C, above) is initialized with the
        oligos in self.required_oligos that fall within the given window,
        and the universe is initialized accordingly.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            only_consider: set giving list of sequence IDs (index in
                alignment) from which to construct universe -- i.e.,
                only consider these sequences. The desired coverage
                (self.cover_frac) is achieved only for the sequences
                in this set. If None (default), consider all sequences

        Returns:
            collection of str representing oligo sequences that were selected
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        # Create the universe, which is all the input sequences for each
        # group
        universe = {}
        for group_id, seq_ids in self.seq_groups.items():
            universe[group_id] = set(seq_ids)
            if only_consider is not None:
                universe[group_id] = universe[group_id] & only_consider

        num_that_can_be_uncovered = {}
        num_left_to_cover = {}
        for group_id, seq_ids in universe.items():
            num_that_can_be_uncovered[group_id] = int(len(seq_ids) -
                self.cover_frac[group_id] * len(seq_ids))
            # Above, use int(..) to take the floor. Also, expand out
            # len(seq_ids) rather than use
            # int((1.0-self.cover_frac[.])*len(seq_ids)) due to precision
            # errors in Python -- e.g., int((1.0-0.8)*5) yields 0
            # on some machines.

            num_left_to_cover[group_id] = (len(seq_ids) -
                num_that_can_be_uncovered[group_id])

        oligos_in_cover = set()
        def add_oligo_to_cover(olg, olg_covered_seqs, olg_pos):
            # The set representing olg goes into the set cover, and all of the
            # sequences it hybridizes to are removed from their group in the
            # universe
            logger.debug(("Adding oligo '%s' (at %d) to cover; it covers "
                "%d sequences") % (olg, olg_pos, len(olg_covered_seqs)))
            logger.debug(("Before adding, there are %s left to cover "
                "per-group") % ([num_left_to_cover[gid] for gid in universe.keys()]))

            oligos_in_cover.add(olg)
            for group_id in universe.keys():
                universe[group_id].difference_update(olg_covered_seqs)
                num_left_to_cover[group_id] = max(0,
                    len(universe[group_id]) - num_that_can_be_uncovered[group_id])
            # Save the position of this oligo in case the oligo needs to be
            # revisited
            self._selected_positions[olg].add(olg_pos)

            logger.debug(("After adding, there are %s left to cover "
                "per-group") % ([num_left_to_cover[gid] for gid in universe.keys()]))

        # Place all oligos from self.required_oligos that fall within this
        # window into oligos_in_cover
        logger.debug("Adding required covers to cover")
        for olg, olg_pos in self.required_oligos.items():
            if (olg_pos < start or
                    olg_pos + len(olg) > end):
                # olg is not fully within this window
                continue
            # Find the sequences in the alignment that are bound by olg
            r = (olg, olg_pos + len(olg) - 1)
            if r in self._memoized_seqs_covered_by_required_oligos:
                olg_covered_seqs = self._memoized_seqs_covered_by_required_oligos[r]
            else:
                # Determine which sequences are bound by olg, and memoize
                # them
                olg_covered_seqs = self.aln.sequences_bound_by_oligo(
                    olg, olg_pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs)
                if len(olg_covered_seqs) == 0:
                    # olg covers no sequences at olg_pos; still initialize with it
                    # but give a warning
                    logger.warning(("Oligo '%s' at position %d does not cover "
                        "any sequences but is being required in the cover") %
                        (olg, olg_pos))
                if self.do_not_memoize:
                    # Skip memoization
                    continue
                else:
                    self._memoized_seqs_covered_by_required_oligos[r] = olg_covered_seqs
            if only_consider is not None:
                # Only cover the sequences that should be considered
                olg_covered_seqs = olg_covered_seqs & only_consider
            # Add olg to the cover, and update the universe
            add_oligo_to_cover(olg, olg_covered_seqs, olg_pos)

        # Keep iterating until desired partial cover is obtained for all
        # groups
        logger.debug(("Iterating to achieve coverage; universe has %s "
            "elements per-group, with %s that can be uncovered per-group") %
            ([len(universe[gid]) for gid in universe.keys()],
             [num_that_can_be_uncovered for gid in universe.keys()]))
        while [True for group_id in universe.keys()
               if num_left_to_cover[group_id] > 0]:
            # Find the oligo that hybridizes to the most sequences, among
            # those that are not in the cover
            olg, olg_covered_seqs, olg_pos, olg_score = self._find_optimal_oligo_in_window(
                start, end, universe, num_left_to_cover)

            if olg is None or len(olg_covered_seqs) == 0:
                # No suitable oligos could be constructed within the window
                raise CannotAchieveDesiredCoverageError(("No suitable oligos "
                    "could be constructed in the window [%d, %d), but "
                    "more are needed to achieve desired coverage") %
                    (start, end))

            # Add olg to the set cover
            add_oligo_to_cover(olg, olg_covered_seqs, olg_pos)

        return oligos_in_cover

    def _score_collection(self, oligos):
        """Calculate a score representing how redundant oligos are in covering
        target genomes.

        Many windows may have minimal oligo designs that have the same
        number of oligos, and it can be difficult to pick between these.
        For a set of oligos S, this calculates a score that represents
        the redundancy of S so that more redundant ("better") sets of
        oligos receive a higher score. The objective is to assign a higher
        score to sets of oligos that cover genomes with multiple oligos
        and/or in which many of the oligos cover multiple genomes. A lower
        score should go to sets of oligos in which oligos only cover
        one genome (or a small number).

        Because this is loosely defined, we use a crude heuristic to
        calculate this score. For a set of oligos S, the score is the
        average fraction of sequences that need to be covered (as specified
        by cover_frac) that are covered by oligos in S, where the average
        is taken over the oligos. That is, it is the sum of the fraction of
        needed sequences covered by each oligo in S divided by the size
        of S. The score is a value in [0, 1].

        The score is meant to be compared across sets of oligos that
        are the same size (i.e., have the same number of oligos). It
        is not necessarily useful for comparing across sets of oligos
        that differ in size.

        Args:
            oligos: collection of str representing oligo sequences

        Returns:
            score of oligos, as defined above
        """
        # For each group, calculate the number of sequences in the group
        # that ought to be covered and also store the seq_ids as a set
        num_needed_to_cover_in_group = {}
        total_num_needed_to_cover = 0
        for group_id, seq_ids in self.seq_groups.items():
            num_needed = math.ceil(self.cover_frac[group_id] * len(seq_ids))
            num_needed_to_cover_in_group[group_id] = (num_needed, set(seq_ids))
            total_num_needed_to_cover += num_needed

        # For each oligo olg_seq, calculate the fraction of sequences that
        # need to be covered that are covered by olg_seq
        sum_of_frac_of_seqs_bound = 0
        for olg_seq in oligos:
            # Determine all the sequences covered by olg_seq
            seqs_bound = set()
            for pos in self._selected_positions[olg_seq]:
                seqs_bound.update(self.aln.sequences_bound_by_oligo(olg_seq,
                    pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs))

            # For each group, find the number of sequences that need to
            # be covered that are covered by olg_seq, and sum these over
            # all the groups
            total_num_covered = 0
            for group_id in num_needed_to_cover_in_group.keys():
                num_needed, seqs_in_group = num_needed_to_cover_in_group[group_id]
                covered_in_group = seqs_bound & seqs_in_group
                num_covered = min(num_needed, len(covered_in_group))
                total_num_covered += num_covered

            # Calculate the fraction of sequences that need to be covered
            # (total_num_needed_to_cover) that are covered by olg_seq
            # (total_num_covered)
            frac_bound = float(total_num_covered) / total_num_needed_to_cover
            sum_of_frac_of_seqs_bound += frac_bound

        score = sum_of_frac_of_seqs_bound / float(len(oligos))
        return score

    def _seqs_bound(self, oligos):
        """Determine the sequences in the alignment bound by the oligos.

        Args:
            oligos: collection of str representing oligo sequences

        Returns:
            set of sequence identifiers (index in alignment) bound by
            a oligo
        """
        seqs_bound = set()
        for olg_seq in oligos:
            # Determine all sequences covered by olg_seq
            for pos in self._selected_positions[olg_seq]:
                seqs_bound.update(self.aln.sequences_bound_by_oligo(olg_seq,
                    pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs))
        return seqs_bound

    def total_frac_bound(self, oligos):
        """Calculate the total fraction of sequences in the alignment
        bound by the oligos.

        Note that if the sequences are grouped (e.g., by year), this
        might be small because many sequences might be from a group
        (e.g., year) with a low desired coverage.

        Args:
            oligos: collection of str representing oligo sequences

        Returns:
            total fraction of all sequences bound by a oligo
        """
        seqs_bound = self._seqs_bound(oligos)
        frac_bound = float(len(seqs_bound)) / self.aln.num_sequences
        return frac_bound

    def construct_oligo(self, start, oligo_length, seqs_to_consider,
            num_needed=None):
        """Construct a single oligo to target a set of sequences

        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses of OligoSearcherMinimizeGuides "
                                  "must implement construct_oligo.")


class OligoSearcherMaximizeActivity(OligoSearcher):
    """Methods to maximize expected activity of the oligo set.

    This is a base class, with subclasses defining methods depending on the
    oligo. It should not be used without subclassing, as it does not define
    all the positional arguments necessary for search.OligoSearcher.
    """

    def __init__(self, soft_constraint, hard_constraint, penalty_strength,
            algorithm='random-greedy', **kwargs):
        """
        Args:
            soft_constraint: number of oligos for the soft constraint
            hard_constraint: number of oligos for the hard constraint
            penalty_strength: coefficient in front of the soft penalty term
                (i.e., its importance relative to expected activity)
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design oligos overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            algorithm: 'greedy' or 'random-greedy'; 'greedy' is the
                canonical greedy algorithm (Nemhauser 1978) for constrained
                monotone submodular maximization, which may perform
                well in practice but has poor theoretical guarantees here
                (the function is not monotone); 'random-greedy' is the
                randomized greedy algorithm (Buchbinder 2014) for
                constrained non-monotone submodular maximization that has
                good worst-case theoretical guarantees
            kwargs: see OligoSearcher.__init__()
        """
        if (soft_constraint < 1 or
                hard_constraint < soft_constraint):
            raise ValueError(("soft_constraint must be >=1 and "
                              "hard_constraint must be >= soft_constraint"))

        if penalty_strength < 0:
            raise ValueError("penalty_strength must be >= 0")

        if 'predictor' not in kwargs or kwargs['predictor'] is None:
            raise Exception(("predictor must be specified to __init__() "
                             "in order to maximize expected activity"))

        if algorithm not in ['greedy', 'random-greedy']:
            raise ValueError(("algorithm must be 'greedy' or "
                              "'random-greedy'"))

        self.soft_constraint = soft_constraint
        self.hard_constraint = hard_constraint
        self.penalty_strength = penalty_strength
        self.algorithm = algorithm

        # In addition to memoizing oligo computations at each site,
        # memoize the ground set of oligos at each site
        self._memoized_ground_sets = {}

        super().__init__(**kwargs)

    def _obj_value_from_params(self, expected_activity, num_oligos):
        """Compute value of objective function from parameter values.

        The objective value for a oligo set G is:
          F(G) - max(0, |G| - h)
        where F(G) is the expected activity of G across the target sequences
        and h is a soft constraint on the number of oligos. The right-hand
        term represents a penalty for the soft constraint.

        Args:
            expected_activity: expected activity of a oligo set
            num_oligos: number of oligos in the oligo set

        Returns:
            value of objective function
        """
        if num_oligos > self.hard_constraint:
            raise Exception(("Objective being computed when hard constraint "
                "is not met"))
        return expected_activity - self.penalty_strength * max(0, num_oligos -
                self.soft_constraint)

    def obj_value(self, window_start, window_end, oligo_set, activities=None):
        """Compute value of objective function from oligo set in window.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            oligo_set: set of strings representing oligo sequences that
                have been selected to be in a oligo set
            activities: output of self.oligo_set_activities(); if not set,
                this calls that function

        Returns:
            value of objective function
        """
        if activities is None:
            activities = self.oligo_set_activities(window_start, window_end,
                    oligo_set)

        # Use the mean (i.e., uniform prior over target sequences)
        expected_activity = np.mean(activities)

        num_oligos = len(oligo_set)

        return self._obj_value_from_params(expected_activity, num_oligos)

    def best_obj_value(self):
        """Return the best possible objective value (or a rough estimate).

        Returns:
            float
        """
        # The highest objective value occurs when expected activity is
        # at its maximum (which occurs when there is maximal detection for
        # all sequences) and the number of oligos is 1
        return self._obj_value_from_params(self.predictor.rough_max_activity,
                1)

    def _find_optimal_oligo_in_window(self, start, end, curr_oligo_set,
            curr_activities):
        """Select a oligo from the ground set in the given window based on
        its marginal contribution.

        When algorithm is 'greedy', this is a oligo with the maximal marginal
        contribution. When the algorithm is 'random-greedy', this is one of the
        oligos, selected uniformly at random, from those with the highest
        marginal contributions.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            curr_oligo_set: current oligo set; needed to determine what oligos
                in the ground set can be added, and their marginal
                contributions
            curr_activities: list of activities between curr_oligo_set
                and the target sequences (one per target sequence)

        Returns:
            tuple (g, p) where g is a oligo sequence str and p is the position)
            of g in the alignment; or None if no oligo is selected (only the
            case when the algorithm is 'random-greedy'

        Raises:
            CannotFindPositiveMarginalContributionError if no oligo gives
            a positive marginal contribution to the objective (i.e., all hurt
            it). This is only raised when the algorithm is 'greedy'; it
            can also be the case when the algorithm is 'random-greedy', but
            in that case this returns None (or can raise the error if
            all sites are ignored)
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        search_start = start
        search_end = end
        # If there's a predictor, it needs enough context on each end.
        if self.predictor is not None:
            min_search_start = 0
            max_search_end = self.aln.seq_length
            if isinstance(self.predictor,
                    predict_activity.SimpleBinaryPredictor):
                # If it's a SimpleBinaryPredictor, use the length of the
                # flanking sequences as the length of necessary contexts
                if self.predictor.required_flanking_seqs[0] is not None:
                    min_search_start = len(self.predictor.required_flanking_seqs[0])
                if self.predictor.required_flanking_seqs[1] is not None:
                    max_search_end = (self.aln.seq_length-
                        len(self.predictor.required_flanking_seqs[1]))
            else:
                min_search_start = self.predictor.context_nt
                max_search_end = self.aln.seq_length-self.predictor.context_nt
            search_start = max(min_search_start, search_start)
            search_end = min(max_search_end, search_end)
        # Calculate the end of the search (exclusive), which is the last
        # position in the window at which a oligo can start; a oligo needs to
        # fit completely within the window
        search_end += -self.min_oligo_length + 1

        curr_expected_activity = np.mean(curr_activities)
        curr_num_oligos = len(curr_oligo_set)
        curr_obj = self._obj_value_from_params(curr_expected_activity,
                curr_num_oligos)
        def marginal_contribution(expected_activity_with_new_oligo):
            # Compute a marginal contribution after adding a new oligo
            # Note that if this is slow, we may be able to skip calling
            # this function and just select the oligo(s) with the highest
            # new objective value(s) because the curr_obj is fixed, so
            # it should not affect which oligo(s) have the highest
            # marginal contribution(s)
            new_obj = self._obj_value_from_params(
                    expected_activity_with_new_oligo, curr_num_oligos + 1)
            return new_obj - curr_obj

        called_analyze_oligos = False
        possible_oligos_to_select = []
        for pos in range(search_start, search_end):
            if self._overlaps_ignored_range(pos):
                # oligo starting at pos would overlap a ignored range,
                # so skip this site
                continue
            else:
                # After the first call to self._analyze_oligos_memoized in
                # this window, curr_activities will stay the same as the first
                # call, so tell the function to take advantage of this (by
                # avoiding hashing this list again)
                use_last = pos > start and called_analyze_oligos

                p = self._analyze_oligos_memoized(pos, curr_activities,
                        use_last=use_last)
                called_analyze_oligos = True

                # Go over each oligo in the ground set at pos, and
                # skip ones already in the oligo set
                for new_oligo, new_expected_activity in p.items():
                    if new_oligo in curr_oligo_set:
                        continue
                    mc = marginal_contribution(new_expected_activity)
                    possible_oligos_to_select += [(mc, new_oligo, pos)]

        if len(possible_oligos_to_select) == 0:
            # All sites are ignored in this window
            raise CannotFindPositiveMarginalContributionError(("There are "
                "no possible oligos; most likely all sites are set to be "
                "ignored or the ground set is empty (no potential oligos are "
                "suitable)"))

        if self.algorithm == 'greedy':
            # We only want the single oligo with the greatest marginal
            # contribution; optimize for this case
            r = max(possible_oligos_to_select, key=lambda x: x[0])
            if r[0] <= 0:
                # The marginal contribution of the best choice is 0 or negative
                raise CannotFindPositiveMarginalContributionError(("There "
                    "are no suitable oligos to select; all their marginal "
                    "contributions are non-positive"))
            else:
                return (r[1], r[2])
        elif self.algorithm == 'random-greedy':
            # We want a set M, with |M| <= self.hard_constraint,
            # that maximizes \sum_{u \in M} (marginal contribution
            # from adding u)
            # We can get this by taking the self.hard_constraint
            # oligos with the largest marginal contributions (just the
            # positive ones)
            M = sorted(possible_oligos_to_select, key=lambda x: x[0],
                    reverse=True)[:self.hard_constraint]
            M = [x for x in M if x[0] > 0]

            # With probability 1-|M|/self.hard_constraint, do not
            # select a oligo. This accounts for the possibility that
            # some of the best oligos have negative marginal contribution
            # (i.e., |M| < self.hard_constraint)
            if random.random() < 1.0 - float(len(M))/self.hard_constraint:
                return None

            # Choose an element from M uniformly at random
            assert len(M) > 0
            r = random.choice(M)
            return (r[1], r[2])

    def _find_oligos_in_window(self, start, end):
        """Find a collection of oligos that maximizes expected activity in
        a given window.

        This attempts to find a oligo set that maximizes subject to a
        penalty for the number of oligos and subject to a hard constraint
        on the number of oligos. We treat this is a constrained
        submodular maximization problem.

        Let d(g, s) be the predicted activity in oligo g detecting sequence
        s. Then, for a oligo set G, we define
          d(G, s) = max_{g \in G} d(g, s).
        That is, d(G, s) is the activity of the best oligo in detecting s.

        We want to find a oligo set G that maximizes:
          Ftilde(G) = F(G) - L*max(0, |G| - h)  subject to  |G| <= H
        where h represents a soft constraint on |G| and H >= h represents
        a hard constraint on |G|. F(G) is the expected activity of G
        in detecting all target sequences; here, we weight target
        sequences uniformly, so this is the mean of d(G, s) across the
        s in the target sequences (self.aln). L represents an importance
        on the penalty. Here: self.penalty_strength is L,
        self.soft_constraint is h, and self.hard_constraint is H.

        Ftilde(G) is submodular but is not monotone. It can be made to be
        non-negative by restricting the ground set. The classical discrete
        greedy algorithm (Nemhauser et al. 1978) assumes a monotone function so
        in general it does not apply. Nevertheless, it may provide good results
        in practice despite bad worst-case performance. When self.algorithm is
        'greedy', this method implements that algorithm: it chooses a oligo
        that provides the greatest marginal contribution at each iteration.
        Note that, if L=0, then Ftilde(G) is monotone and the classical greedy
        algorithm ('greedy') is likely the best choice.

        In general, we use the randomized greedy algorithm in Buchbinder et al.
        2014. We start with a ground set of oligos and then greedily select
        oligos, with randomization, that are among ones with the highest
        marginal contribution. This works as follows:
          C <- {ground set of oligos}
          C <- C _union_ {2*h 'dummy' elements that contribute 0}
          G <- {}
          for i in 1..H:
            M_i <- H elements from C\G that maximize \sum_{u \in M_i} (
                    Ftilde(G _union {u}) - Ftilde(G))
            g* <- element from M_i chosen uniformly at random
            G <- G _union_ {g*}
          G <- G \ {dummy elements}
          return G
        The collection of oligos G is the approximately optimal oligo set.
        This provides a 1/e-approximation when Ftilde(G) is non-monotone
        and 1-1/e when Ftilde(G) is monotone.

        An equivalent way to look at the above algorithm, and simpler to
        implement, is the way we implement it here:
          C <- {ground set of oligos}
          G <- {}
          for i in 1..H:
            M_i <- H elements from C\G that maximize \sum_{u \in M_i} (
                    Ftilde(G _union {u}) - Ftilde(G))
            With proability (1-|M_i|/H), select no oligo and continue
            Otherwise g* <- element from M_i chosen uniformly at random
            G <- G _union_ {g*}
          return G

        The oligo set (G, above) is initialized with the oligos in
        self.required_oligos that fall within the given window.

        Args:
            start/end: boundaries of the window; the window spans [start, end)

        Returns:
            collection of str representing oligo sequences that were selected
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        # Initialize an empty oligo set, with 0 activity against each
        # target sequence
        curr_oligo_set = set()
        curr_activities = np.zeros(self.aln.num_sequences)

        def add_oligo(olg, olg_pos, olg_activities):
            # Add olg into curr_oligo_set and update curr_activities
            # to reflect the activities of olg
            logger.debug(("Adding oligo '%s' (at %d) to oligo set"),
                    olg, olg_pos)

            # Add the oligo to the current oligo set
            assert olg not in curr_oligo_set
            curr_oligo_set.add(olg)

            # Update curr_activities
            nonlocal curr_activities
            curr_activities = self._activities_after_adding_oligo(
                    curr_activities, olg_activities)

            # Save the position of the oligo in case the oligo needs to be
            # revisited
            self._selected_positions[olg].add(olg_pos)

        # Place all oligos from self.required_oligos that fall within
        # this window into curr_oligo_set
        for olg, olg_pos in self.required_oligos.items():
            if (olg_pos < start or
                    olg_pos + len(olg) > end):
                # olg is not fully within this window
                continue
            # Predict activities of olg against each target sequence
            olg_activities = self.aln.compute_activity(
                    olg_pos, olg, self.predictor)
            logger.info(("Adding required oligo '%s' to the oligo set; it "
                "has mean activity %f over the targets"), olg,
                np.mean(olg_activities))
            add_oligo(olg, olg_pos, olg_activities)

        # At each iteration of the greedy algorithm (random or deterministic),
        # we choose 1 oligo (possibly 0, for the random algorithm) and can
        # stop early; therefore, iterate up to self.hard_constraint times
        logger.debug(("Iterating to construct the oligo set"))
        for i in range(self.hard_constraint):
            # Pick a oligo
            try:
                p = self._find_optimal_oligo_in_window(
                        start, end,
                        curr_oligo_set, curr_activities)
                if p is None:
                    # The random greedy algorithm did not choose a oligo;
                    # skip this
                    continue
                new_olg, olg_pos = p
            except CannotFindPositiveMarginalContributionError:
                # No oligo adds marginal contribution
                # This should only happen in the greedy case (or if
                # all sites are ignored), and the objective function
                # is such that its value would continue to decrease if
                # we were to continue (i.e., it would continue to be
                # the case that no oligos give a positive marginal
                # contribution) -- so stop early
                logger.debug(("At iteration where no oligo adds marginal "
                    "contribution; stopping early"))
                break

            # new_oligo is from the ground set, so we can get its
            # activities across target sequences easily
            new_oligo_activities = self._ground_set_with_activities_memoized(
                    olg_pos)[new_olg]

            add_oligo(new_olg, olg_pos, new_oligo_activities)
            logger.debug(("There are currently %d oligos in the oligo set"),
                    len(curr_oligo_set))

        if len(curr_oligo_set) == 0:
            # It is possible no oligos can be found (e.g., if they all
            # create negative objective values)
            if self.algorithm == 'random-greedy':
                logger.debug(("No oligos could be found, possibly because "
                    "the ground set in this window is empty. 'random-greedy' "
                    "restricts the ground set so that the objective function "
                    "is non-negative. This is more likely to be the case "
                    "if penalty_strength is relatively high and/or the "
                    "difference (hard_constraint - soft_constraint) "
                    "is large. Trying the 'greedy' maximization "
                    "algorithm may avoid this."))
            raise CannotFindAnyOligosError(("No oligos could be constructed "
                "in the window [%d, %d)") % (start, end))

        return curr_oligo_set

    def total_frac_bound(self, window_start, window_end, oligo_set,
            activities=None):
        """Calculate the total fraction of sequences in the alignment
        bound by the oligos.

        This assumes that a sequence is 'bound' if the activity against
        it is >0; otherwise, it is assumed it is not 'bound'. This
        is reasonable because an activity=0 is the result of being
        decided by the classification model to be not active.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            oligo_set: set of strings representing oligo sequences that
                have been selected to be in a oligo set
            activities: output of self.oligo_set_activities(); if not set,
                this calls that function

        Returns:
            total fraction of all sequences detected by a oligo
        """
        if activities is None:
            activities = self.oligo_set_activities(window_start, window_end,
                    oligo_set)

        # Calculate fraction of activity values >0
        num_active = sum(1 for x in activities if x > 0)

        return float(num_active) / len(activities)

    def _ground_set_with_activities_memoized(self, start):
        """Make a memoized call to determine a ground set of oligos at a site.

        This also computes the activity of each oligo in the ground set
        against all sequences in the alignment at its site.

        The 'random-greedy' algorithm has theoretical guarantees on the
        worst-case output, assuming the objective function is non-negative. To
        keep those guarantees and meet the assumption, we only allow oligos in
        the ground set that ensure the objective is non-negative. The objective
        is Ftilde(G) = F(G) - L*max(0, |G| - h) subject to |G| <= H, where F(G)
        is the expected activity of oligo set G, L is self.penalty_strength, h
        is self.soft_constraint, and H is self.hard_constraint. Let
        G only contain oligos g such that F({g}) >= L*(H - h). F(G) is
        monotonically increasing, so then F(G) >= L*(H - h). Thus:
          Ftilde(G) =  F(G) - L*max(0, |G| - h)
                    >= L*(H - h) - L*max(0, |G| - h)
        If |G| <= h, then:
          Ftilde(G) >= L*(H - h) - 0
                    >= 0    (since H >= h)
        If |G| > h, then:
          Ftilde(G) >= L*(H - h) - L*(|G| - h)
                    =  L*(H - h - |G| + h)
                    =  L*(H - |G|)
                    >= 0    (since H >= |G|)
        Therefore, Ftilde(G), our objective function, is >= 0 always. So, to
        enforce, non-negativity, we can restrict our ground set to only contain
        oligos g whose expected activity is >= L*(H - h). In other words, every
        oligo has to be sufficiently good.

        Ftilde(G) = F(G) - \lambda*max(0, |G| - m_g) subject to |G| <=
        \overline{m_g}. We could restrict our domain for G to only contain
        oligos {g} such that F({g}) >= \lambda*(\overline{m_g} - m_g). (Put
        another way, any oligo has to be sufficiently good.) Then Ftilde(G) =
        F(G) - \lambda*max(0, |G| - m_g) >= \lambda*(\overline{m_g} - m_g) -
        \lambda*max(0, |G| - m_g) >= \lambda*(\overline{m_g} - m_g - (|G| -
        m_g)) = \lambda*(\overline{m_g} - m_g) >= 0. So Ftilde(G) is
        non-negative.

        Args:
            start: start position in alignment at which to make ground set

        Returns:
            dict {g: a} where g is a oligo sequence str and a
            is a numpy array giving activities against each sequence
            in the alignment
        """

        if start in self._memoized_ground_sets:
            # Ground set with activities is already memoized
            return self._memoized_ground_sets[start]

        # Determine a threshold for each oligo in the ground set that
        # enforces non-negativity of the objective
        nonnegativity_threshold = (self.penalty_strength *
            (self.hard_constraint - self.soft_constraint))

        # Compute ground set; do so considering all sequences
        seqs_to_consider = {0: set(range(self.aln.num_sequences))}
        dist_to_end = (self.aln.seq_length - start)
        max_oligo_length = dist_to_end if self.max_oligo_length > dist_to_end \
            else self.max_oligo_length
        try:
            ground_set = self.aln.determine_representative_oligos(
                    start, max_oligo_length, seqs_to_consider,
                    self.clusterer, missing_threshold=self.missing_threshold,
                    is_suitable_fns=self.is_suitable_fns,
                    required_flanking_seqs=self.required_flanking_seqs)
        except CannotConstructOligoError:
            # There may be too much missing data or a related issue
            # at this site; do not have a ground set here
            ground_set = set()
        # Predict activity against all target sequences for each oligo
        # in ground set
        ground_set_with_activities = {}
        for olg_seq in ground_set:
            best_seq = None
            best_activities = None
            best_expected_activity = -math.inf
            for oligo_length in range(self.min_oligo_length, len(olg_seq)+1):
                short_olg_seq = olg_seq[:oligo_length]
                try:
                    activities = self.aln.compute_activity(start, short_olg_seq,
                            self.predictor)
                except CannotConstructOligoError:
                    # Most likely this site is too close to an endpoint
                    # and does not have enough context_nt -- there will be
                    # no ground set at this site -- but skip this oligo and
                    # try others anyway
                    continue
                expected_activity = np.mean(activities)
                if self.algorithm == 'random-greedy':
                    # Restrict the ground set to only contain oligos that
                    # are sufficiently good, as described above
                    # That is, their expected activity has to exceed the
                    # threshold described above
                    # This ensures that the objective is non-negative

                    if expected_activity < nonnegativity_threshold:
                        # Oligo does not exceed threshold; ignore it
                        continue
                # Chooses smallest oligo when tied; make a setting?
                if expected_activity > best_expected_activity:
                    best_seq = olg_seq[:oligo_length]
                    best_activities = activities
                    best_expected_activity = expected_activity
            if best_seq is not None:
                ground_set_with_activities[best_seq] = best_activities

        if not self.do_not_memoize:
            # Memoize it
            self._memoized_ground_sets[start] = ground_set_with_activities

        return ground_set_with_activities

    def _cleanup_memoized_ground_sets(self, start):
        """Remove a position that is stored in self._memoized_ground_sets.

        This should be called when the position is no longer needed.

        Args:
            start: start position that no longer needs to be memoized
        """
        if start in self._memoized_ground_sets:
            del self._memoized_ground_sets[start]

    def _activities_after_adding_oligo(self, curr_activities,
            new_oligo_activities):
        """Compute activities after adding a oligo to a oligo set.

        Let S be the set of target sequences (not just in the piece
        at start, but through an entire window). Let G be the
        current oligo set, where the predicted activity of G against
        sequences s_i \in S is curr_activities[i]. That is,
        curr_activities[i] = max_{g \in G} d(g, s_i)
        where d(g, s_i) is the predicted activity of g detecting s_i.
        Now we add a oligo x, from the ground set at the position
        start, into G. Consider sequence s_i. The activity in detecting
        s_i is:
             max_{g \in (G U {x})} d(g, s_i)
           = max[max_{g \in G} d(g, s_i), d(x, s_i)]
           = max[curr_activities[i], d(x, s_i)]
        We can easily calculate this for all sequences s_i by taking
        an element-wise maximum between curr_activities and an array
        giving the activity between x and each target sequence s_i.

        Args:
            curr_activities: list of activities across the target
                sequences yielded by the current oligo set
            new_oligo_activities: list of activities across the target
                sequences for a single new oligo

        Returns:
            list of activities across the target sequences yielded by
            the new oligo added to the current oligo set
        """
        assert len(new_oligo_activities) == len(curr_activities)
        curr_activities_with_new = np.maximum(curr_activities,
                new_oligo_activities)
        return curr_activities_with_new

    def _analyze_oligos(self, start, curr_activities):
        """Compute activities after adding in each ground set oligo to the
        oligo set.

        Args:
            start: start position in alignment at which to target
            curr_activities: list of activities across the target
                sequences yielded by the current oligo set

        Returns:
            dict {g: a} where g is a oligo in the ground set at start
            and a is the expected activity, taken across the target
            sequences, of (current ground set U {g})
        """
        expected_activities = {}
        ground_set_with_activities = self._ground_set_with_activities_memoized(
                start)
        for x, activities_for_x in ground_set_with_activities.items():
            curr_activities_with_x = self._activities_after_adding_oligo(
                    curr_activities, activities_for_x)
            expected_activities[x] = np.mean(curr_activities_with_x)
        return expected_activities

    def _analyze_oligos_memoized(self, start, curr_activities,
        use_last=False):
        """Make a memoized call to self._analyze_oligos().

        Args:
            start: start position in alignment at which to target
            curr_activities: list of activities across the target
                sequences yielded by the current oligo set
            use_last: if set, check for a memoized result by using the last
                key constructed (it is assumed that curr_activities is
                identical to the last provided value)

        Returns:
            result of self._analyze_oligos()
        """
        def analyze():
            return self._analyze_oligos(start, curr_activities)

        # TODO: If memory usage is high, consider not memoizing when
        # the current oligo set is large; similar to the strategy in
        # OligoSearcherMinimizeNumber._construct_memoized()

        if self.do_not_memoize:
            # Analyze the oligos and return that; do not memoize the result
            return analyze()

        def make_key():
            # Make a key for hashing
            # curr_activities is a numpy array; hash(x.data.tobytes())
            # works when x is a numpy array
            key = curr_activities.data.tobytes()
            return key

        # The output of analyze() is already dense, so do not compress
        p = super()._compute_memoized(start, analyze, make_key,
                use_last=use_last)
        return p


class CannotAchieveDesiredCoverageError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class CannotFindPositiveMarginalContributionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NoPredictorError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class CannotConstructOligoError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class CannotFindAnyOligosError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
