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

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class GuideSearcher:
    """Methods to search for guides to use for a diagnostic.

    This is a base class, with subclasses defining methods depending on
    whether the problem is to minimize the number of guides or to
    maximize expected activity.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable guides.
    """

    def __init__(self, aln, guide_length, missing_data_params,
                 guide_is_suitable_fn=None,
                 required_guides={}, blacklisted_ranges={},
                 allow_gu_pairs=False, required_flanking_seqs=(None, None),
                 do_not_memoize_guides=False,
                 predictor=None):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            guide_is_suitable_fn: if set, the value of this argument is a
                function f(x) such that this will only construct a guide x
                for which f(x) is True
            required_guides: dict that maps guide sequences to their position
                in the alignment; all of these guide sequences are immediately
                placed in the set of covering guides for their appropriate
                windows before finding other guides, so that they are
                guaranteed to be in the output (i.e., the set of covering
                guides is initialized with these guides)
            blacklisted_ranges: set of tuples (start, end) that provide
                ranges in the alignment from which guides should not be
                constructed. No guide that might overlap these ranges is
                constructed. Note that start is inclusive and end is
                exclusive.
            allow_gu_pairs: if True, tolerate G-U base pairs between a
                guide and target when computing whether a guide binds
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the guide (in the target, not the guide) that must be
                present for a guide to bind; if either is None, no
                flanking sequence is required for that end
            do_not_memoize_guides: if True, never memoize the results of
                computed guides at a site and always compute the guides (this
                can be useful if we know the memoized result will never be used
                and memoizing it may be slow, or if we want to benchmark
                performance with/without memoization)
            predictor: adapt.utils.predict_activity.Predictor object. If
                None, do not predict activities.
        """
        self.aln = aln
        self.guide_length = guide_length

        # Because calls to compute guides at a site are expensive and are
        # repeated very often, memoize the output
        self._memoized_guides = defaultdict(dict)
        self._memoized_guides_last_inner_dict = None
        self._memoized_guides_last_inner_dict_key = None
        self._memoized_guides_num_removed_since_last_resize = 0
        self.do_not_memoize_guides = do_not_memoize_guides

        # Save the positions of selected guides in the alignment so these can
        # be easily revisited. In case a guide sequence appears in multiple
        # places, store a set of positions
        self._selected_guide_positions = defaultdict(set)

        # Determine a threshold at which to ignore sites with too much
        # missing data
        missing_max, missing_min, missing_coeff = missing_data_params
        self.missing_threshold = min(missing_max, max(missing_min,
            missing_coeff * self.aln.median_sequences_with_missing_data()))

        self.guide_is_suitable_fn = guide_is_suitable_fn

        self.required_guides = required_guides

        # Verify positions of the guides are within the alignment
        highest_possible_gd_pos = self.aln.seq_length - self.guide_length
        for gd, gd_pos in self.required_guides.items():
            if gd_pos < 0 or gd_pos > highest_possible_gd_pos:
                raise Exception(("A guide with sequence '%s' at position %d "
                    "is required to be in the output, but does not fall "
                    "within the alignment") % (gd, gd_pos))

        # Because determining which sequences are covered by each required
        # guide is expensive and will be done repeatedly on the same guide,
        # memoize them
        self._memoized_seqs_covered_by_required_guides = {}

        self.blacklisted_ranges = blacklisted_ranges

        # Verify blacklisted ranges are within the alignment
        for start, end in blacklisted_ranges:
            if start < 0 or end <= start or end > self.aln.seq_length:
                raise Exception(("A blacklisted range [%d, %d) is invalid "
                    "for a given alignment; ranges must fall within the "
                    "alignment: [0, %d)") % (start, end, self.aln.seq_length))

        self.allow_gu_pairs = allow_gu_pairs

        self.required_flanking_seqs = required_flanking_seqs

        self.guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=min(10, int(guide_length/2)))

        self.predictor = predictor

    def _compute_guide_memoized(self, start, call_fn, key_fn,
            use_last=False, compress=True):
        """Make a memoized call to compute a guide.

        The actual computation is defined in a subclass and passed as
        a function (call_fn) here -- it can be to construct a guide or
        to compute information (e.g., expected activity) about different
        guides in a ground set.

        Args:
            start: start position in alignment at which to target
            call_fn: function to call for compute guide(s)
            key_fn: function call to construct a key from current state
            use_last: if set, check for a memoized result by using the last
                key constructed (it is assumed that key is identical to
                the last provided values)
            compress: if True, compress the output of call_fn() before
                memoizing

        Returns:
            result of call_fn()
        """
        if use_last:
            # key (defined below) can be large and slow to hash; therefore,
            # assume that the last computed key is identical to the one that
            # would be computed here (i.e., call_state is the same), and use
            # the last inner dict to avoid having to hash key
            assert self._memoized_guides_last_inner_dict is not None
            inner_dict = self._memoized_guides_last_inner_dict
        else:
            key = key_fn()
            if key in self._memoized_guides:
                inner_dict = self._memoized_guides[key]
            else:
                inner_dict = {}
                self._memoized_guides[key] = inner_dict
            self._memoized_guides_last_inner_dict_key = key

        if start in inner_dict:
            # The result has been memoized

            p_memoized = inner_dict[start]

            # p was compressed before memoizing it; decompress
            # it before returning it
            if p_memoized is None:
                p = None
            else:
                if compress:
                    p = self._decompress_compute_guide_result(p_memoized)
                else:
                    p = p_memoized
        else:
            # The result is not memoized; compute it and memoize it

            p = call_fn()

            if p is None:
                p_to_memoize = None
            else:
                if compress:
                    # Compress p before memoizing it
                    p_to_memoize = self._compress_compute_guide_result(p)
                else:
                    p_to_memoize = p

            inner_dict[start] = p_to_memoize

        self._memoized_guides_last_inner_dict = inner_dict
        return p

    def _cleanup_memoized_guides(self, pos, frac_removed_until_resize=0.1):
        """Remove a position that is stored in self._memoized_guides.

        This should be called when the position no longer needs to be stored.

        Python only resizes dicts on insertions (and, seemingly, only after
        reaching a sufficiently large size); see
        https://github.com/python/cpython/blob/master/Objects/dictnotes.txt
        In this case, it may never resize or resize too infrequently,
        especially for the inner dicts. It appears in many cases to never
        resize. Therefore, this "resizes" the self._memoized_guides dict at
        certain cleanups by copying all the content over to a new dict,
        effectively forcing it to shrink its memory usage. It does this
        by computing the number of elements that have been removed from the
        data structure relative to its current total number of elements,
        and resizing when this fraction exceeds `frac_removed_until_resize`.
        Since the total number of elements should stay roughly constant
        as we scan along the alignment (i.e., pos increases), this fraction
        should grow over time. At each resizing, the fraction will drop back
        down to 0.

        This also cleans up memoizations in the predictor, if that was set.

        Args:
            pos: start position that no longer needs to be memoized (i.e., where
                guides covering at that start position are no longer needed)
            frac_removed_until_resize: resize the self._memoized_guides
                data structure when the total number of elements removed
                due to cleanup exceeds this fraction of the total size
        """
        keys_to_rm = set()
        for key in self._memoized_guides.keys():
            if pos in self._memoized_guides[key]:
                del self._memoized_guides[key][pos]
                self._memoized_guides_num_removed_since_last_resize += 1
            if len(self._memoized_guides[key]) == 0:
                keys_to_rm.add(key)

        for key in keys_to_rm:
            del self._memoized_guides[key]

        # Decide whether to resize
        total_size = sum(len(self._memoized_guides[k])
                for k in self._memoized_guides.keys())
        if total_size > 0:
            frac_removed = float(self._memoized_guides_num_removed_since_last_resize) / total_size
            logger.debug(("Deciding to resize with a fraction %f removed "
                "(%d / %d)"), frac_removed,
                self._memoized_guides_num_removed_since_last_resize,
                total_size)
            if frac_removed >= frac_removed_until_resize:
                # Resize self._memoized_guides by copying all content to a new dict
                new_memoized_guides = defaultdict(dict)
                for key in self._memoized_guides.keys():
                    for i in self._memoized_guides[key].keys():
                        new_memoized_guides[key][i] = self._memoized_guides[key][i]
                    if key == self._memoized_guides_last_inner_dict_key:
                        self._memoized_guides_last_inner_dict = new_memoized_guides[key]
                self._memoized_guides = new_memoized_guides
                self._memoized_guides_num_removed_since_last_resize = 0

        # Cleanup the predictor's memoizations at this position
        if self.predictor is not None:
            self.predictor.cleanup_memoized(pos)

    def _guide_overlaps_blacklisted_range(self, gd_pos):
        """Determine whether a guide would overlap a blacklisted range.

        The blacklisted ranges are given in self.blacklisted_ranges.

        Args:
            gd_pos: start position of a guide

        Returns:
            True iff the guide overlaps a blacklisted range
        """
        gd_end = gd_pos + self.guide_length - 1
        for start, end in self.blacklisted_ranges:
            if ((gd_pos >= start and gd_pos < end) or
                    (gd_end >= start and gd_end < end)):
                return True
        return False

    def guide_set_activities(self, window_start, window_end, guide_set):
        """Compute activity across target sequences for guide set in a window.

        Let S be the set of target sequences. Let G be guide_set, and
        let the predicted activity of a guide g in detecting s_i \in S
        be d(g, s_i). Then the activity of G in detecting s_i is
        max_{g \in G} d(g, s_i). We can compute an activity for G
        against all sequences s_i by repeatedly taking element-wise
        maxima; see
        GuideSearcherMaximizeActivity._activities_after_adding_guide() for
        an explanation.

        Note that if guide_set is a subset of the ground set, then
        we already have these computed in GuideSearcherMaximizeAcitivity.
        Re-implementing it here lets us use have them with
        GuideSearcherMinimizeGuides, and is also not a bad check
        to re-compute.

        This assumes that guide positions for each guide in guide_set
        are stored in self._selected_guide_positions.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            guide_set: set of strings representing guide sequences that
                have been selected to be in a guide set

        Returns:
            list of activities across the target sequences (self.aln)
            yielded by guide_set
        """
        if self.predictor is None:
            raise NoPredictorError(("Cannot compute activities when "
                "predictor is not set"))

        activities = np.zeros(self.aln.num_sequences)
        for gd_seq in guide_set:
            if gd_seq not in self._selected_guide_positions:
                raise Exception(("Guide must be selected and its position "
                    "saved"))

            # The guide could hit multiple places
            for start in self._selected_guide_positions[gd_seq]:
                if start < window_start or start > window_end - len(gd_seq):
                    # Guide is outside window
                    continue
                try:
                    gd_activities = self.aln.compute_activity(start, gd_seq,
                            self.predictor)
                except alignment.CannotConstructGuideError:
                    # Most likely this site is too close to an endpoint and
                    # does not have enough context_nt; skip it
                    continue

                # Update activities with gd_activities
                activities = np.maximum(activities, gd_activities)

        return activities

    def guide_set_activities_percentile(self, window_start, window_end,
            guide_set, q, activities=None):
        """Compute percentiles of activity across target sequences for
        a guide set in a window.

        For example, when percentiles is 0.5, this returns the median
        activity across the target sequences that the guide set provides.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            guide_set: set of strings representing guide sequences that
                have been selected to be in a guide set
            q: list of percentiles to compute, each in [0,100]
                (0 is minimum, 100 is maximum)
            activities: output of self.guide_set_activities(); if not set,
                this calls that function

        Returns:
            list of percentile values
        """
        if activities is None:
            activities = self.guide_set_activities(window_start, window_end,
                    guide_set)

        # Do not interpolate (choose lower value)
        p = np.percentile(activities, q, interpolation='lower')
        return list(p)

    def guide_set_activities_expected_value(self, window_start, window_end,
            guide_set, activities=None):
        """Compute expected activity across target sequences for
        a guide set in a window.

        This assumes the distribution across target sequences is uniform,
        so it is equivalent to the mean.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            guide_set: set of strings representing guide sequences that
                have been selected to be in a guide set
            activities: output of self.guide_set_activities(); if not set,
                this calls that function

        Returns:
            expected (here, mean) activity
        """
        if activities is None:
            activities = self.guide_set_activities(window_start, window_end,
                    guide_set)

        return np.mean(activities)

    def guide_activities_expected_value(self, window_start, window_end, gd_seq):
        """Compute expected activity across target sequences for a single
        guide in a window.

        Let S be the set of target sequences. Then the activity of a guide
        g in detecting s_i \in S be d(g, s_i). We assume a uniform
        distribution over the s_i, so the expected value for g is the
        mean of d(g, s_i) across the s_i.

        Note that if gd_seq is in the ground set, then we already have these
        computed in GuideSearcherMaximizeAcitivity. Re-implementing it here
        lets us use have them with GuideSearcherMinimizeGuides, and is also not
        a bad check to re-compute.

        This assumes that the guide position for gd_seq is stored in
        self._selected_guide_positions.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            gd_seq: string representing guide sequence

        Returns:
            mean activity across the target sequences (self.aln) yielded
            by gd_seq
        """
        if self.predictor is None:
            raise NoPredictorError(("Cannot compute activities when "
                "predictor is not set"))

        if gd_seq not in self._selected_guide_positions:
            raise Exception(("Guide must be selected and its position "
                "saved"))

        # The guide could hit multiple places; account for that just in case
        activities = np.zeros(self.aln.num_sequences)
        for start in self._selected_guide_positions[gd_seq]:
            if start < window_start or start > window_end - len(gd_seq):
                # Guide is outside window
                continue
            try:
                gd_activities = self.aln.compute_activity(start, gd_seq,
                        self.predictor)
            except alignment.CannotConstructGuideError:
                # Most likely this site is too close to an endpoint and
                # does not have enough context_nt; skip it
                continue

            activities = np.maximum(activities, gd_activities)

        return np.mean(activities)

    def _find_guides_for_each_window(self, window_size,
            window_step=1, hide_warnings=False):
        """Find a collection of guides in each window.

        This runs a sliding window across the aligned sequences and, in each
        window, computes a guide set by calling self._find_guides_in_window().

        This returns guides for each window.

        This does not return guides for windows where it cannot design
        guides in the window (e.g., due to indels or ambiguity).

        Args:
            window_size: length of the window to use when sliding across
                alignment
            window_step: amount by which to increase the window start for
                every search
            hide_warnings: when set, this does not provide log warnings
                when no more suitable guides can be constructed

        Returns:
            yields x_i in which each x_i corresponds to a window;
            x_i is a tuple consisting of the following values, in order:
              1) start position of the window
              2) end position of the window
              3) set of guides for the window
        """
        if window_size > self.aln.seq_length:
            raise ValueError(("window size must be < the length of the "
                              "alignment"))
        if window_size < self.guide_length:
            raise ValueError("window size must be >= guide length") 

        for start in range(0, self.aln.seq_length - window_size + 1,
                window_step):
            end = start + window_size
            logger.info("Searching for guides within window [%d, %d)" %
                        (start, end))

            try:
                guides = self._find_guides_in_window(start, end)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                if not hide_warnings:
                    logger.warning(("No more suitable guides could be constructed "
                        "in the window [%d, %d), but more are needed to "
                        "achieve the desired coverage") % (start, end))
                self._cleanup_memoized_guides(start)
                continue
            except CannotFindAnyGuidesError:
                # Cannot find any guides in this window; log and skip it
                if not hide_warnings:
                    logger.warning(("No suitable guides could be constructed "
                        "in the window [%d, %d)") % (start, end))
                self._cleanup_memoized_guides(start)
                continue

            yield (start, end, guides)

            # We no longer need to memoize results for guides that start at
            # this position
            self._cleanup_memoized_guides(start)


class GuideSearcherMinimizeGuides(GuideSearcher):
    """Methods to minimize the number of guides.
    """

    def __init__(self, aln, guide_length, mismatches, cover_frac,
            missing_data_params, seq_groups=None, **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a guide; see seq_groups
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            seq_groups: dict that maps group ID to collection of sequences in
                that group. If set, cover_frac must also be a dict that maps
                group ID to the fraction of sequences in that group that
                must be 'captured' by a guide. If None, then do not divide
                the sequences into groups.
            kwargs: see GuideSearcher.__init__()
        """
        super().__init__(aln, guide_length, missing_data_params, **kwargs)

        if seq_groups is None and (cover_frac <= 0 or cover_frac > 1):
            raise ValueError("cover_frac must be in (0,1]")

        if seq_groups is not None:
            # Check that each group has a valid cover fraction
            for group_id in seq_groups.keys():
                assert group_id in cover_frac
                if cover_frac[group_id] <= 0 or cover_frac[group_id] > 1:
                    raise ValueError(("cover_frac for group %d must be "
                        "in (0,1]") % group_id)

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

        self.mismatches = mismatches

    def _compress_compute_guide_result(self, p):
        """Compress the result of alignment.Alignment.construct_guide().

        Args:
            p: result of calling construct_guide()

        Returns:
            compressed version of p
        """
        gd, covered_seqs = p

        # covered_seqs may contain mostly contiguous indices
        covered_seqs_compressed = index_compress.compress_mostly_contiguous(covered_seqs)

        return (gd, covered_seqs_compressed)

    def _decompress_compute_guide_result(self, p_compressed):
        """Decompress the compressed version of an output of construct_guide().

        Args:
            p_compressed: output of _compress_construct_guide_result()

        Returns:
            decompressed version of p_compressed
        """
        gd, covered_seqs_compressed = p_compressed

        # Decompress covered_seqs
        covered_seqs = index_compress.decompress_ranges(covered_seqs_compressed)

        return (gd, covered_seqs)

    def _construct_guide_memoized(self, start, seqs_to_consider,
            num_needed=None, use_last=False, memoize_threshold=0.1):
        """Make a memoized call to alignment.Alignment.construct_guide().

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: dict mapping universe group ID to collection
                of indices to use when constructing the guide
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
            result of alignment.Alignment.construct_guide()
        """
        def construct_p():
            try:
                p = self.aln.construct_guide(start, self.guide_length,
                        seqs_to_consider, self.mismatches, self.allow_gu_pairs,
                        self.guide_clusterer, num_needed=num_needed,
                        missing_threshold=self.missing_threshold,
                        guide_is_suitable_fn=self.guide_is_suitable_fn,
                        required_flanking_seqs=self.required_flanking_seqs,
                        predictor=self.predictor)
            except alignment.CannotConstructGuideError:
                p = None
            return p

        # Only memoize results if this is being computed for a sufficiently
        # high fraction of sequences; for cases where seqs_to_consider
        # represents a small fraction of all sequences, we are less likely
        # to re-encounter the same seqs_to_consider in the future (so
        # there is little need to memoize the result) and the call to
        # construct_guide() should be relatively quick (so it is ok to have
        # to repeat the call if we do re-encounter the same seqs_to_consider)
        num_seqs_to_consider = sum(len(v) for k, v in seqs_to_consider.items())
        frac_seqs_to_consider = float(num_seqs_to_consider) / self.aln.num_sequences
        should_memoize = frac_seqs_to_consider >= memoize_threshold

        if not should_memoize or self.do_not_memoize_guides:
            # Construct the guide and return it; do not memoize the result
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
        
        p = super()._compute_guide_memoized(start, construct_p, make_key,
                use_last=use_last)
        return p

    def obj_value(self, guide_set):
        """Compute objective value for a guide set.

        This is just the number of guides, which we seek to minimize.

        Args:
            guide_set: set of guide sequences

        Returns:
            number of guides
        """
        return float(len(guide_set))

    def best_obj_value(self):
        """Return the best possible objective value (or a rough estimate).

        Returns:
            float
        """
        # The best objective value occurs when there is 1 guide.
        return 1.0

    def _find_optimal_guide_in_window(self, start, end, seqs_to_consider,
            num_needed):
        """Find the guide that hybridizes to the most sequences in a given window.

        This considers each position within the specified window at which a guide
        can start. At each, it determines the optimal guide (i.e., attempting to cover
        the most number of sequences) as well as the number of sequences that the
        guide covers (hybridizes to). It selects the guide that covers the most. This
        breaks ties arbitrarily.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            seqs_to_consider: dict mapping universe group ID to collection of
                indices of sequences to use when selecting a guide
            num_needed: dict mapping universe group ID to the number of
                sequences from the group that are left to cover in order to
                achieve the desired (partial) cover

        Returns:
            tuple (w, x, y, z) where:
                w is the sequence of the selected guide
                x is a collection of indices of sequences (a subset of
                    sequence IDs in seqs_to_consider) to which guide w will
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
        # position in the window at which a guide can start; a guide needs to
        # fit completely within the window
        search_end = end - self.guide_length + 1

        called_construct_guide = False
        max_guide_cover = None
        for pos in range(start, search_end):
            if self._guide_overlaps_blacklisted_range(pos):
                # guide starting at pos would overlap a blacklisted range,
                # so skip this guide
                p = None
            else:
                # After the first call to self._construct_guide_memoized in
                # this window, seqs_to_consider and num_needed will all be
                # the same as the first call, so tell the function to
                # take advantage of this (by avoiding hashing these
                # dicts)
                use_last = pos > start and called_construct_guide

                p = self._construct_guide_memoized(pos, seqs_to_consider,
                    num_needed, use_last=use_last)
                called_construct_guide = True

            if p is None:
                # There is no suitable guide at pos
                if max_guide_cover is None:
                    max_guide_cover = (None, set(), None, 0)
            else:
                gd, covered_seqs = p

                # Calculate a score for this guide based on the partial
                # coverage it achieves across the groups
                score = 0
                for group_id, needed in num_needed.items():
                    covered_in_group = covered_seqs & seqs_to_consider[group_id]

                    # needed is the number of elements that have yet to be
                    # covered in order to obtain the desired partial cover
                    # in group_id; there is no reason to favor a guide that
                    # covers more than needed
                    score += min(needed, len(covered_in_group))

                if max_guide_cover is None:
                    max_guide_cover = (gd, covered_seqs, pos, score)
                else:
                    if score > max_guide_cover[3]:
                        # gd has the highest score
                        max_guide_cover = (gd, covered_seqs, pos, score)
        return max_guide_cover

    def _find_guides_in_window(self, start, end,
            only_consider=None):
        """Find a collection of guides that cover sequences in a given window.

        This attempts to find the smallest number of guides such that, within
        the specified window, at least the fraction self.cover_frac of
        sequences have a guide that hybridizes to it.

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
        a possible guide and the elements of S_i are integers representing
        the sequences to which the guide hybridizes (i.e., the sequences that
        the guide "covers"). When approximating the solution, we do not need
        to actually construct all possible guides (or subsets S_i). Instead,
        on the first iteration we construct the guide p_1 that hybridizes to
        (covers) the most number of sequences; S_1 is then the sequences that
        p_1 covers. We include p_1 in the output and subtract all of the
        sequences in S_1 from U. On the next iteration, we construct the
        guide that covers the most number of sequences that remain in U, and
        so on.

        Although it is not done here, we could choose to assign each guide
        a cost (e.g., based on its sequence composition) and then select the
        guides that have the smallest total cost while achieving the desired
        coverage of sequences. This would be a weighted set cover problem.
        Without assigning costs, we can think of each guide as having a cost
        of 1.0; in this case, we simply select the smallest number of guides
        that achieve the desired coverage.

        The collection of covering guides (C, above) is initialized with the
        guides in self.required_guides that fall within the given window,
        and the universe is initialized accordingly.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            only_consider: set giving list of sequence IDs (index in
                alignment) from which to construct universe -- i.e.,
                only consider these sequences. The desired coverage
                (self.cover_frac) is achieved only for the sequences
                in this set. If None (default), consider all sequences

        Returns:
            collection of str representing guide sequences that were selected
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

        guides_in_cover = set()
        def add_guide_to_cover(gd, gd_covered_seqs, gd_pos):
            # The set representing gd goes into the set cover, and all of the
            # sequences it hybridizes to are removed from their group in the
            # universe
            logger.debug(("Adding guide '%s' (at %d) to cover; it covers "
                "%d sequences") % (gd, gd_pos, len(gd_covered_seqs)))
            logger.debug(("Before adding, there are %s left to cover "
                "per-group") % ([num_left_to_cover[gid] for gid in universe.keys()]))

            guides_in_cover.add(gd)
            for group_id in universe.keys():
                universe[group_id].difference_update(gd_covered_seqs)
                num_left_to_cover[group_id] = max(0,
                    len(universe[group_id]) - num_that_can_be_uncovered[group_id])
            # Save the position of this guide in case the guide needs to be
            # revisited
            self._selected_guide_positions[gd].add(gd_pos)

            logger.debug(("After adding, there are %s left to cover "
                "per-group") % ([num_left_to_cover[gid] for gid in universe.keys()]))

        # Place all guides from self.required_guides that fall within this
        # window into guides_in_cover
        logger.debug("Adding required covers to cover")
        for gd, gd_pos in self.required_guides.items():
            if (gd_pos < start or
                    gd_pos + self.guide_length > end):
                # gd is not fully within this window
                continue
            # Find the sequences in the alignment that are bound by gd
            r = (gd, gd_pos)
            if r in self._memoized_seqs_covered_by_required_guides:
                gd_covered_seqs = self._memoized_seqs_covered_by_required_guides[r]
            else:
                # Determine which sequences are bound by gd, and memoize
                # them
                gd_covered_seqs = self.aln.sequences_bound_by_guide(
                    gd, gd_pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs)
                if len(gd_covered_seqs) == 0:
                    # gd covers no sequences at gd_pos; still initialize with it
                    # but give a warning
                    logger.warning(("Guide '%s' at position %d does not cover "
                        "any sequences but is being required in the cover") %
                        (gd, gd_pos))
                if self.do_not_memoize_guides:
                    # Skip memoization
                    continue
                else:
                    self._memoized_seqs_covered_by_required_guides[r] = gd_covered_seqs
            if only_consider is not None:
                # Only cover the sequences that should be considered
                gd_covered_seqs = gd_covered_seqs & only_consider
            # Add gd to the cover, and update the universe
            add_guide_to_cover(gd, gd_covered_seqs, gd_pos)

        # Keep iterating until desired partial cover is obtained for all
        # groups
        logger.debug(("Iterating to achieve coverage; universe has %s "
            "elements per-group, with %s that can be uncovered per-group") %
            ([len(universe[gid]) for gid in universe.keys()],
             [num_that_can_be_uncovered for gid in universe.keys()]))
        while [True for group_id in universe.keys()
               if num_left_to_cover[group_id] > 0]:
            # Find the guide that hybridizes to the most sequences, among
            # those that are not in the cover
            gd, gd_covered_seqs, gd_pos, gd_score = self._find_optimal_guide_in_window(
                start, end, universe, num_left_to_cover)

            if gd is None or len(gd_covered_seqs) == 0:
                # No suitable guides could be constructed within the window
                raise CannotAchieveDesiredCoverageError(("No suitable guides "
                    "could be constructed in the window [%d, %d), but "
                    "more are needed to achieve desired coverage") %
                    (start, end))

            # Add gd to the set cover
            add_guide_to_cover(gd, gd_covered_seqs, gd_pos)

        return guides_in_cover

    def _score_collection_of_guides(self, guides):
        """Calculate a score representing how redundant guides are in covering
        target genomes.

        Many windows may have minimal guide designs that have the same
        number of guides, and it can be difficult to pick between these.
        For a set of guides S, this calculates a score that represents
        the redundancy of S so that more redundant ("better") sets of
        guides receive a higher score. The objective is to assign a higher
        score to sets of guides that cover genomes with multiple guides
        and/or in which many of the guides cover multiple genomes. A lower
        score should go to sets of guides in which guides only cover
        one genome (or a small number).

        Because this is loosely defined, we use a crude heuristic to
        calculate this score. For a set of guides S, the score is the
        average fraction of sequences that need to be covered (as specified
        by cover_frac) that are covered by guides in S, where the average
        is taken over the guides. That is, it is the sum of the fraction of
        needed sequences covered by each guide in S divided by the size
        of S. The score is a value in [0, 1].

        The score is meant to be compared across sets of guides that
        are the same size (i.e., have the same number of guides). It
        is not necessarily useful for comparing across sets of guides
        that differ in size.

        Args:
            guides: collection of str representing guide sequences

        Returns:
            score of guides, as defined above
        """
        # For each group, calculate the number of sequences in the group
        # that ought to be covered and also store the seq_ids as a set
        num_needed_to_cover_in_group = {}
        total_num_needed_to_cover = 0
        for group_id, seq_ids in self.seq_groups.items():
            num_needed = math.ceil(self.cover_frac[group_id] * len(seq_ids))
            num_needed_to_cover_in_group[group_id] = (num_needed, set(seq_ids))
            total_num_needed_to_cover += num_needed

        # For each guide gd_seq, calculate the fraction of sequences that
        # need to be covered that are covered by gd_seq
        sum_of_frac_of_seqs_bound = 0
        for gd_seq in guides:
            # Determine all the sequences covered by gd_seq
            seqs_bound = set()
            for pos in self._selected_guide_positions[gd_seq]:
                seqs_bound.update(self.aln.sequences_bound_by_guide(gd_seq,
                    pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs))

            # For each group, find the number of sequences that need to
            # be covered that are covered by gd_seq, and sum these over
            # all the groups
            total_num_covered = 0
            for group_id in num_needed_to_cover_in_group.keys():
                num_needed, seqs_in_group = num_needed_to_cover_in_group[group_id]
                covered_in_group = seqs_bound & seqs_in_group
                num_covered = min(num_needed, len(covered_in_group))
                total_num_covered += num_covered

            # Calculate the fraction of sequences that need to be covered
            # (total_num_needed_to_cover) that are covered by gd_seq
            # (total_num_covered)
            frac_bound = float(total_num_covered) / total_num_needed_to_cover
            sum_of_frac_of_seqs_bound += frac_bound

        score = sum_of_frac_of_seqs_bound / float(len(guides))
        return score

    def _seqs_bound_by_guides(self, guides):
        """Determine the sequences in the alignment bound by the guides.

        Args:
            guides: collection of str representing guide sequences

        Returns:
            set of sequence identifiers (index in alignment) bound by
            a guide
        """
        seqs_bound = set()
        for gd_seq in guides:
            # Determine all sequences covered by gd_seq
            for pos in self._selected_guide_positions[gd_seq]:
                seqs_bound.update(self.aln.sequences_bound_by_guide(gd_seq,
                    pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs))
        return seqs_bound

    def total_frac_bound_by_guides(self, guides):
        """Calculate the total fraction of sequences in the alignment
        bound by the guides.

        Note that if the sequences are grouped (e.g., by year), this
        might be small because many sequences might be from a group
        (e.g., year) with a low desired coverage.

        Args:
            guides: collection of str representing guide sequences

        Returns:
            total fraction of all sequences bound by a guide
        """
        seqs_bound = self._seqs_bound_by_guides(guides)
        frac_bound = float(len(seqs_bound)) / self.aln.num_sequences
        return frac_bound

    def find_guides_with_sliding_window(self, window_size, out_fn,
                               window_step=1, sort=False, print_analysis=True):
        """Find the smallest collection of guides that cover sequences, across
        all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            window_size: length of the window to use when sliding across
                alignment
            out_fn: output TSV file to write guide sequences by window
            window_step: amount by which to increase the window start for
                every search
            sort: if set, sort output TSV by number of guides (ascending)
                then by score (descending); when not set, default is to
                sort by window position
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the smallest number of guides and highest
                score
        """
        guide_collections = list(self._find_guides_for_each_window(
            window_size, window_step=window_step))

        if sort:
            # Sort by number of guides ascending (len(x[2])), then by
            # score of guides descending
            guide_collections.sort(key=lambda x: (len(x[2]),
                -self._score_collection_of_guides(x[2])))

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'score', 'total-frac-bound', 'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, guide_seqs = guides_in_window
                score = self._score_collection_of_guides(guide_seqs)
                frac_bound = self.total_frac_bound_by_guides(guide_seqs)
                count = len(guide_seqs)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_guide_positions[gd_seq]
                             for gd_seq in guide_seqs_sorted]
                positions_str = ' '.join(str(p) for p in positions)
                line = [start, end, count, score, frac_bound, guide_seqs_str,
                        positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')

        if print_analysis:
            num_windows_scanned = len(
                range(0, self.aln.seq_length - window_size + 1))
            num_windows_with_guides = len(guide_collections)

            if num_windows_with_guides == 0:
                stat_display = [
                    ("Number of windows scanned", num_windows_scanned),
                    ("Number of windows with guides", num_windows_with_guides)
                ]
            else:
                min_count = min(len(x[2]) for x in guide_collections)
                num_with_min_count = sum(1 for x in guide_collections
                    if len(x[2]) == min_count)

                min_count_str = (str(min_count) + " guide" + 
                                 ("s" if min_count > 1 else ""))

                stat_display = [
                    ("Number of windows scanned", num_windows_scanned),
                    ("Number of windows with guides", num_windows_with_guides),
                    ("Minimum number of guides required in a window", min_count),
                    ("Number of windows with " + min_count_str,
                        num_with_min_count),
                ]

            # Print the above statistics, with padding on the left
            # so that the statistic names are right-justified in a
            # column and the values line up, left-justified, in a column
            max_stat_name_len = max(len(name) for name, val in stat_display)
            for name, val in stat_display:
                pad_spaces = max_stat_name_len - len(name)
                name_padded = " "*pad_spaces + name + ":"
                print(name_padded, str(val))


class GuideSearcherMaximizeActivity(GuideSearcher):
    """Methods to maximize expected activity of the guide set.
    """

    def __init__(self, aln, guide_length, soft_guide_constraint,
            hard_guide_constraint, penalty_strength,
            missing_data_params, algorithm='random-greedy', **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            soft_guide_constraint: number of guides for the soft constraint
            hard_guide_constraint: number of guides for the hard constraint
            penalty_strength: coefficient in front of the soft penalty term
                (i.e., its importance relative to expected activity)
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
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
            kwargs: see GuideSearcher.__init__()
        """
        super().__init__(aln, guide_length, missing_data_params, **kwargs)

        if (soft_guide_constraint < 1 or
                hard_guide_constraint < soft_guide_constraint):
            raise ValueError(("soft_guide_constraint must be >=1 and "
                "hard_guide_constraint must be >= soft_guide_constraint"))

        if penalty_strength < 0:
            raise ValueError("penalty_strength must be >= 0")

        if 'predictor' not in kwargs or kwargs['predictor'] is None:
            raise Exception(("predictor must be specified to __init__() "
                "in order to maximize expected activity"))

        if algorithm not in ['greedy', 'random-greedy']:
            raise ValueError(("algorithm must be 'greedy' or "
                "'random-greedy'"))

        self.soft_guide_constraint = soft_guide_constraint
        self.hard_guide_constraint = hard_guide_constraint
        self.penalty_strength = penalty_strength
        self.algorithm = algorithm

        # In addition to memoizing guide computations at each site,
        # memoize the ground set of guides at each site
        self._memoized_ground_sets = {}

    def _obj_value_from_params(self, expected_activity, num_guides):
        """Compute value of objective function from parameter values.

        The objective value for a guide set G is:
          F(G) - max(0, |G| - h)
        where F(G) is the expected activity of G across the target sequences
        and h is a soft constraint on the number of guides. The right-hand
        term represents a penalty for the soft constraint.

        Args:
            expected_activity: expected activity of a guide set
            num_guides: number of guides in the guide set

        Returns:
            value of objective function
        """
        if num_guides > self.hard_guide_constraint:
            raise Exception(("Objective being computed when hard constraint "
                "is not met"))
        return expected_activity - self.penalty_strength * max(0, num_guides -
                self.soft_guide_constraint)

    def obj_value(self, window_start, window_end, guide_set, activities=None):
        """Compute value of objective function from guide set in window.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            guide_set: set of strings representing guide sequences that
                have been selected to be in a guide set
            activities: output of self.guide_set_activities(); if not set,
                this calls that function

        Returns:
            value of objective function
        """
        if activities is None:
            activities = self.guide_set_activities(window_start, window_end,
                    guide_set)
        
        # Use the mean (i.e., uniform prior over target sequences)
        expected_activity = np.mean(activities)

        num_guides = len(guide_set)

        return self._obj_value_from_params(expected_activity, num_guides)

    def best_obj_value(self):
        """Return the best possible objective value (or a rough estimate).

        Returns:
            float
        """
        # The highest objective value occurs when expected activity is
        # at its maximum (which occurs when there is maximal detection for
        # all sequences) and the number of guides is 1
        return self._obj_value_from_params(self.predictor.rough_max_activity,
                1)

    def total_frac_bound_by_guides(self, window_start, window_end, guide_set,
            activities=None):
        """Calculate the total fraction of sequences in the alignment
        bound by the guides.

        This assumes that a sequence is 'bound' if the activity against
        it is >0; otherwise, it is assumed it is not 'bound'. This
        is reasonable because an activity=0 is the result of being
        decided by the classification model to be not active.

        Args:
            window_start/window_end: start (inclusive) and end (exclusive)
                positions of the window
            guide_set: set of strings representing guide sequences that
                have been selected to be in a guide set
            activities: output of self.guide_set_activities(); if not set,
                this calls that function

        Returns:
            total fraction of all sequences detected by a guide
        """
        if activities is None:
            activities = self.guide_set_activities(window_start, window_end,
                    guide_set)

        # Calculate fraction of activity values >0
        num_active = sum(1 for x in activities if x > 0)

        return float(num_active) / len(activities)

    def _ground_set_with_activities_memoized(self, start):
        """Make a memoized call to determine a ground set of guides at a site.

        This also computes the activity of each guide in the ground set
        against all sequences in the alignment at its site.

        The 'random-greedy' algorithm has theoretical guarantees on the
        worst-case output, assuming the objective function is non-negative. To
        keep those guarantees and meet the assumption, we only allow guides in
        the ground set that ensure the objective is non-negative. The objective
        is Ftilde(G) = F(G) - L*max(0, |G| - h) subject to |G| <= H, where F(G)
        is the expected activity of guide set G, L is self.penalty_strength, h
        is self.soft_guide_constraint, and H is self.hard_guide_constraint. Let
        G only contain guides g such that F({g}) >= L*(H - h). F(G) is
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
        guides g whose expected activity is >= L*(H - h). In other words, every
        guide has to be sufficiently good.

        Ftilde(G) = F(G) - \lambda*max(0, |G| - m_g) subject to |G| <=
        \overline{m_g}. We could restrict our domain for G to only contain
        guides {g} such that F({g}) >= \lambda*(\overline{m_g} - m_g). (Put
        another way, any guide has to be sufficiently good.) Then Ftilde(G) =
        F(G) - \lambda*max(0, |G| - m_g) >= \lambda*(\overline{m_g} - m_g) -
        \lambda*max(0, |G| - m_g) >= \lambda*(\overline{m_g} - m_g - (|G| -
        m_g)) = \lambda*(\overline{m_g} - m_g) >= 0. So Ftilde(G) is
        non-negative.

        Args:
            start: start position in alignment at which to make ground set

        Returns:
            dict {g: a} where g is a guide sequence str and a
            is a numpy array giving activities against each sequence
            in the alignment
        """
        if start in self._memoized_ground_sets:
            # Ground set with activities is already memoized
            return self._memoized_ground_sets[start]

        # Determine a threshold for each guide in the ground set that
        # enforces non-negativity of the objective
        nonnegativity_threshold = (self.penalty_strength *
            (self.hard_guide_constraint - self.soft_guide_constraint))

        # Compute ground set; do so considering all sequences
        seqs_to_consider = {0: set(range(self.aln.num_sequences))}
        try:
            ground_set = self.aln.determine_representative_guides(
                    start, self.guide_length, seqs_to_consider,
                    self.guide_clusterer,
                    missing_threshold=self.missing_threshold,
                    guide_is_suitable_fn=self.guide_is_suitable_fn,
                    required_flanking_seqs=self.required_flanking_seqs)
        except alignment.CannotConstructGuideError:
            # There may be too much missing data or a related issue
            # at this site; do not have a ground set here
            ground_set = set()

        # Predict activity against all target sequences for each guide
        # in ground set
        ground_set_with_activities = {}
        for gd_seq in ground_set:
            try:
                activities = self.aln.compute_activity(start, gd_seq,
                        self.predictor)
            except alignment.CannotConstructGuideError:
                # Most likely this site is too close to an endpoint
                # and does not have enough context_nt -- there will be
                # no ground set at this site -- but skip this guide and
                # try others anyway
                continue
            if self.algorithm == 'random-greedy':
                # Restrict the ground set to only contain guides that
                # are sufficiently good, as described above
                # That is, their expected activity has to exceed the
                # threshold described above
                # This ensures that the objective is non-negative
                expected_activity = np.mean(activities)
                if expected_activity < nonnegativity_threshold:
                    # Guide does not exceed threshold; ignore it
                    continue

            ground_set_with_activities[gd_seq] = activities

        if self.do_not_memoize_guides:
            # Return the ground set and do *not* memoize it
            return ground_set_with_activities

        # Memoize it
        self._memoized_ground_sets[start] = ground_set_with_activities

        return ground_set_with_activities

    def _cleanup_memoized_ground_sets(self, pos):
        """Remove a position that is stored in self._memoized_ground_sets.

        This should be called when the position is no longer needed.

        Args:
            pos: start position that no longer needs to be memoized
        """
        if pos in self._memoized_ground_sets:
            del self._memoized_ground_sets[pos]

    def _activities_after_adding_guide(self, curr_activities,
            new_guide_activities):
        """Compute activities after adding a guide to a guide set.

        Let S be the set of target sequences (not just in the piece
        at start, but through an entire window). Let G be the
        current guide set, where the predicted activity of G against
        sequences s_i \in S is curr_activities[i]. That is,
        curr_activities[i] = max_{g \in G} d(g, s_i)
        where d(g, s_i) is the predicted activity of g detecting s_i.
        Now we add a guide x, from the ground set at the position
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
                sequences yielded by the current guide set
            new_guide_activities: list of activities across the target
                sequences for a single new guide

        Returns:
            list of activities across the target sequences yielded by
            the new guide added to the current guide set
        """
        assert len(new_guide_activities) == len(curr_activities)
        curr_activities_with_new = np.maximum(curr_activities,
                new_guide_activities)
        return curr_activities_with_new

    def _analyze_guides(self, start, curr_activities):
        """Compute activities after adding in each ground set guide to the
        guide set.

        Args:
            start: start position in alignment at which to target
            curr_activities: list of activities across the target
                sequences yielded by the current guide set

        Returns:
            dict {g: a} where g is a guide in the ground set at start
            and a is the expected activity, taken across the target
            sequences, of (current ground set U {g})
        """
        expected_activities = {}
        ground_set_with_activities = self._ground_set_with_activities_memoized(
                start)
        for x, activities_for_x in ground_set_with_activities.items():
            curr_activities_with_x = self._activities_after_adding_guide(
                    curr_activities, activities_for_x)
            expected_activities[x] = np.mean(curr_activities_with_x)
        return expected_activities

    def _analyze_guides_memoized(self, start, curr_activities,
            use_last=False):
        """Make a memoized call to self._analyze_guides().

        Args:
            start: start position in alignment at which to target
            curr_activities: list of activities across the target
                sequences yielded by the current guide set
            use_last: if set, check for a memoized result by using the last
                key constructed (it is assumed that curr_activities is
                identical to the last provided value)

        Returns:
            result of self._analyze_guides()
        """
        def analyze():
            return self._analyze_guides(start, curr_activities)

        # TODO: If memory usage is high, consider not memoizing when
        # the current guide set is large; similar to the strategy in
        # GuideSearcherMinimizeGuides._construct_guide_memoized()

        if self.do_not_memoize_guides:
            # Analyze the guides and return that; do not memoize the result
            return analyze()

        def make_key():
            # Make a key for hashing
            # curr_activities is a numpy array; hash(x.data.tobytes())
            # works when x is a numpy array
            key = curr_activities.data.tobytes()
            return key

        # The output of analyze() is already dense, so do not compress
        p = super()._compute_guide_memoized(start, analyze, make_key,
                use_last=use_last, compress=False)
        return p

    def _find_optimal_guide_in_window(self, start, end, curr_guide_set,
            curr_activities):
        """Select a guide from the ground set in the given window based on
        its marginal contribution.

        When algorithm is 'greedy', this is a guide with the maximal marginal
        contribution. When the algorithm is 'random-greedy', this is one of the
        guides, selected uniformly at random, from those with the highest
        marginal contributions.

        Args:
            start/end: boundaries of the window; the window spans [start, end)
            curr_guide_set: current guide set; needed to determine what guides
                in the ground set can be added, and their marginal
                contributions
            curr_activities: list of activities between curr_guide_set
                and the target sequences (one per target sequence)

        Returns:
            tuple (g, p) where g is a guide sequence str and p is the position)
            of g in the alignment; or None if no guide is selected (only the
            case when the algorithm is 'random-greedy'

        Raises:
            CannotFindPositiveMarginalContributionError if no guide gives
            a positive marginal contribution to the objective (i.e., all hurt
            it). This is only raised when the algorithm is 'greedy'; it
            can also be the case when the algorithm is 'random-greedy', but
            in that case this returns None (or can raise the error if
            all sites are blacklisted)
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        # Calculate the end of the search (exclusive), which is the last
        # position in the window at which a guide can start; a guide needs to
        # fit completely within the window
        search_end = end - self.guide_length + 1

        curr_expected_activity = np.mean(curr_activities)
        curr_num_guides = len(curr_guide_set)
        curr_obj = self._obj_value_from_params(curr_expected_activity,
                curr_num_guides)
        def marginal_contribution(expected_activity_with_new_guide):
            # Compute a marginal contribution after adding a new guide
            # Note that if this is slow, we may be able to skip calling
            # this function and just select the guide(s) with the highest
            # new objective value(s) because the curr_obj is fixed, so
            # it should not affect which guide(s) have the highest
            # marginal contribution(s)
            new_obj = self._obj_value_from_params(
                    expected_activity_with_new_guide, curr_num_guides + 1)
            return new_obj - curr_obj

        called_analyze_guides = False
        possible_guides_to_select = []
        for pos in range(start, search_end):
            if self._guide_overlaps_blacklisted_range(pos):
                # guide starting at pos would overlap a blacklisted range,
                # so skip this site
                continue
            else:
                # After the first call to self._analyze_guides_memoized in
                # this window, curr_activities will stay the same as the first
                # call, so tell the function to take advantage of this (by
                # avoiding hashing this list again)
                use_last = pos > start and called_analyze_guides

                p = self._analyze_guides_memoized(pos, curr_activities,
                        use_last=use_last)
                called_analyze_guides = True

                # Go over each guide in the ground set at pos, and
                # skip ones already in the guide set
                for new_guide, new_expected_activity in p.items():
                    if new_guide in curr_guide_set:
                        continue
                    mc = marginal_contribution(new_expected_activity)
                    possible_guides_to_select += [(mc, new_guide, pos)]

        if len(possible_guides_to_select) == 0:
            # All sites are blacklisted in this window
            raise CannotFindPositiveMarginalContributionError(("There are "
                "no possible guides; most likely all sites are blacklisted "
                "or the ground set is empty (no potential guides are "
                "suitable)"))

        if self.algorithm == 'greedy':
            # We only want the single guide with the greatest marginal
            # contribution; optimize for this case
            r = max(possible_guides_to_select, key=lambda x: x[0])
            if r[0] <= 0:
                # The marginal contribution of the best choice is 0 or negative
                raise CannotFindPositiveMarginalContributionError(("There "
                    "are no suitable guides to select; all their marginal "
                    "contributions are non-positive"))
            else:
                return (r[1], r[2])
        elif self.algorithm == 'random-greedy':
            # We want a set M, with |M| <= self.hard_guide_constraint,
            # that maximizes \sum_{u \in M} (marginal contribution
            # from adding u)
            # We can get this by taking the self.hard_guide_constraint
            # guides with the largest marginal contributions (just the
            # positive ones)
            M = sorted(possible_guides_to_select, key=lambda x: x[0],
                    reverse=True)[:self.hard_guide_constraint]
            M = [x for x in M if x[0] > 0]

            # With probability 1-|M|/self.hard_guide_constraint, do not
            # select a guide. This accounts for the possibility that
            # some of the best guides have negative marginal contribution
            # (i.e., |M| < self.hard_guide_constraint)
            if random.random() < 1.0 - float(len(M))/self.hard_guide_constraint:
                return None

            # Choose an element from M uniformly at random
            assert len(M) > 0
            r = random.choice(M)
            return (r[1], r[2])

    def _find_guides_in_window(self, start, end):
        """Find a collection of guides that maximizes expected activity in
        a given window.

        This attempts to find a guide set that maximizes subject to a
        penalty for the number of guides and subject to a hard constraint
        on the number of guides. We treat this is a constrained
        submodular maximization problem.

        Let d(g, s) be the predicted activity in guide g detecting sequence
        s. Then, for a guide set G, we define
          d(G, s) = max_{g \in G} d(g, s).
        That is, d(G, s) is the activity of the best guide in detecting s.

        We want to find a guide set G that maximizes:
          Ftilde(G) = F(G) - L*max(0, |G| - h)  subject to  |G| <= H
        where h represents a soft constraint on |G| and H >= h represents
        a hard constraint on |G|. F(G) is the expected activity of G
        in detecting all target sequences; here, we weight target
        sequences uniformly, so this is the mean of d(G, s) across the
        s in the target sequences (self.aln). L represents an importance
        on the penalty. Here: self.penalty_strength is L,
        self.soft_guide_constraint is h, and self.hard_guide_constraint is H.

        Ftilde(G) is submodular but is not monotone. It can be made to be
        non-negative by restricting the ground set. The classical discrete
        greedy algorithm (Nemhauser et al. 1978) assumes a monotone function so
        in general it does not apply. Nevertheless, it may provide good results
        in practice despite bad worst-case performance. When self.algorithm is
        'greedy', this method implements that algorithm: it chooses a guide
        that provides the greatest marginal contribution at each iteration.
        Note that, if L=0, then Ftilde(G) is monotone and the classical greedy
        algorithm ('greedy') is likely the best choice.

        In general, we use the randomized greedy algorithm in Buchbinder et al.
        2014. We start with a ground set of guides and then greedily select
        guides, with randomization, that are among ones with the highest
        marginal contribution. This works as follows:
          C <- {ground set of guides}
          C <- C _union_ {2*h 'dummy' elements that contribute 0}
          G <- {}
          for i in 1..H:
            M_i <- H elements from C\G that maximize \sum_{u \in M_i} (
                    Ftilde(G _union {u}) - Ftilde(G))
            g* <- element from M_i chosen uniformly at random
            G <- G _union_ {g*}
          G <- G \ {dummy elements}
          return G
        The collection of guides G is the approximately optimal guide set.
        This provides a 1/e-approximation when Ftilde(G) is non-monotone
        and 1-1/e when Ftilde(G) is monotone.

        An equivalent way to look at the above algorithm, and simpler to
        implement, is the way we implement it here:
          C <- {ground set of guides}
          G <- {}
          for i in 1..H:
            M_i <- H elements from C\G that maximize \sum_{u \in M_i} (
                    Ftilde(G _union {u}) - Ftilde(G))
            With proability (1-|M_i|/H), select no guide and continue
            Otherwise g* <- element from M_i chosen uniformly at random
            G <- G _union_ {g*}
          return G

        The guide set (G, above) is initialized with the guides in
        self.required_guides that fall within the given window.

        Args:
            start/end: boundaries of the window; the window spans [start, end)

        Returns:
            collection of str representing guide sequences that were selected
        """
        if start < 0:
            raise ValueError("window start must be >= 0")
        if end <= start:
            raise ValueError("window end must be > start")
        if end > self.aln.seq_length:
            raise ValueError("window end must be <= alignment length")

        # Initialize an empty guide set, with 0 activity against each
        # target sequence
        curr_guide_set = set()
        curr_activities = np.zeros(self.aln.num_sequences)

        def add_guide(gd, gd_pos, gd_activities):
            # Add gd into curr_guide_set and update curr_activities
            # to reflect the activities of gd
            logger.debug(("Adding guide '%s' (at %d) to guide set"),
                    gd, gd_pos)

            # Add the guide to the current guide set
            assert gd not in curr_guide_set
            curr_guide_set.add(gd)

            # Update curr_activities
            nonlocal curr_activities
            curr_activities = self._activities_after_adding_guide(
                    curr_activities, gd_activities)

            # Save the position of the guide in case the guide needs to be
            # revisited
            self._selected_guide_positions[gd].add(gd_pos)

        # Place all guides from self.required_guides that fall within
        # this window into curr_guide_set
        for gd, gd_pos in self.required_guides.items():
            if (gd_pos < start or
                    gd_pos + self.guide_length > end):
                # gd is not fully within this window
                continue
            # Predict activities of gd against each target sequence
            gd_activities = self.aln.compute_activity(
                    gd_pos, gd, self.predictor)
            logger.info(("Adding required guide '%s' to the guide set; it "
                "has mean activity %f over the targets"), gd,
                np.mean(gd_activities))
            add_guide(gd, gd_pos, gd_activities)

        # At each iteration of the greedy algorithm (random or deterministic),
        # we choose 1 guide (possibly 0, for the random algorithm) and can
        # stop early; therefore, iterate up to self.hard_guide_constraint times
        logger.debug(("Iterating to construct the guide set"))
        for i in range(self.hard_guide_constraint):
            # Pick a guide
            try:
                p = self._find_optimal_guide_in_window(
                        start, end,
                        curr_guide_set, curr_activities)
                if p is None:
                    # The random greedy algorithm did not choose a guide;
                    # skip this
                    continue
                new_gd, gd_pos = p
            except CannotFindPositiveMarginalContributionError:
                # No guide adds marginal contribution
                # This should only happen in the greedy case (or if
                # all sites are blacklisted), and the objective function
                # is such that its value would continue to decrease if
                # we were to continue (i.e., it would continue to be
                # the case that no guides give a positive marginal
                # contribution) -- so stop early
                logger.debug(("At iteration where no guide adds marginal "
                    "contribution; stopping early"))
                break

            # new_guide is from the ground set, so we can get its
            # activities across target sequences easily
            new_guide_activities = self._ground_set_with_activities_memoized(
                    gd_pos)[new_gd]

            add_guide(new_gd, gd_pos, new_guide_activities)
            logger.debug(("There are currently %d guides in the guide set"),
                    len(curr_guide_set))

        if len(curr_guide_set) == 0:
            # It is possible no guides can be found (e.g., if they all
            # create negative objective values)
            if self.algorithm == 'random-greedy':
                logger.warning(("No guides could be found, possibly because "
                    "the ground set in this window is empty. 'random-greedy' "
                    "restricts the ground set so that the objective function "
                    "is non-negative. This is more likely to be the case "
                    "if penalty_strength is relatively high and/or the "
                    "difference (hard_guide_constraint - soft_guide_constraint) "
                    "is large. Trying the 'greedy' maximization "
                    "algorithm may avoid this."))
            raise CannotFindAnyGuidesError(("No guides could be constructed "
                "in the window [%d, %d)") % (start, end))

        return curr_guide_set

    def find_guides_with_sliding_window(self, window_size, out_fn,
            window_step=1, sort=False, print_analysis=True):
        """Find a collection of guides that maximizes expected activity,
        across all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            window_size: length of the window to use when sliding across
                alignment
            out_fn: output TSV file to write guide sequences by window
            window_step: amount by which to increase the window start for
                every search
            sort: if set, sort output TSV by objective value
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the highest objective value
        """
        guide_collections = list(self._find_guides_for_each_window(
            window_size, window_step=window_step))

        if sort:
            # Sort by objective value descending
            guide_collections.sort(key=lambda x: self.obj_value(x[2]),
                    reverse=True)

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'objective-value', 'total-frac-bound',
                'guide-set-expected-activity',
                'guide-set-median-activity', 'guide-set-5th-pctile-activity',
                'guide-expected-activities',
                'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, guide_seqs = guides_in_window
                count = len(guide_seqs)
                activities = self.guide_set_activities(start, end, guide_seqs)
                obj = self.obj_value(start, end, guide_seqs,
                        activities=activities)
                frac_bound = self.total_frac_bound_by_guides(start, end,
                        guide_seqs, activities=activities)
                guides_activity_expected = self.guide_set_activities_expected_value(
                        start, end, guide_seqs, activities=activities)
                guides_activity_median, guides_activity_5thpctile = \
                        self.guide_set_activities_percentile(start, end,
                                guide_seqs, [50, 5], activities=activities)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_guide_positions[gd_seq]
                             for gd_seq in guide_seqs_sorted]
                positions_str = ' '.join(str(p) for p in positions)
                expected_activities_per_guide = \
                        [self.guide_activities_expected_value(start, end,
                            gd_seq) for gd_seq in guide_seqs_sorted]
                expected_activities_per_guide_str = ' '.join(
                        str(a) for a in expected_activities_per_guide)
                line = [start, end, count, obj, frac_bound,
                        guides_activity_expected,
                        guides_activity_median, guides_activity_5thpctile,
                        expected_activities_per_guide_str,
                        guide_seqs_str,
                        positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')

        if print_analysis:
            num_windows_scanned = len(
                range(0, self.aln.seq_length - window_size + 1))
            num_windows_with_guides = len(guide_collections)

            stat_display = [
                ("Number of windows scanned", num_windows_scanned),
                ("Number of windows with guides", num_windows_with_guides)
            ]

            # Print the above statistics, with padding on the left
            # so that the statistic names are right-justified in a
            # column and the values line up, left-justified, in a column
            max_stat_name_len = max(len(name) for name, val in stat_display)
            for name, val in stat_display:
                pad_spaces = max_stat_name_len - len(name)
                name_padded = " "*pad_spaces + name + ":"
                print(name_padded, str(val))


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


class CannotFindAnyGuidesError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class NoPredictorError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
