"""Methods for searching for optimal guides to use for a diagnostic.
"""

from collections import defaultdict
import logging
import math

from dxguidedesign import alignment
from dxguidedesign.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class GuideSearcher:
    """Methods to search for guides to use for a diagnostic.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable guides.
    """

    def __init__(self, aln, guide_length, mismatches, cover_frac,
                 missing_data_params, guide_is_suitable_fn=None,
                 seq_groups=None, required_guides={}, blacklisted_ranges={},
                 allow_gu_pairs=False):
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
            guide_is_suitable_fn: if set, the value of this argument is a
                function f(x) such that this will only construct a guide x
                for which f(x) is True
            seq_groups: dict that maps group ID to collection of sequences in
                that group. If set, cover_frac must also be a dict that maps
                group ID to the fraction of sequences in that group that
                must be 'captured' by a guide. If None, then do not divide
                the sequences into groups.
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
        """
        if seq_groups is None and (cover_frac <= 0 or cover_frac > 1):
            raise ValueError("cover_frac must be in (0,1]")

        self.aln = aln
        self.guide_length = guide_length
        self.mismatches = mismatches

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

        # Because calls to alignment.Alignment.construct_guide() are expensive
        # and are repeated very often, memoize the output
        self._memoized_guides = defaultdict(dict)

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

        self.guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=min(10, int(guide_length/2)))

    def _construct_guide_memoized(self, start, seqs_to_consider,
            num_needed=None):
        """Make a memoized call to alignment.Alignment.construct_guide().

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: dict mapping universe group ID to collection
                of indices to use when constructing the guide
            num_needed: dict mapping universe group ID to the number of
                sequences from the group that are left to cover in order to
                achieve the desired partial cover

        Returns:
            result of alignment.Alignment.construct_guide()
        """
        # Make frozen version of both dicts; note that values in
        # seqs_to_consider may be sets that need to be frozen
        seqs_to_consider_frozen = set()
        for k, v in seqs_to_consider.items():
            seqs_to_consider_frozen.add((k, frozenset(v)))
        seqs_to_consider_frozen = frozenset(seqs_to_consider_frozen)
        if num_needed is None:
            num_needed_frozen = None
        else:
            num_needed_frozen = frozenset(num_needed.items())

        key = (seqs_to_consider_frozen, num_needed_frozen)
        if (start in self._memoized_guides and
                key in self._memoized_guides[start]):
            return self._memoized_guides[start][key]
        else:
            try:
                p = self.aln.construct_guide(start, self.guide_length,
                        seqs_to_consider, self.mismatches, self.allow_gu_pairs,
                        self.guide_clusterer, num_needed=num_needed,
                        missing_threshold=self.missing_threshold,
                        guide_is_suitable_fn=self.guide_is_suitable_fn)
            except alignment.CannotConstructGuideError:
                p = None
            self._memoized_guides[start][key] = p
            return p

    def _cleanup_memoized_guides(self, pos):
        """Remove a position that is stored in self._memoized_guides.

        This should be called when the position no longer needs to be stored.

        Args:
            pos: start position that no longer needs to be memoized (i.e., where
                guides covering at that start position are no longer needed)
        """
        if pos in self._memoized_guides:
            del self._memoized_guides[pos]

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

        max_guide_cover = None
        for pos in range(start, search_end):
            if self._guide_overlaps_blacklisted_range(pos):
                # guide starting at pos would overlap a blacklisted range,
                # so skip this guide
                p = None
            else:
                p = self._construct_guide_memoized(pos, seqs_to_consider,
                    num_needed)

            if p is not None and self.guide_is_suitable_fn is not None:
                # Verify that the guide is suitable according to the given
                # function (it should be, in order to have been output
                # by self.aln.construct_guide(..))
                gd, _ = p
                if not self.guide_is_suitable_fn(gd):
                    p = None

            if p is None:
                # There is no suitable guide at pos
                if max_guide_cover is None:
                    max_guide_cover = (None, set(), None, 0)
            else:
                gd, covered_seqs = p
                covered_seqs = set(covered_seqs)

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

    def _find_guides_that_cover_in_window(self, start, end,
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
                    gd, gd_pos, self.mismatches, self.allow_gu_pairs)
                if len(gd_covered_seqs) == 0:
                    # gd covers no sequences at gd_pos; still initialize with it
                    # but give a warning
                    logger.warning(("Guide '%s' at position %d does not cover "
                        "any sequences but is being required in the cover") %
                        (gd, gd_pos))
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
                    pos, self.mismatches, self.allow_gu_pairs))

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
                    pos, self.mismatches, self.allow_gu_pairs))
        return seqs_bound

    def _total_frac_bound_by_guides(self, guides):
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

    def _find_guides_that_cover_for_each_window(self, window_size,
            hide_warnings=False):
        """Find the smallest collection of guides that cover sequences
        in each window.

        This runs a sliding window across the aligned sequences and, in each
        window, calculates the smallest number of guides needed in the window
        to cover the sequences by calling self._find_guides_that_cover_in_window().

        This returns guides for each window, along with summary information
        about the guides in the window.

        This does not return guides for windows where it cannot achieve
        the desired coverage in the window (e.g., due to indels or ambiguity).

        Args:
            window_size: length of the window to use when sliding across
                alignment
            hide_warnings: when set, this does not provide log warnings
                when no more suitable guides can be constructed

        Returns:
            yields x_i in which each x_i corresponds to a window;
            x_i is a tuple consisting of the following values, in order:
              1) start position of the window
              2) end position of the window
              3) number of guides designed for the window (i.e., length of
                 the set in (4))
              4) score corresponding to the guides in the window, which can
                 be used to break ties across windows that have the same
                 number of minimal guides (higher is better)
              5) total fraction of all sequences in the alignment bound
                 by a guide
              6) set of guides that achieve the desired coverage and is
                 minimal for the window
        """
        if window_size > self.aln.seq_length:
            raise ValueError(("window size must be < the length of the "
                              "alignment"))
        if window_size < self.guide_length:
            raise ValueError("window size must be >= guide length") 

        for start in range(0, self.aln.seq_length - window_size + 1):
            end = start + window_size
            logger.info("Searching for guides within window [%d, %d)" %
                        (start, end))

            try:
                guides_in_cover = self._find_guides_that_cover_in_window(
                    start, end)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                if not hide_warnings:
                    logger.warning(("No more suitable guides could be constructed "
                        "in the window [%d, %d), but more are needed to "
                        "achieve the desired coverage") % (start, end))
                self._cleanup_memoized_guides(start)
                continue

            num_guides = len(guides_in_cover)
            score = self._score_collection_of_guides(guides_in_cover)
            frac_bound = self._total_frac_bound_by_guides(guides_in_cover)
            cover = (start, end, num_guides, score, frac_bound, guides_in_cover)
            yield cover

            # We no longer need to memoize results for guides that start at
            # this position
            self._cleanup_memoized_guides(start)

    def find_guides_that_cover(self, window_size, out_fn,
                               sort=False, print_analysis=True):
        """Find the smallest collection of guides that cover sequences, across
        all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            window_size: length of the window to use when sliding across
                alignment
            out_fn: output TSV file to write guide sequences by window
            sort: if set, sort output TSV by number of guides (ascending)
                then by score (descending); when not set, default is to
                sort by window position
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the smallest number of guides and highest
                score
        """
        guide_collections = list(self._find_guides_that_cover_for_each_window(
            window_size))

        if sort:
            # Sort by number of guides ascending (x[1]), then by
            # score of guides descending (-x[2])
            guide_collections.sort(key=lambda x: (x[1], -x[2]))

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'score', 'total-frac-bound', 'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, count, score, frac_bound, guide_seqs = guides_in_window
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
                min_count = min(x[1] for x in guide_collections)
                num_with_min_count = sum(1 for x in guide_collections
                    if x[1] == min_count)
                max_score_for_count = max(x[2] for x in guide_collections
                    if x[1] == min_count)
                num_with_max_score = sum(1 for x in guide_collections if
                    x[1] == min_count and x[2] == max_score_for_count)

                min_count_str = (str(min_count) + " guide" + 
                                 ("s" if min_count > 1 else ""))

                stat_display = [
                    ("Number of windows scanned", num_windows_scanned),
                    ("Number of windows with guides", num_windows_with_guides),
                    ("Minimum number of guides required in a window", min_count),
                    ("Number of windows with " + min_count_str,
                        num_with_min_count),
                    ("Maximum score across windows with " + min_count_str,
                        max_score_for_count),
                    ("Number of windows with " + min_count_str + " and this score",
                        num_with_max_score)
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
