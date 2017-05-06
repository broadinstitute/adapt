"""Methods for searching for optimal guides to use for a diagnostic.
"""

from collections import defaultdict
import logging

from dxguidedesign import alignment

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class GuideSearcher:
    """Methods to search for guides to use for a diagnostic.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable guides.
    """

    def __init__(self, aln, guide_length, mismatches, window_size, cover_frac):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence
            window_size: length of window such that a set of guides are only selected
                if they are all within a window of this length
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a guide
        """
        if window_size > aln.seq_length:
            raise ValueError("window_size must be less than the length of the alignment")
        if guide_length > window_size:
            raise ValueError("guide_length must be less than the window size")
        if cover_frac <= 0 or cover_frac > 1:
            raise ValueError("cover_frac must be in (0,1]")

        self.aln = aln
        self.guide_length = guide_length
        self.mismatches = mismatches
        self.window_size = window_size
        self.cover_frac = cover_frac

        # Because calls to alignment.Alignment.construct_guide() are expensive
        # and are repeated very often, memoize the output
        self._memoized_guides = defaultdict(dict)

        # Save the positions of selected guides in the alignment so these can
        # be easily revisited. In case a guide sequence appears in multiple
        # places, store a set of positions
        self._selected_guide_positions = defaultdict(set)

    def _construct_guide_memoized(self, start, seqs_to_consider):
        """Make a memoized call to alignment.Alignment.construct_guide().

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: collection of indices of sequences to use when
                constructing the guide

        Returns:
            result of alignment.Alignment.construct_guide()
        """
        seqs_to_consider_frozen = frozenset(seqs_to_consider)
        if (start in self._memoized_guides and
                seqs_to_consider_frozen in self._memoized_guides[start]):
            return self._memoized_guides[start][seqs_to_consider_frozen]
        else:
            try:
                p = self.aln.construct_guide(start, self.guide_length,
                        seqs_to_consider, self.mismatches)
            except alignment.CannotConstructGuideError:
                p = None
            self._memoized_guides[start][seqs_to_consider_frozen] = p
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

    def _find_optimal_guide_in_window(self, start, seqs_to_consider):
        """Find the guide that hybridizes to the most sequences in a given window.

        This considers each position within the specified window at which a guide
        can start. At each, it determines the optimal guide (i.e., attempting to cover
        the most number of sequences) as well as the number of sequences that the
        guide covers (hybridizes to). It selects the guide that covers the most. This
        breaks ties arbitrarily.

        Args:
            start: starting position of the window; the window spans [start,
                start + self.window_size)
            seqs_to_consider: collection of indices of sequences to use when selecting
                a guide

        Returns:
            tuple (x, y, z) where:
                x is the sequence of the selected guide
                y is a collection of indices of sequences (a subset of
                    seqs_to_consider) to which guide x will hybridize
                z is the starting position of x in the alignment
        """
        assert start >= 0
        assert start + self.window_size <= self.aln.seq_length

        # Calculate the end of the search (exclusive), which is the last
        # position in the window at which a guide can start; a guide needs to
        # fit completely within the window
        search_end = start + self.window_size - self.guide_length + 1

        max_guide_cover = None
        for pos in range(start, search_end):
            p = self._construct_guide_memoized(pos, seqs_to_consider)
            if p is None:
                # There is no suitable guide at pos
                if max_guide_cover is None:
                    max_guide_cover = (None, set(), None)
            else:
                gd, covered_seqs = p
                covered_seqs = set(covered_seqs)
                if max_guide_cover is None:
                    max_guide_cover = (gd, covered_seqs, pos)
                else:
                    if len(covered_seqs) > len(max_guide_cover[1]):
                        # gd covers the most sequences of all guides so far in
                        # this window
                        max_guide_cover = (gd, covered_seqs, pos)
        return max_guide_cover

    def _find_guides_that_cover_in_window(self, start):
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
        the partial cover.

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

        Args:
            start: starting position of the window; the window spans [start,
                start + self.window_size)

        Returns:
            collection of str representing guide sequences that were selected
        """
        # Create the universe, which is all the input sequences
        universe = set(range(self.aln.num_sequences))

        num_that_can_be_uncovered = int(len(universe) -
                                        self.cover_frac * len(universe))
        # Above, use int(..) to take the floor. Also, expand out
        # len(universe) rather than use int((1.0-self.cover_frac)*len(universe))
        # due to precision errors in Python -- e.g., int((1.0-0.8)*5) yields 0
        # on some machines.
        num_left_to_cover = len(universe) - num_that_can_be_uncovered

        guides_in_cover = set()
        # Keep iterating until desired partial cover is obtained
        while num_left_to_cover > 0:
            # Find the guide that hybridizes to the most sequences, among
            # those that are not in the cover
            gd, gd_covered_seqs, gd_pos = self._find_optimal_guide_in_window(
                start, universe)

            if gd is None or len(gd_covered_seqs) == 0:
                # No suitable guides could be constructed within the window
                raise CannotAchieveDesiredCoverageError(("No suitable guides "
                    "could be constructed in the window starting at %d, but "
                    "more are needed to achieve desired coverage") % start)

            # The set representing gd goes into the set cover, and all of the
            # sequences it hybridizes to are removed from the universe
            guides_in_cover.add(gd)
            universe.difference_update(gd_covered_seqs)
            num_left_to_cover = max(0, len(universe) - num_that_can_be_uncovered)

            # Save the position of this guide in case the guide needs to be
            # revisited
            self._selected_guide_positions[gd].add(gd_pos)

        return guides_in_cover

    def _find_sets_of_guides_that_cover(self):
        """Find all of the smallest collections of guides that cover sequences,
        across all windows.

        This runs a sliding window across the aligned sequences and, in each
        window, calculates the smallest number of guides needed in the window
        to cover the sequences by calling self._find_guides_that_cover_in_window().

        Because there can be ties (i.e., different collections of guides such
        that each collection has the same number of guides), this outputs a set
        of sets of guides.

        Returns:
            set of sets x_i, such that each x_i holds a unique collection of
            guides that achieve the desired coverage and is minimal (the
            length of each x_i is the same)
        """
        min_guides_in_cover = set()
        min_guides_in_cover_count = None
        for start in range(0, self.aln.seq_length - self.window_size + 1):
            if start == 0:
                logger.info(("Starting search for guides within window "
                             "starting at 0"))
            elif min_guides_in_cover_count is not None:
                logger.info(("Searching within window starting at %d, current "
                             "minimum guides needed is %d") % (start,
                            min_guides_in_cover_count))

            try:
                guides_in_cover = self._find_guides_that_cover_in_window(start)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                logger.warning(("No more suitable guides could be constructed "
                    "in the window starting at %d, but more are needed to "
                    "achieve the desired coverage") % start)
                continue

            if min_guides_in_cover_count is None:
                min_guides_in_cover.add(frozenset(guides_in_cover))
                min_guides_in_cover_count = len(guides_in_cover)
            elif len(guides_in_cover) == min_guides_in_cover_count:
                min_guides_in_cover.add(frozenset(guides_in_cover))
            elif len(guides_in_cover) < min_guides_in_cover_count:
                min_guides_in_cover = set()
                min_guides_in_cover.add(frozenset(guides_in_cover))
                min_guides_in_cover_count = len(guides_in_cover)

            # We no longer need to memoize results for guides that start at
            # this position
            self._cleanup_memoized_guides(start)

        return min_guides_in_cover

    def find_guides_that_cover(self):
        """Find the smallest collection of guides that cover sequences, across
        all windows.

        This breaks possible ties in the output of
        self._find_sets_of_guides_that_cover(). In that case, there can be
        multiple (or many) collections of guides such that each collection is
        minimal and has the same number of guides. It breaks ties by trying to
        maximize the redundancy of the guides -- in particular, by trying
        to select one collection of guides in which genomes are covered
        by multiple guides and/or in which many of the guides cover multiple
        genomes. For example, part of this is to make it less likely to
        select guides that only cover one genome (or a small number).

        Because this is loosely defined, we use a crude heuristic to select
        a single collection of guides. For each collection of guides, we sum
        the number of sequences that are covered by each guide. Then, we
        select the collection of guides that has the greatest sum. Since the
        number of guides in each collection is the same, this is equivalent
        to picking the collection whose guides, on average, cover the
        greatest number of sequences.

        This heuristic may result in ties as well; these are broken
        arbitrarily.

        Returns:
            collection of guides that achieves the desired coverage
        """
        gd_collections_for_cover = self._find_sets_of_guides_that_cover()

        logger.info(("There are %d guide collections that are tied; breaking "
                     "ties") % len(gd_collections_for_cover))

        max_gd_collection = None
        max_gd_collection_sum = 0
        max_gd_collection_num = 0
        for gd_collection in gd_collections_for_cover:
            sum_of_seqs_bound = 0
            for gd_seq in gd_collection:
                seqs_bound = set()
                for pos in self._selected_guide_positions[gd_seq]:
                    seqs_bound.update(self.aln.sequences_bound_by_guide(gd_seq,
                        pos, self.mismatches))
                sum_of_seqs_bound += len(seqs_bound)

            if max_gd_collection is None:
                # Store: the guide collection, the sum for the collection, and
                # the number of guide collections that have this particular sum
                max_gd_collection = gd_collection
                max_gd_collection_sum = sum_of_seqs_bound
                max_gd_collection_num = 1
            elif sum_of_seqs_bound == max_gd_collection_sum:
                # Increment the number of guide collections that have this
                # particular sum
                max_gd_collection_num += 1
            elif sum_of_seqs_bound > max_gd_collection_sum:
                max_gd_collection = gd_collection
                max_gd_collection_sum = sum_of_seqs_bound
                max_gd_collection_num = 1

        if max_gd_collection_num > 1:
            logger.warning(("There are %d collections of guides that are still "
                            "tied after trying to break the tie based on the "
                            "sum of the number of sequences to which the "
                            "guides hybridize; breaking the tie arbitrarily") %
                           max_gd_collection_num)

        return max_gd_collection


class CannotAchieveDesiredCoverageError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
