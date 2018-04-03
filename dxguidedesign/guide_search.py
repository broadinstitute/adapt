"""Methods for searching for optimal guides to use for a diagnostic.
"""

# temp comment

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
        average fraction of sequences that are covered by guides in S
        (i.e., the sum of the fraction of sequences covered by each guide
        divided by the size of S). The score is a value in [0, 1].

        The score is meant to be compared across sets of guides that
        are the same size (i.e., have the same number of guides). It
        is not necessarily useful for comparing across sets of guides
        that differ in size.

        Args:
            guides: collection of str representing guide sequences

        Returns:
            score of guides, as defined above
        """
        sum_of_frac_of_seqs_bound = 0
        for gd_seq in guides:
            seqs_bound = set()
            for pos in self._selected_guide_positions[gd_seq]:
                seqs_bound.update(self.aln.sequences_bound_by_guide(gd_seq,
                    pos, self.mismatches))
            frac_bound = len(seqs_bound) / float(self.aln.num_sequences)
            sum_of_frac_of_seqs_bound += frac_bound
        score = sum_of_frac_of_seqs_bound / float(len(guides))
        return score

    def _find_guides_that_cover_for_each_window(self):
        """Find the smallest collection of guides that cover sequences
        in each window.

        This runs a sliding window across the aligned sequences and, in each
        window, calculates the smallest number of guides needed in the window
        to cover the sequences by calling self._find_guides_that_cover_in_window().

        This returns guides for each window, along with summary information
        about the guides in the window.

        This does not return guides for windows where it cannot achieve
        the desired coverage in the window (e.g., due to indels or ambiguity).

        Returns:
            list of elements x_i in which each x_i corresponds to a window;
            x_i is a tuple consisting of the following values, in order:
              1) start position of the window
              2) number of guides designed for the window (i.e., length of
                 the set in (4))
              3) score corresponding to the guides in the window, which can
                 be used to break ties across windows that have the same
                 number of minimal guides (higher is better)
              4) set of guides that achieve the desired coverage and is
                 minimal for the window
        """
        min_guides_in_cover = set()
        min_guides_in_cover_count = None
        guides = []
        for start in range(0, self.aln.seq_length - self.window_size + 1):
            logger.info(("Searching for guides within window starting "
                         "at %d") % start)

            try:
                guides_in_cover = self._find_guides_that_cover_in_window(start)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                logger.warning(("No more suitable guides could be constructed "
                    "in the window starting at %d, but more are needed to "
                    "achieve the desired coverage") % start)
                continue

            num_guides = len(guides_in_cover)
            score = self._score_collection_of_guides(guides_in_cover)
            guides += [ [start, num_guides, score, guides_in_cover] ]

            # We no longer need to memoize results for guides that start at
            # this position
            self._cleanup_memoized_guides(start)

        return guides

    def find_guides_that_cover(self, out_fn, sort=False, print_analysis=True):
        """Find the smallest collection of guides that cover sequences, across
        all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            out_fn: output TSV file to write guide sequences by window
            sort: if set, sort output TSV by number of guides (ascending)
                then by score (descending); when not set, default is to
                sort by window position
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the smallest number of guides and highest
                score
        """
        guide_collections = self._find_guides_that_cover_for_each_window()

        if sort:
            # Sort by number of guides ascending (x[1]), then by
            # score of guides descending (-x[2])
            guide_collections.sort(key=lambda x: (x[1], -x[2]))

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'score', 'guide-sequences']) + '\n')

            for guides_in_window in guide_collections:
                start, count, score, guide_seqs = guides_in_window
                end = start + self.window_size
                guide_seqs_str = ' '.join(sorted(list(guide_seqs)))
                line = [start, end, count, score, guide_seqs_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')

        if print_analysis:
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
                ("Number of windows scanned", len(guide_collections)),
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
