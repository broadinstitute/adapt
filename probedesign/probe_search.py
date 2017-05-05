"""Methods for searching for optimal probes to use for a diagnostic.
"""

from collections import defaultdict
import logging

from probedesign import alignment

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class ProbeSearcher:
    """Methods to search for probes to use for a diagnostic.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable probes.
    """

    def __init__(self, aln, probe_length, mismatches, window_size, cover_frac):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            probe_length: length of the probe to construct
            mismatches: threshold on number of mismatches for determining whether
                a probe would hybridize to a target sequence
            window_size: length of window such that a set of probes are only selected
                if they are all within a window of this length
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a probe
        """
        if window_size > aln.seq_length:
            raise ValueError("window_size must be less than the length of the alignment")
        if probe_length > window_size:
            raise ValueError("probe_length must be less than the window size")
        if cover_frac <= 0 or cover_frac > 1:
            raise ValueError("cover_frac must be in (0,1]")

        self.aln = aln
        self.probe_length = probe_length
        self.mismatches = mismatches
        self.window_size = window_size
        self.cover_frac = cover_frac

        # Because calls to alignment.Alignment.construct_probe() are expensive
        # and are repeated very often, memoize the output
        self._memoized_probes = defaultdict(dict)

    def _construct_probe_memoized(self, start, seqs_to_consider):
        """Make a memoized call to alignment.Alignment.construct_probe().

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: collection of indices of sequences to use when
                constructing the probe

        Returns:
            result of alignment.Alignment.construct_probe()
        """
        seqs_to_consider_frozen = frozenset(seqs_to_consider)
        if (start in self._memoized_probes and
                seqs_to_consider_frozen in self._memoized_probes[start]):
            return self._memoized_probes[start][seqs_to_consider_frozen]
        else:
            try:
                p = self.aln.construct_probe(start, self.probe_length,
                        seqs_to_consider, self.mismatches)
            except alignment.CannotConstructProbeError:
                p = None
            self._memoized_probes[start][seqs_to_consider_frozen] = p
            return p

    def _cleanup_memoized_probes(self, pos):
        """Remove a position that is stored in self._memoized_probes.

        This should be called when the position no longer needs to be stored.

        Args:
            pos: start position that no longer needs to be memoized (i.e., where
                probes covering at that start position are no longer needed)
        """
        if pos in self._memoized_probes:
            del self._memoized_probes[pos]

    def _find_optimal_probe_in_window(self, start, seqs_to_consider):
        """Find the probe that hybridizes to the most sequences in a given window.

        This considers each position within the specified window at which a probe
        can start. At each, it determines the optimal probe (i.e., attempting to cover
        the most number of sequences) as well as the number of sequences that the
        probe covers (hybridizes to). It selects the probe that covers the most. This
        breaks ties arbitrarily.

        Args:
            start: starting position of the window; the window spans [start,
                start + self.window_size)
            seqs_to_consider: collection of indices of sequences to use when selecting
                a probe

        Returns:
            tuple (x, y) where:
                x is the sequence of the selected probe
                y is a collection of indices of sequences (a subset of
                    seqs_to_consider) to which probe x will hybridize
        """
        assert start >= 0
        assert start + self.window_size <= self.aln.seq_length

        # Calculate the end of the search (exclusive), which is the last
        # position in the window at which a probe can start; a probe needs to
        # fit completely within the window
        search_end = start + self.window_size - self.probe_length + 1

        max_probe_cover = None
        for pos in range(start, search_end):
            p = self._construct_probe_memoized(pos, seqs_to_consider)
            if p is None:
                # There is no suitable probe at pos
                if max_probe_cover is None:
                    max_probe_cover = (None, set())
            else:
                prb, covered_seqs = p
                covered_seqs = set(covered_seqs)
                if max_probe_cover is None:
                    max_probe_cover = (prb, covered_seqs)
                else:
                    if len(covered_seqs) > len(max_probe_cover[1]):
                        # prb covers the most sequences of all probes so far in
                        # this window
                        max_probe_cover = (prb, covered_seqs)
        return max_probe_cover

    def _find_probes_that_cover_in_window(self, start):
        """Find a collection of probes that cover sequences in a given window.

        This attempts to find the smallest number of probes such that, within
        the specified window, at least the fraction self.cover_frac of
        sequences have a probe that hybridizes to it.

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
        a possible probe and the elements of S_i are integers representing
        the sequences to which the probe hybridizes (i.e., the sequences that
        the probe "covers"). When approximating the solution, we do not need
        to actually construct all possible probes (or subsets S_i). Instead,
        on the first iteration we construct the probe p_1 that hybridizes to
        (covers) the most number of sequences; S_1 is then the sequences that
        p_1 covers. We include p_1 in the output and subtract all of the
        sequences in S_1 from U. On the next iteration, we construct the
        probe that covers the most number of sequences that remain in U, and
        so on.

        Although it is not done here, we could choose to assign each probe
        a cost (e.g., based on its sequence composition) and then select the
        probes that have the smallest total cost while achieving the desired
        coverage of sequences. This would be a weighted set cover problem.
        Without assigning costs, we can think of each probe as having a cost
        of 1.0; in this case, we simply select the smallest number of probes
        that achieve the desired coverage.

        Args:
            start: starting position of the window; the window spans [start,
                start + self.window_size)

        Returns:
            collection of str representing probe sequences that were selected
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

        probes_in_cover = set()
        # Keep iterating until desired partial cover is obtained
        while num_left_to_cover > 0:
            # Find the probe that hybridizes to the most sequences, among
            # those that are not in the cover
            prb, prb_covered_seqs = self._find_optimal_probe_in_window(start,
                universe)

            if prb is None or len(prb_covered_seqs) == 0:
                # No suitable probes could be constructed within the window
                raise CannotAchieveDesiredCoverageError(("No suitable probes "
                    "could be constructed in the window starting at %d, but "
                    "more are needed to achieve desired coverage") % start)

            # The set representing prb goes into the set cover, and all of the
            # sequences it hybridizes to are removed from the universe
            probes_in_cover.add(prb)
            universe.difference_update(prb_covered_seqs)
            num_left_to_cover = max(0, len(universe) - num_that_can_be_uncovered)

        return probes_in_cover

    def find_probes_that_cover(self):
        """Find the smallest collection of probes that cover sequences, across
        all windows.

        This runs a sliding window across the aligned sequences and, in each
        window, calculates the smallest number of probes needed in the window
        to cover the sequences by calling self._find_probes_that_cover_in_window().

        Returns:
            set of sets x_i, such that each x_i holds a unique collection of
            probes that achieve the desired coverage and is minimal (the
            length of each x_i is the same)
        """
        min_probes_in_cover = set()
        min_probes_in_cover_count = None
        for start in range(0, self.aln.seq_length - self.window_size + 1):
            if start == 0:
                logger.info(("Starting search for probes within window "
                             "starting at 0"))
            elif min_probes_in_cover_count is not None:
                logger.info(("Searching within window starting at %d, current "
                             "minimum probes needed is %d") % (start,
                            min_probes_in_cover_count))

            try:
                probes_in_cover = self._find_probes_that_cover_in_window(start)
            except CannotAchieveDesiredCoverageError:
                # Cannot achieve the desired coverage in this window; log and
                # skip it
                logger.warning(("No more suitable probes could be constructed "
                    "in the window starting at %d, but more are needed to "
                    "achieve the desired coverage") % start)
                continue

            if min_probes_in_cover_count is None:
                min_probes_in_cover.add(frozenset(probes_in_cover))
                min_probes_in_cover_count = len(probes_in_cover)
            elif len(probes_in_cover) == min_probes_in_cover_count:
                min_probes_in_cover.add(frozenset(probes_in_cover))
            elif len(probes_in_cover) < min_probes_in_cover_count:
                min_probes_in_cover = set()
                min_probes_in_cover.add(frozenset(probes_in_cover))
                min_probes_in_cover_count = len(probes_in_cover)

            # We no longer need to memoize results for probes that start at
            # this position
            self._cleanup_memoized_probes(start)

        return min_probes_in_cover


class CannotAchieveDesiredCoverageError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
