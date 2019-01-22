"""Methods for searching for amplicons and guides within them.

This combines primer_search, to search for amplicons bound by primers,
and guide_search, to search for guides within those amplicons. We
term the combination of these a "target".

These methods score possible targets and search for the top N of them.
"""

import heapq
import itertools
import logging
import math

from dxguidedesign import guide_search
from dxguidedesign import primer_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class TargetSearcher:
    """Methods to search for targets over a genome."""

    def __init__(self, ps, gs, max_primers_at_site=2):
        """
        Args:
            ps: PrimerSearcher object
            gs: GuideSearcher object
            max_primers_at_site: only allow amplicons in which each
                end has at most this number of primers
        """
        self.ps = ps
        self.gs = gs
        self.max_primers_at_site = max_primers_at_site

        # Define weights in the cost function
        self._cost_weight_primer = 0.6667
        self._cost_weight_window = 0.2222
        self._cost_weight_guides = 0.1111

    def _find_primer_pairs(self):
        """Find suitable primer pairs using self.ps.

        Yields:
            tuple (p_i, p_j) where p_i is a tuple containing a
            primer_search.PrimerResult object, and likewise for p_j.
            The start position of p_j is > the start position of p_i.
        """
        primers = list(self.ps.find_primers(max_at_site=self.max_primers_at_site))
        for i in range(len(primers) - 1):
            p1 = primers[i]
            for j in range(i + 1, len(primers)):
                p2 = primers[j]
                yield (p1, p2)

    def find_targets(self, best_n=10):
        """Find targets across an alignment.

        Args:
            best_n: only store and output the best_n targets with the
                lowest cost

        Returns:
            list of tuples (cost, target) where target is a tuple
            ((p1, p2), guides); (p1, p2) is a pair of PrimerResult
            objects and guides is a list of guides that achieve the
            desired coverage in the window bound by (p1, p2)
        """

        # Store a heap containing the best_n primer/guide targets;
        # store each as a tuple (-cost, push_id, target), where we store
        # based on the negative cost because the heap is a min heap
        # (so popped values will be the one with the greatest cost,
        # and target_heap[0] refers to the target with the greatest cost).
        # We also store push_id (a counter on the push) to break ties
        # when costs are equivalent (choosing the one pushed first, i.e.,
        # the lower push_id); otherwise, there will be an error on
        # ties if target is not comparable
        target_heap = []
        push_id_counter = itertools.count()

        last_window_start = -1
        for p1, p2 in self._find_primer_pairs():
            # Determine a window between the two primers
            window_start = p1.start + p1.primer_length
            window_end = p2.start

            # To be a suitable window, p2 must start after p1 ends
            if window_end <= window_start:
                continue
            window_length = window_end - window_start

            # To consider this window, a guide must fit within it
            if window_length < self.gs.guide_length:
                continue

            # Since window_start increases monotonically, we no
            # longer need to memoize guide covers between the last
            # and current start
            assert window_start >= last_window_start
            for pos in range(last_window_start, window_start):
                self.gs._cleanup_memoized_guides(pos)
            last_window_start = window_start

            # Calculate a cost of the primers
            p1_num = p1.num_primers
            p2_num = p2.num_primers
            cost_primers = self._cost_weight_primer * (p1_num + p2_num)

            # Calculate a cost of the window
            cost_window = self._cost_weight_window * math.log2(window_length)

            # Check if we should bother trying to find guides in this window
            if len(target_heap) > 0:
                curr_highest_cost = -1 * target_heap[0][0]
                if cost_primers + cost_window > curr_highest_cost:
                    # The cost is already greater than the current best_n,
                    # so there is no reason to continue considering this
                    # window
                    continue

            # Find the sequences that are bound by some primer in p1 AND
            # some primer in p2
            p1_bound_seqs = self.ps.seqs_bound_by_primers(p1.primers_in_cover)
            p2_bound_seqs = self.ps.seqs_bound_by_primers(p2.primers_in_cover)
            primer_bound_seqs = p1_bound_seqs & p2_bound_seqs

            # Find guides in the window, only searching across
            # the sequences in primer_bound_seqs
            try:
                guides = self.gs._find_guides_that_cover_in_window(
                    window_start, window_end,
                    only_consider=primer_bound_seqs)
            except guide_search.CannotAchieveDesiredCoverageError:
                # No more suitable guides; skip this window
                continue

            # Calculate a cost of the guides, and a total cost
            cost_guides = self._cost_weight_guides * len(guides)
            cost_total = cost_primers + cost_window + cost_guides

            # Add target to the heap (but only keep it if there are not
            # yet best_n targets or it has one of the best_n smallest
            # costs)
            target = ((p1, p2), guides)
            entry = (-cost_total, next(push_id_counter), target)
            if len(target_heap) < best_n:
                heapq.heappush(target_heap, entry)
            else:
                heapq.heappushpop(target_heap, entry)

        # Invert the costs (target_heap had been storing the negative
        # of each cost), toss push_id, and sort by cost
        r = [(-cost, target) for cost, push_id, target in target_heap]
        r = sorted(r, key=lambda x: x[0])

        return r
