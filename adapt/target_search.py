"""Methods for searching for amplicons and guides within them.

This combines primer_search, to search for amplicons bound by primers,
and guide_search, to search for guides within those amplicons. We
term the combination of these a "target".

These methods score possible targets and search for the top N of them.
"""

import copy
import heapq
import itertools
import logging
import math
import multiprocessing
import pathos.multiprocessing
import dill

from adapt import guide_search
from adapt import primer_search
from adapt.utils import timeout

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class TargetSearcher:
    """Methods to search for targets over a genome."""

    def __init__(self, ps, gs, max_primers_at_site=None,
            max_target_length=None, cost_weights=None,
            guides_should_cover_over_all_seqs=False):
        """
        Args:
            ps: PrimerSearcher object
            gs: GuideSearcher object
            max_primers_at_site: only allow amplicons in which each
                end has at most this number of primers; or None for
                no limit
            max_target_length: only allow amplicons whose length is at
                most this; or None for no limit
            cost_weights: a tuple giving weights in the cost function
                in the order (primers, window, guides)
            guides_should_cover_over_all_seqs: design guides so as to cover
                the specified fraction (gs.cover_frac) of *all* sequences,
                rather than gs.cover_frac of only sequences bound by the
                primers
        """
        self.ps = ps
        self.gs = gs
        self.max_primers_at_site = max_primers_at_site
        self.max_target_length = max_target_length

        if cost_weights is None:
            cost_weights = (0.6667, 0.2222, 0.1111)
        self.cost_weight_primers = cost_weights[0]
        self.cost_weight_window = cost_weights[1]
        self.cost_weight_guides = cost_weights[2]

        self.guides_should_cover_over_all_seqs = guides_should_cover_over_all_seqs

    def _find_primer_pairs(self):
        """Find suitable primer pairs using self.ps.

        Yields:
            tuple (p_i, p_j) where p_i is a tuple containing a
            primer_search.PrimerResult object, and likewise for p_j.
            The start position of p_j is > the start position of p_i.
        """
        primers = list(self.ps.find_primers(
            max_at_site=self.max_primers_at_site))
        for i in range(len(primers) - 1):
            p1 = primers[i]
            for j in range(i + 1, len(primers)):
                p2 = primers[j]
                yield (p1, p2)

    def find_targets(self, best_n=10, no_overlap=True):
        """Find targets across an alignment.

        Args:
            best_n: only store and output the best_n targets with the
                lowest cost
            no_overlap: if True, do not allow targets in the output
                whose primers overlap on both ends. When True, if a
                target overlaps with a target already in the output,
                this *replaces* the overlapping one with the new one
                if the new one has a smaller cost (if it has a greater
                cost, it ignores the new one). When False,
                many targets in the best_n may be very similar

        Returns:
            list of tuples (cost, target) where target is a tuple
            ((p1, p2), (guides_frac_bound, guides)); (p1, p2) is a pair
            of PrimerResult objects, and guides is a list of guides that
            achieve the desired coverage (with guides_frac_bound fraction
            of the sequences covered) in the window bound by (p1, p2)
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
        push_id_counter = itertools.count()
        print('starting _find_primer_pairs')

        # First prune out unsuitable primer pairs, and then find the primer pair
        # spanning the longest distance. This distance will the length of genome
        # allocated to each process.
        num_primer_pairs = 0
        num_suitable_primer_pairs = 0
        last_window_start = -1
        suitable_primer_pairs = []
        max_primer_pair_span = 0
        for p1, p2 in self._find_primer_pairs():
            num_primer_pairs += 1

            target_length = p2.start + p2.primer_length - p1.start
            if (self.max_target_length is not None and
                    target_length > self.max_target_length):
                # This is longer than allowed, so skip it
                continue

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

            # If here, the window passed basic checks
            num_suitable_primer_pairs += 1
            max_primer_pair_span = max(max_primer_pair_span, target_length)
            suitable_primer_pairs.append((p1, p2))

        # Separate out loop body of finding guides between primers in function.
        def get_guides_for_primers(gs, suitable_primer_pairs):
            for p1, p2 in suitable_primer_pairs:
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                # Since window_start increases monotonically, we no
                # longer need to memoize guide covers between the last
                # and current start
                assert window_start >= last_window_start
                for pos in range(last_window_start, window_start):
                    gs._cleanup_memoized_guides(pos)
                last_window_start = window_start
                target_heap = []

                # Calculate a cost of the primers
                p1_num = p1.num_primers
                p2_num = p2.num_primers
                cost_primers = self.cost_weight_primers * (p1_num + p2_num)

                # Calculate a cost of the window
                window_length = window_end - window_start
                cost_window = self.cost_weight_window * math.log2(window_length)

                # Calculate a lower bound on the total cost of this target,
                # which can be done assuming >= 1 guide will be found; this
                # is useful for pruning the search
                cost_total_lo = (cost_primers + cost_window +
                    self.cost_weight_guides * 1)

                # Check if we should bother trying to find guides in this window
                if len(target_heap) >= best_n:
                    curr_highest_cost = -1 * target_heap[0][0]
                    if cost_total_lo >= curr_highest_cost:
                        # The cost is already >= than all in the current best_n,
                        # and will only stay the same or get bigger after adding
                        # the term for guides, so there is no reason to continue
                        # considering this window
                        continue

                # If targets should not overlap and this overlaps a target
                # already in target_heap, check if we should bother trying to
                # find guides in this window
                overlapping_i = []
                if no_overlap:
                    # Check if any targets already in target_heap have
                    # primer pairs overlapping with (p1, p2)
                    for i in range(len(target_heap)):
                        _, _, target_i = target_heap[i]
                        (p1_i, p2_i), _ = target_i
                        if p1.overlaps(p1_i) and p2.overlaps(p2_i):
                            # Replace target_i
                            overlapping_i += [i]
                if overlapping_i:
                    # target overlaps with one or more entries already in
                    # target_heap
                    cost_is_too_high = True
                    for i in overlapping_i:
                        cost_i = -1 * target_heap[i][0]
                        if cost_total_lo < cost_i:
                            cost_is_too_high = False
                            break
                    if cost_is_too_high:
                        # We will try to replace an entry in overlapping_i (i) with
                        # a new entry, but the cost is already >= than what it
                        # is for i, and will only stay the same or get bigger
                        # after adding the term for guides, so there is no reason
                        # for considering this window (it will never replace i)
                        continue

                logger.info(("Found window [%d, %d) bound by primers that could "
                    "be in the best %d targets; looking for guides within this"),
                    window_start, window_end, best_n)

                # Determine what set of sequences the guides should cover
                if self.guides_should_cover_over_all_seqs:
                    # Design across the entire collection of sequences
                    # This may lead to more guides being designed than if only
                    # designing across primer_bound_seqs (below) because more
                    # sequences must be considered in the design
                    # But this can improve runtime because the seqs_to_consider
                    # used by gs in the design will be more constant
                    # across different windows, which will enable it to better
                    # take advantage of memoization (in gs._memoized_guides)
                    # during the design
                    guide_seqs_to_consider = None
                else:
                    # Find the sequences that are bound by some primer in p1 AND
                    # some primer in p2
                    p1_bound_seqs = self.ps.seqs_bound_by_primers(p1.primers_in_cover)
                    p2_bound_seqs = self.ps.seqs_bound_by_primers(p2.primers_in_cover)
                    primer_bound_seqs = p1_bound_seqs & p2_bound_seqs
                    guide_seqs_to_consider = primer_bound_seqs

                # Find guides in the window, only searching across
                # the sequences in guide_seqs_to_consider
                try:
                    guides = gs._find_guides_that_cover_in_window(
                        window_start, window_end,
                        only_consider=guide_seqs_to_consider)
                except guide_search.CannotAchieveDesiredCoverageError:
                    # No more suitable guides; skip this window
                    continue

                if len(guides) == 0:
                    # This can happen, for example, if primer_bound_seqs is
                    # empty; then no guides are required
                    # Skip this window
                    continue

                # Calculate fraction of sequences bound by the guides
                guides_frac_bound = gs._total_frac_bound_by_guides(guides)

                # Calculate a cost of the guides, and a total cost
                cost_guides = self.cost_weight_guides * len(guides)
                cost_total = cost_primers + cost_window + cost_guides

                # Add target to the heap (but only keep it if there are not
                # yet best_n targets or it has one of the best_n smallest
                # costs)
                target = ((p1, p2), (guides_frac_bound, guides))
                entry = (-cost_total, next(push_id_counter), target)
                if overlapping_i:
                    # target overlaps with an entry already in target_heap;
                    # consider replacing that entry with new entry
                    if len(overlapping_i) == 1:
                        # Almost all cases will satisfy this: the new entry
                        # overlaps just one existing entry; optimize for this case
                        i = overlapping_i[0]
                        cost_i = -1 * target_heap[i][0]
                        if cost_total < cost_i:
                            # Replace i with entry
                            target_heap[i] = entry
                            heapq.heapify(target_heap)
                    else:
                        # For some edge cases, the new entry overlaps with
                        # multiple existing entries
                        # Check if the new entry has a sufficiently low cost
                        # to justify removing any existing overlapping entries
                        cost_is_sufficiently_low = False
                        for i in overlapping_i:
                            cost_i = -1 * target_heap[i][0]
                            if cost_total < cost_i:
                                cost_is_sufficiently_low = True
                                break
                        if cost_is_sufficiently_low:
                            # Remove all entries that overlap the new entry,
                            # then add the new entry
                            for i in sorted(overlapping_i, reverse=True):
                                del target_heap[i]
                            target_heap.append(entry)
                            heapq.heapify(target_heap)
                            # Note that this can lead to cases in which the size
                            # of the output (target_heap) is < best_n, or in which
                            # there is an entry with very high cost, because
                            # in certain edge cases we might finish the search
                            # having removed >1 entry but only replacing it with
                            # one, and if the search continues for a short time
                            # after, len(target_heap) < best_n could result in
                            # some high-cost entries being placed into the heap
                elif len(target_heap) < best_n:
                    # target_heap is not yet full, so push entry to it
                    heapq.heappush(target_heap, entry)
                else:
                    # target_heap is full; consider replacing the entry that has
                    # the highest cost with the new entry
                    curr_highest_cost = -1 * target_heap[0][0]
                    if cost_total < curr_highest_cost:
                        # Push the entry into target_heap, and pop out the
                        # one with the highest cost
                        heapq.heappushpop(target_heap, entry)

            return target_heap

        p = dill.dumps(get_guides_for_primers)
        print('correctly pickled get_guides_for_primers')

        # create GuideSearcher for each chunk and map primer pair to chunk
        gs_list = []
        for i in range(0, self.gs.aln.seq_length, max_primer_pair_span):
            # Alternative approach was to extract sub-alignments from the main
            # alignment and create GuideSearchers as copies of self.gs but this
            # involved a lot of copying that seemed unwieldy
            # aln = self.gs.aln.extract_range(i, i + max_primer_pair_span)
            gs = copy.copy(self.gs) # or deepcopy?
            gs.blacklisted_ranges = [(0, i), (i+2*max_primer_pair_span,
                                              self.gs.aln.seq_length)]
            # I multiply by 2 in case a primer pair starts near the end of one
            # chunk and bleeds into another
            gs_list.append(gs)
        suitable_primer_pairs_by_gs = [[]] * len(gs_list)
        for p1, p2 in suitable_primer_pairs:
            i = math.floor(p1.start / max_primer_pair_span) # index of gs in gs_list to use
            suitable_primer_pairs_by_gs[i].append((p1, p2))

        # Create multiprocessing pool.
        # Sometimes opening a pool (via multiprocessing.Pool) hangs indefinitely,
        # particularly when many pools are opened/closed repeatedly by a master
        # process; this likely stems from issues in multiprocessing.Pool. So set
        # a timeout on opening the pool, and try again if it times out. It
        # appears, from testing, that opening a pool may timeout a few times in
        # a row, but eventually succeeds.
        time_limit = 60
        num_processes = multiprocessing.cpu_count()
        while True:
            try:
                with timeout.time_limit(time_limit):
                    _process_pool = pathos.multiprocessing.ProcessPool(num_processes)
                break
            except timeout.TimeoutException:
                # Try again
                logger.debug("Pool initialization timed out; trying again")
                time_limit *= 2
                continue

        print('mapping get_guides_for_primers on %d processes...' % (num_processes, ))
        # Map the action of finding guides between primers to processes.
        target_heaps = _process_pool.map(get_guides_for_primers,
                                         zip(gs_list,
                                             suitable_primer_pairs_by_gs))
        print('finished mapping get_guides_for_primers')

        # Merge heaps from multiple processes.
        target_heap = heapq.merge(*target_heaps)

        if len(target_heap) == 0:
            logger.warning(("Zero targets were found. The number of total "
                "primer pairs found was %d and the number of them that "
                "were suitable (passing basic criteria, e.g., on length) "
                "was %d"), num_primer_pairs, num_suitable_primer_pairs)

        # Invert the costs (target_heap had been storing the negative
        # of each cost), toss push_id, and sort by cost
        # In particular, sort by a 'sort_tuple' whose first element is
        # cost and then the target endpoints; it has the target endpoints
        # to break ties in cost
        r = [(-cost, target) for cost, push_id, target in target_heap]
        r_with_sort_tuple = []
        for (cost, target) in r:
            ((p1, p2), (guides_frac_bound, guides)) = target
            target_start = p1.start
            target_end = p2.start + p2.primer_length
            sort_tuple = (cost, target_start, target_end)
            r_with_sort_tuple += [(sort_tuple, cost, target)]
        r_with_sort_tuple = sorted(r_with_sort_tuple, key=lambda x: x[0])
        r = [(cost, target) for sort_tuple, cost, target in r_with_sort_tuple]
        print('finished _find_primer_pairs')

        return r

    def find_and_write_targets(self, out_fn, best_n=10):
        """Find targets across an alignment, and write them to a file.

        This writes a table of the targets to a file, in which each
        row corresponds to a target in the genome (primer and guide
        sets).

        Args:
            out_fn: output TSV file to write targets
            best_n: only store and output the best_n targets with the
                lowest cost
        """
        targets = self.find_targets(best_n=best_n)

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['cost', 'target-start', 'target-end',
                'target-length',
                'left-primer-start', 'left-primer-num-primers',
                'left-primer-frac-bound', 'left-primer-target-sequences',
                'right-primer-start', 'right-primer-num-primers',
                'right-primer-frac-bound', 'right-primer-target-sequences',
                'num-guides', 'total-frac-bound-by-guides',
                'guide-target-sequences', 'guide-target-sequence-positions']) +
                '\n')

            for (cost, target) in targets:
                ((p1, p2), (guides_frac_bound, guides)) = target

                # Determine the target endpoints
                target_start = p1.start
                target_end = p2.start + p2.primer_length
                target_length = target_end - target_start

                # Construct string of primers and guides
                p1_seqs_sorted = sorted(list(p1.primers_in_cover))
                p1_seqs_str = ' '.join(p1_seqs_sorted)
                p2_seqs_sorted = sorted(list(p2.primers_in_cover))
                p2_seqs_str = ' '.join(p2_seqs_sorted)
                guides_seqs_sorted = sorted(list(guides))
                guides_seqs_str = ' '.join(guides_seqs_sorted)

                # Find positions of the guides
                guides_positions = [self.gs._selected_guide_positions[gd_seq]
                                    for gd_seq in guides_seqs_sorted]
                guides_positions_str = ' '.join(str(p) for p in guides_positions)

                line = [cost, target_start, target_end, target_length,
                    p1.start, p1.num_primers, p1.frac_bound, p1_seqs_str,
                    p2.start, p2.num_primers, p2.frac_bound, p2_seqs_str,
                    len(guides), guides_frac_bound, guides_seqs_str,
                    guides_positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')
