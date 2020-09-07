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
import os

from adapt import guide_search
from adapt import primer_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class TargetSearcher:
    """Methods to search for targets over a genome."""

    def __init__(self, ps, gs, obj_type='min',
            max_primers_at_site=None,
            max_target_length=None, obj_weights=None,
            only_account_for_amplified_seqs=False,
            halt_early=False, obj_value_shift=None):
        """
        Args:
            ps: PrimerSearcher object
            gs: GuideSearcher object
            obj_type: 'min' or 'max' indicating whether the objective
                should be considered to be a minimization or maximization
                problem
            max_primers_at_site: only allow amplicons in which each
                end has at most this number of primers; or None for
                no limit
            max_target_length: only allow amplicons whose length is at
                most this; or None for no limit
            obj_weights: a tuple giving weights in the target objective
                function for penalties on number of primers and amplicon
                length, relative to the guide objective value. Tuple
                (A, B) where A weighs total number of primers and B weighs
                amplicon length
            only_account_for_amplified_seqs: design guides so as to account for
                only sequences bound by primers (e.g., when obj_type is 'min',
                cover only sequences bound by the primers), rather than
                all sequences; must be False if obj_type is 'max'
            halt_early: if True, stop as soon as there are the desired number
                of targets found, even if this does not complete the
                search over the whole genome (i.e., the targets meet the
                constraints but may not be optimal)
            obj_value_shift: amount by which to shift objective values
                before they are reported. This is only intended to avoid
                confusion about reported objective values -- e.g., if
                values for some targets would be positive and others
                negative, we can shift all values upward so they are all
                positive so that there is not confusion about differences
                between positive/negative. If None (default), defaults are
                set below based on obj_type.
        """
        self.ps = ps
        self.gs = gs

        if obj_type not in ['min', 'max']:
            raise ValueError(("obj_type must be 'min' or 'max'"))
        self.obj_type = obj_type

        self.max_primers_at_site = max_primers_at_site
        self.max_target_length = max_target_length

        if obj_weights is None:
            obj_weights = (0.50, 0.25)
        self.obj_weight_primers = obj_weights[0]
        self.obj_weight_length = obj_weights[1]

        if (only_account_for_amplified_seqs and
                isinstance(self.gs, guide_search.GuideSearcherMaximizeActivity)):
            # GuideSearcherMaximizeActivity only considers all sequences --
            # this is mostly for technical implementation reasons
            raise ValueError(("When maximizing activity, "
                "only_account_for_amplified_seqs must be False"))

        self.only_account_for_amplified_seqs = only_account_for_amplified_seqs

        self.halt_early = halt_early

        if obj_value_shift is None:
            if obj_type == 'min':
                # Values for all design options should all be positive, so
                # no need to shift
                self.obj_value_shift = 0
            elif obj_type == 'max':
                # Values for some design options may be positive and others
                # negative, so shift all up so that most are positive
                self.obj_value_shift = 4.0

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
            best_n: only store and output the best_n targets according
                to the objective
            no_overlap: if True, do not allow targets in the output
                whose amplicons (target range) overlap. When True,
                if a target overlaps with a target already in the output,
                this *replaces* the overlapping one with the new one
                if the new one has a better objective value. When False,
                many targets in the best_n may be very similar

        Returns:
            list of tuples (obj_value, target) where target is a tuple
            ((p1, p2), (guides_stats, guides)); (p1, p2) is a pair
            of PrimerResult objects, and guides is a list of guides that
            meet the constraints within the window bound by (p1, p2)
        """

        # Store a heap containing the best_n primer/guide targets;
        # store each as a tuple ((+/-)obj_value, push_id, target)
        # The heap is a min heap, so popped values are the ones with
        # the smallest value in the first element
        #   - When self.obj_type is 'min' we store -obj_value in the
        #     first element so that the one with the highest objective
        #     value (worst) is popped.
        #   - When self.obj_type is 'max' we store +obj_value in
        #     the first element so that the one with the smallest
        #     objective value (worst) is popped
        # In both cases, target_heap[0] refers to the target with the
        # worst objective value
        # We also store push_id (a counter on the push) to break ties
        # when costs are equivalent (choosing the one pushed first, i.e.,
        # the lower push_id); otherwise, there will be an error on
        # ties if target is not comparable
        target_heap = []
        push_id_counter = itertools.count()

        assert self.obj_type in ['min', 'max']
        def obj_value(i):
            # Return objective value of the i'th element
            if self.obj_type == 'min':
                return -1 * target_heap[i][0]
            elif self.obj_type == 'max':
                return target_heap[i][0]

        def obj_value_is_better(new, old):
            # Compare new objective value to old, and return True iff
            # new is better
            if self.obj_type == 'min':
                if new < old:
                    return True
            elif self.obj_type == 'max':
                if new > old:
                    return True
            return False

        # Determine a best-case objective value from the guide search
        # This is useful for pruning the search
        best_guide_obj_value = self.gs.best_obj_value()

        num_primer_pairs = 0
        num_suitable_primer_pairs = 0
        last_window_start = -1
        for p1, p2 in self._find_primer_pairs():
            if self.halt_early and len(target_heap) >= best_n:
                # Enough targets were found
                # Stop early without completing the search
                break

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
            cost_primers = self.obj_weight_primers * (p1_num + p2_num)

            # Calculate a cost of the window length
            cost_window = self.obj_weight_length * math.log2(window_length)

            # Calculate a best-case objective value for this target,
            # which can be done assuming a best-case for the guide search
            # This is useful for pruning the search
            if self.obj_type == 'min':
                best_possible_obj_value = (best_guide_obj_value +
                        cost_primers + cost_window)
            elif self.obj_type == 'max':
                best_possible_obj_value = (best_guide_obj_value -
                        cost_primers - cost_window)

            # Check if we should bother trying to find guides in this window
            if len(target_heap) >= best_n:
                curr_worst_obj_value = obj_value(0)
                if self.obj_type == 'min':
                    if best_possible_obj_value >= curr_worst_obj_value:
                        # The objective value is already >= than all in the
                        # current best_n, and will only stay the same or get
                        # bigger after adding the term for guides, so there is
                        # no reason to continue considering this window
                        continue
                elif self.obj_type == 'max':
                    if best_possible_obj_value <= curr_worst_obj_value:
                        # The objective value is <= all in the current best_n
                        # even in the best-case outcome for the guide search,
                        # and it will only stay the same or be smaller after
                        # adding the term for guides, so there is no reason to
                        # continue considering this window
                        continue

            # If targets should not overlap and this overlaps a target
            # already in target_heap, check if we should bother trying to
            # find guides in this window
            overlapping_i = []
            if no_overlap:
                # Check if any targets already in target_heap overlap
                # with this target
                # Note that this strategy does have the possibility of leading
                # to suboptimal solutions. Effectively, successive windows
                # can repeatedly remove other overlapping ones that might
                # otherwise be in the best_n. For example, consider targets
                # A, B, and C where A overlaps B and B overlaps C but A
                # does not overlap C. And say the true ranking is
                # A<B<C, but their objective values are close and better than
                # any other possible target option. Adding B would remove
                # A, and then adding C would remove B. So we are left with
                # only C from this set of targets, but ideally we would
                # like to have both A and C. One way around this might be
                # to keep a heap with >best_n options and only prune for
                # diverse solutions (non-overlap) later on.
                this_start = p1.start
                this_end = this_start + target_length
                for i in range(len(target_heap)):
                    _, _, target_i = target_heap[i]
                    (p1_i, p2_i), _ = target_i
                    i_start = p1_i.start
                    i_end = p2_i.start + p2_i.primer_length
                    if (this_start < i_end) and (i_start < this_end):
                        # Replace target_i
                        overlapping_i += [i]
            if overlapping_i:
                # target overlaps with one or more entries already in
                # target_heap
                could_replace = False
                for i in overlapping_i:
                    if obj_value_is_better(best_possible_obj_value,
                            obj_value(i)):
                        could_replace = True
                        break
                if could_replace is False:
                    # We will try to replace an entry in overlapping_i (i) with
                    # a new entry, but the objective value is such that, even
                    # in the best-case, it will not replace i. So there is
                    # no reason for considering this window (it will never
                    # replace i)
                    continue

            logger.info(("Found window [%d, %d) bound by primers that could "
                "be in the best %d targets; looking for guides within this; "
                "heap has %d valid targets currently"),
                window_start, window_end, best_n, len(target_heap))

            # Determine what set of sequences the guides should account for
            if self.only_account_for_amplified_seqs:
                # Find the sequences that are bound by some primer in p1 AND
                # some primer in p2
                # Note that this is only implemented when self.gs is an
                # instance of GuideSearcherMinimizeGuides (i.e., not for
                # MaximizeActivity -- mostly for technical implementation
                # reasons)
                p1_bound_seqs = self.ps.seqs_bound_by_primers(p1.primers_in_cover)
                p2_bound_seqs = self.ps.seqs_bound_by_primers(p2.primers_in_cover)
                primer_bound_seqs = p1_bound_seqs & p2_bound_seqs
                guide_seqs_to_consider = primer_bound_seqs
            else:
                # Design across the entire collection of sequences
                # This may lead to more guides being designed than if only
                # designing across primer_bound_seqs (below) because more
                # sequences must be considered in the design
                # But this can improve runtime because the seqs_to_consider
                # used by self.gs in the design will be more constant
                # across different windows, which will enable it to better
                # take advantage of memoization (in self.gs._memoized_guides)
                # during the design
                guide_seqs_to_consider = None

            # Find guides in the window
            if isinstance(self.gs, guide_search.GuideSearcherMinimizeGuides):
                extra_args = {'only_consider': guide_seqs_to_consider}
            else:
                extra_args = {}
            try:
                guides = self.gs._find_guides_in_window(
                    window_start, window_end, **extra_args)
            except guide_search.CannotAchieveDesiredCoverageError:
                # No more suitable guides; skip this window
                continue
            except guide_search.CannotFindAnyGuidesError:
                # No suitable guides in this window; skip it
                continue

            if len(guides) == 0:
                # This can happen, for example, if primer_bound_seqs is
                # empty; then no guides are required
                # Skip this window
                continue

            # Compute activities across target sequences, and expected, median,
            # and 5th percentile of activities
            if self.gs.predictor is not None:
                activities = self.gs.guide_set_activities(window_start,
                        window_end, guides)
                guides_activity_expected = self.gs.guide_set_activities_expected_value(
                        window_start, window_end, guides,
                        activities=activities)
                guides_activity_median, guides_activity_5thpctile = \
                        self.gs.guide_set_activities_percentile(
                                window_start, window_end, guides, [50, 5],
                                activities=activities)
            else:
                # There is no predictor to predict activities
                # This should only be the case if self.obj_type is 'min',
                # and may not necessarily be the case if self.obj_type is
                # 'min'
                activities = None
                guides_activity_expected = math.nan
                guides_activity_median, guides_activity_5thpctile = \
                        math.nan, math.nan
            # Calculate fraction of sequences bound by the guides
            if isinstance(self.gs, guide_search.GuideSearcherMinimizeGuides):
                guides_frac_bound = self.gs.total_frac_bound_by_guides(guides)
            elif isinstance(self.gs, guide_search.GuideSearcherMaximizeActivity):
                guides_frac_bound = self.gs.total_frac_bound_by_guides(
                        window_start, window_end, guides,
                        activities=activities)
            guides_stats = (guides_frac_bound, guides_activity_expected,
                    guides_activity_median, guides_activity_5thpctile)

            # Calculate a total objective value
            if self.obj_type == 'min':
                obj_value_total = (self.gs.obj_value(guides) +
                        cost_primers + cost_window)
                obj_value_to_add = -1 * obj_value_total
            elif self.obj_type == 'max':
                gs_obj_value = self.gs.obj_value(window_start, window_end,
                        guides, activities=activities)
                obj_value_total = (gs_obj_value -
                        cost_primers - cost_window)
                obj_value_to_add = obj_value_total

            # Add target to the heap (but only keep it if there are not
            # yet best_n targets or it has one of the best_n best objective
            # values)
            target = ((p1, p2), (guides_stats, guides))
            entry = (obj_value_to_add, next(push_id_counter), target)
            if overlapping_i:
                # target overlaps with an entry already in target_heap;
                # consider replacing that entry with new entry
                if len(overlapping_i) == 1:
                    # Almost all cases will satisfy this: the new entry
                    # overlaps just one existing entry; optimize for this case
                    i = overlapping_i[0]
                    if obj_value_is_better(obj_value_total, obj_value(i)):
                        # Replace i with entry
                        target_heap[i] = entry
                        heapq.heapify(target_heap)
                else:
                    # For some edge cases, the new entry overlaps with
                    # multiple existing entries
                    # Check if the new entry has a sufficiently low cost
                    # to justify removing any existing overlapping entries
                    obj_value_is_sufficiently_good = False
                    for i in overlapping_i:
                        if obj_value_is_better(obj_value_total, obj_value(i)):
                            obj_value_is_sufficiently_good = True
                            break
                    if obj_value_is_sufficiently_good:
                        # Remove all entries that overlap the new entry,
                        # then add the new entry
                        for i in sorted(overlapping_i, reverse=True):
                            del target_heap[i]
                        target_heap.append(entry)
                        heapq.heapify(target_heap)
                        # Note that this can lead to cases in which the size
                        # of the output (target_heap) is < best_n, or in which
                        # there is an entry with poor objective value, because
                        # in certain edge cases we might finish the search
                        # having removed >1 entry but only replacing it with
                        # one, and if the search continues for a short time
                        # after, len(target_heap) < best_n could result in
                        # some poor entries being placed into the heap
            elif len(target_heap) < best_n:
                # target_heap is not yet full, so push entry to it
                heapq.heappush(target_heap, entry)
            else:
                # target_heap is full; consider replacing the worst entry
                # with the new entry
                curr_worst_obj_value = obj_value(0)
                if obj_value_is_better(obj_value_total, curr_worst_obj_value):
                    # Push the entry into target_heap, and pop out the
                    # worst one
                    heapq.heappushpop(target_heap, entry)

        if len(target_heap) == 0:
            logger.warning(("Zero targets were found. The number of total "
                "primer pairs found was %d and the number of them that "
                "were suitable (passing basic criteria, e.g., on length) "
                "was %d"), num_primer_pairs, num_suitable_primer_pairs)

        # Invert the objective values if they were inverted (target_heap had
        # been storing the negative of each objective value for minimization),
        # toss push_id, and sort by objective value
        # In particular, sort by a 'sort_tuple' whose first element is
        # objective_value and then the target endpoints; it has the target
        # endpoints to break ties in objective value
        if self.obj_type == 'min':
            r = [(-obj_value, target) for obj_value, push_id, target in target_heap]
            r_with_sort_tuple = []
            for (obj_value, target) in r:
                ((p1, p2), (guides_stats, guides)) = target
                target_start = p1.start
                # Sort all fields of sort_tuple in ascending order
                sort_tuple = (obj_value, target_start)
                r_with_sort_tuple += [(sort_tuple, obj_value, target)]
            r_with_sort_tuple = sorted(r_with_sort_tuple, key=lambda x: x[0])
        elif self.obj_type == 'max':
            r = [(obj_value, target) for obj_value, push_id, target in target_heap]
            r_with_sort_tuple = []
            for (obj_value, target) in r:
                ((p1, p2), (guides_stats, guides)) = target
                target_start = p1.start
                # Sort obj_value in descending order, and target_start in
                # ascending order
                sort_tuple = (-1 * obj_value, target_start)
                r_with_sort_tuple += [(sort_tuple, obj_value, target)]
            r_with_sort_tuple = sorted(r_with_sort_tuple, key=lambda x: x[0])
        r = [(obj_value, target) for sort_tuple, obj_value, target in r_with_sort_tuple]

        # Shift obj_value
        r = [(obj_value + self.obj_value_shift, target)
                for obj_value, target in r]

        return r

    def find_and_write_targets(self, out_fn, best_n=10):
        """Find targets across an alignment, and write them to a file.

        This writes a table of the targets to a file, in which each
        row corresponds to a target in the genome (primer and guide
        sets).

        Args:
            out_fn: output TSV file to write targets
            best_n: only store and output the best_n targets according to
                objective value
        """
        targets = self.find_targets(best_n=best_n)

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['objective-value', 'target-start', 'target-end',
                'target-length',
                'left-primer-start', 'left-primer-num-primers',
                'left-primer-frac-bound', 'left-primer-target-sequences',
                'right-primer-start', 'right-primer-num-primers',
                'right-primer-frac-bound', 'right-primer-target-sequences',
                'num-guides', 'total-frac-bound-by-guides',
                'guide-set-expected-activity',
                'guide-set-median-activity', 'guide-set-5th-pctile-activity',
                'guide-expected-activities',
                'guide-target-sequences', 'guide-target-sequence-positions']) +
                '\n')

            for (obj_value, target) in targets:
                ((p1, p2), (guides_stats, guides)) = target

                # Break out guides_stats
                guides_frac_bound, guides_activity_expected, guides_activity_median, guides_activity_5thpctile = guides_stats

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

                # Find expected activity for each guide
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                if self.gs.predictor is not None:
                    expected_activities_per_guide = \
                            [self.gs.guide_activities_expected_value(
                                window_start, window_end, gd_seq)
                                for gd_seq in guides_seqs_sorted]
                else:
                    # There is no predictor to predict activities
                    # This should only be the case if self.obj_type is 'min',
                    # and may not necessarily be the case if self.obj_type is
                    # 'min'
                    expected_activities_per_guide = \
                            [math.nan for gd_seq in guides_seqs_sorted]
                expected_activities_per_guide_str = ' '.join(
                        str(a) for a in expected_activities_per_guide)

                line = [obj_value, target_start, target_end, target_length,
                    p1.start, p1.num_primers, p1.frac_bound, p1_seqs_str,
                    p2.start, p2.num_primers, p2.frac_bound, p2_seqs_str,
                    len(guides), guides_frac_bound, guides_activity_expected,
                    guides_activity_median, guides_activity_5thpctile,
                    expected_activities_per_guide_str,
                    guides_seqs_str, guides_positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')


class DesignTarget:
    """Store information on a design of a single target.
    """

    def __init__(self, target_start, target_end, guide_seqs,
            left_primer_seqs, right_primer_seqs, obj_value):
        self.target_start = target_start
        self.target_end = target_end
        self.guide_seqs = tuple(sorted(guide_seqs))
        self.left_primer_seqs = tuple(sorted(left_primer_seqs))
        self.right_primer_seqs = tuple(sorted(right_primer_seqs))
        self.obj_value = obj_value

    @staticmethod
    def read_design_targets(fn, num_targets=None):
        """Read a collection of targets from a file.

        Args:
            fn: path to a TSV file giving targets
            num_targets: only construct a Design from the top num_targets
                targets, as ordered by cost (if None, use all)

        Returns:
            list of DesignTarget objects
        """
        if not os.path.isfile(fn):
            return None

        rows = []
        has_objective_value = False
        with open(fn) as f:
            col_names = {}
            for i, line in enumerate(f):
                line = line.rstrip()
                ls = line.split('\t')
                if i == 0:
                    # Parse header
                    for j in range(len(ls)):
                        col_names[j] = ls[j]
                else:
                    # Read each column as a variable
                    cols = {}
                    for j in range(len(ls)):
                        cols[col_names[j]] = ls[j]
                    if has_objective_value:
                        rows += [(cols['objective-value'], cols['target-start'],
                                 cols['target-end'], cols)]
                    else:
                        rows += [cols]

        # Pull out the best N targets, assuming rows are already sorted
        # as desired
        if num_targets != None:
            if has_objective_value is False:
                raise Exception(("Cannot pull out best targets; objective "
                    "value is not given"))
            if len(rows) < num_targets:
                raise Exception(("The number of rows in a design (%d) is fewer "
                    "than the number of targets to read (%d)") %
                    (len(rows), num_targets))
            rows = rows[:num_targets]

        targets = []
        for row in rows:
            if has_objective_value:
                _, _, _, cols = row
                obj_value = float(cols['objective-value'])
            else:
                cols = row
                obj_value = None
            targets += [DesignTarget(
                int(cols['target-start']),
                int(cols['target-end']),
                cols['guide-target-sequences'].split(' '),
                cols['left-primer-target-sequences'].split(' '),
                cols['right-primer-target-sequences'].split(' '),
                obj_value
            )]

        return targets
