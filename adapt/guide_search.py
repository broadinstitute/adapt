"""Methods for searching for optimal guides to use for a diagnostic.
"""

from collections import defaultdict
import logging
import math
import random

import numpy as np

from adapt import alignment
from adapt.utils import oligo
from adapt.utils import search
from adapt.utils import index_compress
from adapt.utils import lsh
from adapt.utils import predict_activity

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)

class GuideSearcher(search.OligoSearcher):
    """Methods to search for guides to use for a diagnostic.

    This is a base class, with subclasses defining methods depending on the
    objective. It should not be used without subclassing, as it does not define
    all the positional arguments necessary for search.OligoSearcher.

    """
    def __init__(self, guide_length, **kwargs):
        """
        Args:
            guide_length: integer length of the guide. Sets guide length to
                min_oligo_length and max_oligo_length.
            kwargs: see search.OligoSearcher.__init__()
        """
        if 'min_oligo_length' in kwargs or 'max_oligo_length' in kwargs:
            raise ValueError("Variable oligo lengths are not yet implemented "
                "for guides; you cannot use min_oligo_length or "
                "max_oligo_length.")
        super().__init__(min_oligo_length=guide_length,
            max_oligo_length=guide_length, **kwargs)


class GuideSearcherMinimizeGuides(search.OligoSearcherMinimizeNumber,
        GuideSearcher):
    """Methods to minimize the number of guides.
    """

    def __init__(self, aln, guide_length, mismatches, cover_frac,
            missing_data_params, **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            mismatches: threshold on number of mismatches for determining
                whether a guide would hybridize to a target sequence
            cover_frac: fraction in (0, 1] of sequences that must be 'captured'
                by a guide; see seq_groups
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
            kwargs: see OligoSearcherMinimizeNumber.init() and
                GuideSearcher.__init__()
        """
        super().__init__(aln=aln, guide_length=guide_length,
            mismatches=mismatches, missing_data_params=missing_data_params,
            cover_frac=cover_frac, **kwargs)

    def _compress_result(self, p):
        """Compress the information to be stored in self._memo

        Args:
            p: result of calling construct_oligo()

        Returns:
            compressed version of p
        """
        gd, covered_seqs, score = p

        # covered_seqs may contain mostly contiguous indices
        covered_seqs_compressed = index_compress.compress_mostly_contiguous(covered_seqs)

        return (gd, covered_seqs_compressed, score)

    def _decompress_result(self, p_compressed):
        """"Decompress the information stored in self._memo

        Args:
            p_compressed: output of _compress_result()

        Returns:
            decompressed version of p_compressed
        """
        gd, covered_seqs_compressed, score = p_compressed

        # Decompress covered_seqs
        covered_seqs = index_compress.decompress_ranges(covered_seqs_compressed)

        return (gd, covered_seqs, score)

    def construct_oligo(self, start, oligo_length, seqs_to_consider,
            num_needed=None, stop_early=True):
        """Construct a single guide to target a set of sequences in the alignment.

        This constructs a guide to target sequence within the range [start,
        start+guide_length]. It only considers the sequences with indices given in
        seqs_to_consider.

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: dict mapping universe group ID to collection of
                indices to use when constructing the guide
            num_needed: dict mapping universe group ID to the number of sequences
                from the group that are left to cover in order to achieve
                a desired coverage; these are used to help construct a
                guide
            stop_early: if True, impose early stopping criteria while iterating
                over clusters to improve runtime

        Returns:
            tuple (x, y, z) where:
                x is the sequence of the constructed guide
                y is a set of indices of sequences (a subset of
                    values in seqs_to_consider) to which the guide x will
                    hybridize
                z is the marginal contribution of the guide to the objective
            (Note that it is possible that x binds to no sequences and that
            y will be empty.)
        """
        # TODO: There are several optimizations that can be made to
        # this function that take advantage of G-U pairing in order
        # to lower the number of guides that need to be designed.
        # Two are:
        #  1) The function SequenceClusterer.cluster(..), which is
        #     used here, can cluster accounting for G-U pairing (e.g.,
        #     such that 'A' hashes to 'G' and 'C' hashes to 'T',
        #     so that similar guide sequences hash to the same
        #     value tolerating G-U similarity).
        #  2) Instead of taking a consensus sequence across a
        #     cluster, this can take a pseudo-consensus that
        #     accounts for G-U pairing (e.g., in which 'A' is
        #     treated as 'G' and 'C' is treated as 'T' for
        #     the purposes of generating a consensus sequence).

        assert start + oligo_length <= self.aln.seq_length
        assert len(seqs_to_consider) > 0

        for pos in range(start, start + oligo_length):
            if self.aln.frac_missing_at_pos(pos) > self.missing_threshold:
                raise alignment.CannotConstructOligoError(("Too much missing "
                    "data at a position in the target range"))

        aln_for_guide = self.aln.extract_range(start, start + oligo_length)

        if self.predictor is not None:
            # Extract the target sequences, including context to use with
            # prediction
            if (start - self.predictor.context_nt < 0 or
                    start + oligo_length + self.predictor.context_nt >
                    self.aln.seq_length):
                raise alignment.CannotConstructOligoError(("The context needed "
                    "for the target to predict activity falls outside the "
                    "range of the alignment at this position"))
            aln_for_guide_with_context = self.aln.extract_range(
                    start - self.predictor.context_nt,
                    start + oligo_length + self.predictor.context_nt)

        # Before modifying seqs_to_consider, make a copy of it
        seqs_to_consider_cp = {}
        for group_id in seqs_to_consider.keys():
            seqs_to_consider_cp[group_id] = set(seqs_to_consider[group_id])
        seqs_to_consider = seqs_to_consider_cp

        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Ignore any sequences in the alignment that have a gap in
        # this region
        if self.predictor is not None:
            seqs_with_gap = set(aln_for_guide_with_context.seqs_with_gap(
                all_seqs_to_consider))
        else:
            seqs_with_gap = set(aln_for_guide.seqs_with_gap(
                all_seqs_to_consider))
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].difference_update(seqs_with_gap)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Only consider sequences in the alignment that have the
        # required flanking sequence(s)
        seqs_with_flanking = self.aln.seqs_with_required_flanking(
            start, oligo_length, self.required_flanking_seqs,
            seqs_to_consider=all_seqs_to_consider)
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].intersection_update(seqs_with_flanking)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # If every sequence in this region has a gap or does not contain
        # required flanking sequence, then there are none left to consider
        if len(all_seqs_to_consider) == 0:
            raise alignment.CannotConstructOligoError(("All sequences in region "
                "have a gap and/or do not contain required flanking sequences"))

        seq_rows = aln_for_guide.make_list_of_seqs(all_seqs_to_consider,
            include_idx=True)
        if self.predictor is not None:
            seq_rows_with_context = aln_for_guide_with_context.make_list_of_seqs(
                    all_seqs_to_consider, include_idx=True)

        if self.predictor is not None:
            # Memoize activity evaluations
            pair_eval = {}
        def determine_binding_and_active_seqs(gd_sequence):
            binding_seqs = set()
            num_bound = 0
            if self.predictor is not None:
                num_passed_predict_active = 0
                # Determine what calls to make to
                # self.predictor.determine_highly_active(); it
                # is best to batch these
                pairs_to_eval = []
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        pairs_to_eval += [pair]
                # Evaluate activity
                evals = self.predictor.determine_highly_active(start, pairs_to_eval)
                for pair, y in zip(pairs_to_eval, evals):
                    pair_eval[pair] = y
                # Fill in binding_seqs
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        num_bound += 1
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        if pair_eval[pair]:
                            num_passed_predict_active += 1
                            binding_seqs.add(seq_idx)
            else:
                num_passed_predict_active = None
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        num_bound += 1
                        binding_seqs.add(seq_idx)
            return binding_seqs, num_bound, num_passed_predict_active

        # Define a score function (higher is better) for a collection of
        # sequences covered by a guide
        if num_needed is not None:
            # This is the number of sequences it contains that are needed to
            # achieve the partial cover; we can compute this by summing over
            # the number of needed sequences it contains, taken across the
            # groups in the universe
            # Memoize the scores because this computation might be expensive
            seq_idxs_scores = {}
            def seq_idxs_score(seq_idxs):
                seq_idxs = set(seq_idxs)
                tc = tuple(seq_idxs)
                if tc in seq_idxs_scores:
                    return seq_idxs_scores[tc]
                score = 0
                for group_id, needed in num_needed.items():
                    contained_in_seq_idxs = seq_idxs & seqs_to_consider[group_id]
                    score += min(needed, len(contained_in_seq_idxs))
                seq_idxs_scores[tc] = score
                return score
        else:
            # Score by the number of sequences it contains
            def seq_idxs_score(seq_idxs):
                return len(seq_idxs)

        # First construct the optimal guide to cover the sequences. This would be
        # a string x that maximizes the number of sequences s_i such that x and
        # s_i are equal to within 'mismatches' mismatches; it's called the "max
        # close string" or "close to most strings" problem. For simplicity, let's
        # do the following: cluster the sequences (just the portion with
        # potential guides) with LSH, choose a guide for each cluster to be
        # the consensus of that cluster, and choose the one (across clusters)
        # that has the highest score
        if self.clusterer is None:
            # Don't cluster; instead, draw the consensus from all the
            # sequences. Effectively, treat it as there being just one
            # cluster, which consists of all sequences to consider
            clusters_ordered = [all_seqs_to_consider]
        else:
            # Cluster the sequences
            clusters = self.clusterer.cluster(seq_rows)

            # Sort the clusters by score, from highest to lowest
            # Here, the score is determined by the sequences in the cluster
            clusters_ordered = sorted(clusters, key=seq_idxs_score, reverse=True)

        best_gd = None
        best_gd_binding_seqs = None
        best_gd_score = 0
        stopped_early = False
        for cluster_idxs in clusters_ordered:
            if stop_early and best_gd_score > seq_idxs_score(cluster_idxs):
                # The guide from this cluster is unlikely to exceed the current
                # score; stop early
                stopped_early = True
                break

            gd = aln_for_guide.determine_consensus_sequence(
                cluster_idxs)
            if 'N' in gd:
                # Skip this; all sequences at a position in this cluster
                # are 'N'
                continue
            skip_cluster = False
            for is_suitable_fn in self.is_suitable_fns:
                if is_suitable_fn(gd) is False:
                    # Skip this cluster
                    skip_cluster = True
                    break
            if skip_cluster:
                continue
            # Determine the sequences that are bound by this guide (and
            # where it is 'active', if self.predictor is set)
            binding_seqs, num_bound, num_passed_predict_active = \
                    determine_binding_and_active_seqs(gd)
            score = seq_idxs_score(binding_seqs)
            if score > best_gd_score:
                best_gd = gd
                best_gd_binding_seqs = binding_seqs
                best_gd_score = score

            # Impose an early stopping criterion if self.predictor is
            # used, because using it is slow
            if self.predictor is not None and stop_early:
                if (num_bound >= 0.5*len(cluster_idxs) and
                        num_passed_predict_active < 0.1*len(cluster_idxs)):
                    # gd binds (according to guide.guide_binds()) many
                    # sequences, but is not predicted to be active against
                    # many; it is likely that this region is poor according to
                    # the self.predictor (e.g., due to sequence composition).
                    # Rather than trying other clusters at this site, just skip
                    # it. Note that this is just a heuristic; it can help
                    # runtime when the self.predictor is used, but is not needed
                    # and may hurt the optimality of the solution
                    stopped_early = True
                    break
        gd = best_gd
        binding_seqs = best_gd_binding_seqs
        score = best_gd_score

        # It's possible that the consensus sequence (guide) of no cluster
        # binds to any of the sequences. In this case, simply go through all
        # sequences and find the first that has no ambiguity and is suitable
        # and active, and make this the guide
        if gd is None and not stopped_early:
            for i, (s, idx) in enumerate(seq_rows):
                if not set(s).issubset(set(['A', 'C', 'G', 'T'])):
                    # s has ambiguity; skip it
                    continue
                skip_cluster = False
                for is_suitable_fn in self.is_suitable_fns:
                    if is_suitable_fn(s) is False:
                        # Skip this cluster
                        skip_cluster = True
                        break
                if skip_cluster:
                    continue
                if self.predictor is not None:
                    s_with_context, _ = seq_rows_with_context[i]
                    if not self.predictor.determine_highly_active(start,
                            [(s_with_context, s)])[0]:
                        # s is not active against itself; skip it
                        continue
                # s has no ambiguity and is a suitable guide; use it
                gd = s
                binding_seqs, _, _ = determine_binding_and_active_seqs(gd)
                break

        if gd is None:
            raise alignment.CannotConstructOligoError(("No guides are suitable "
                "or active"))

        return (gd, binding_seqs, score)

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
        guide_collections = list(self._find_oligos_for_each_window(
            window_size, window_step=window_step))

        if sort:
            # Sort by number of guides ascending (len(x[2])), then by
            # score of guides descending
            guide_collections.sort(key=lambda x: (len(x[2]),
                -self._score_collection(x[2])))

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'score', 'total-frac-bound', 'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, guide_seqs = guides_in_window
                score = self._score_collection(guide_seqs)
                frac_bound = self.total_frac_bound(guide_seqs)
                count = len(guide_seqs)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_positions[gd_seq]
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


class GuideSearcherMaximizeActivity(search.OligoSearcherMaximizeActivity,
        GuideSearcher):
    """Methods to maximize expected activity of the guide set.
    """

    def __init__(self, aln, guide_length, soft_guide_constraint,
            hard_guide_constraint, penalty_strength,
            missing_data_params, **kwargs):
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
            kwargs: see OligoSearcherMaximizeActivity.init() and
                GuideSearcher.__init__()
        """
        super().__init__(aln=aln, guide_length=guide_length,
            soft_constraint=soft_guide_constraint,
            hard_constraint=hard_guide_constraint,
            penalty_strength=penalty_strength,
            missing_data_params=missing_data_params, **kwargs)

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
        guide_collections = list(self._find_oligos_for_each_window(
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
                activities = self.oligo_set_activities(start, end, guide_seqs)
                obj = self.obj_value(start, end, guide_seqs,
                        activities=activities)
                frac_bound = self.total_frac_bound(start, end,
                        guide_seqs, activities=activities)
                guides_activity_expected = self.oligo_set_activities_expected_value(
                        start, end, guide_seqs, activities=activities)
                guides_activity_median, guides_activity_5thpctile = \
                        self.oligo_set_activities_percentile(start, end,
                                guide_seqs, [50, 5], activities=activities)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_positions[gd_seq]
                             for gd_seq in guide_seqs_sorted]
                positions_str = ' '.join(str(p) for p in positions)
                expected_activities_per_guide = \
                        [self.oligo_activities_expected_value(start, end,
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
