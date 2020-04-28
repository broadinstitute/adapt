"""Tests for target_search module.
"""

import logging
import random
import unittest

from adapt import alignment
from adapt import guide_search
from adapt import primer_search
from adapt import target_search
from adapt.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestTargetSearch(unittest.TestCase):
    """Tests methods in the TargetSearch class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAATGTACGGTCAACATTCTCACCTATGGATGCAGTGA',
                       'ATCGAATGTACGGTCAACATTCTCACCTATGGATGCAGTGA',
                       'GGGGAATGTACGGTCGGGGTTCTCACCTATGGCCCCAGTGA',
                       'CCCCAATGTACGGTCCCCCTTCTCACCTATGGGGGGAGTGA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        a_ps = primer_search.PrimerSearcher(
            self.a_aln, 4, 0, 1.0, (1, 1, 100))

        a_min_gs = guide_search.GuideSearcherMinimizeGuides(
            self.a_aln, 6, 0, 1.0, (1, 1, 100))
        self.a_min = target_search.TargetSearcher(a_ps, a_min_gs, obj_type='min',
            max_primers_at_site=2)

    def test_find_primer_pairs_simple_minimize(self):
        suitable_primer_sites = [3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22,
                                 23, 24, 25, 26, 27, 28, 35, 36, 37]
        for p1, p2 in self.a_min._find_primer_pairs():
            # Primers should be a suitable sites, and p2 must come
            # after p1
            self.assertIn(p1.start, suitable_primer_sites)
            self.assertIn(p2.start, suitable_primer_sites)
            self.assertGreater(p2.start, p1.start)

            # We required coverage of 1.0 for primers
            self.assertEqual(p1.frac_bound, 1.0)
            self.assertEqual(p2.frac_bound, 1.0)

            # We allowed at most 2 primers at each site
            self.assertLessEqual(p1.num_primers, 2)
            self.assertLessEqual(p2.num_primers, 2)

            # Verify that the primer sequences are in the alignment
            # at their given locations
            for p in (p1, p2):
                for primer in p.primers_in_cover:
                    in_aln = False
                    for seq in self.a_seqs:
                        if primer == seq[p.start:(p.start + 4)]:
                            in_aln = True
                            break
                    self.assertTrue(in_aln)

    def test_find_targets_allowing_overlap_minmize(self):
        for best_n in [1, 2, 3, 4, 5, 6]:
            targets = self.a_min.find_targets(best_n=best_n, no_overlap=False)
            self.assertEqual(len(targets), best_n)

            for cost, target in targets:
                (p1, p2), (guides_stats, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start
                guides_frac_bound, _, _, _ = guides_stats

                # All windows are at least 10 nt long; verify this
                self.assertGreaterEqual(window_length, 10)

                # For up to the top 6 targets, only 1 primer on each
                # end and 1 guide is needed
                self.assertEqual(p1.num_primers, 1)
                self.assertEqual(p2.num_primers, 1)
                self.assertEqual(len(guides), 1)

                # The guides should cover all sequences
                self.assertEqual(guides_frac_bound, 1.0)

    def test_find_targets_without_overlap_minimize(self):
        for best_n in [1, 2, 3, 4, 5, 6]:
            targets = self.a_min.find_targets(best_n=best_n, no_overlap=True)

            # It is possible the number of targets is slightly
            # less than best_n
            self.assertLessEqual(len(targets), best_n)

            for cost, target in targets:
                (p1, p2), (guides_stats, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start
                guides_frac_bound, _, _, _ = guides_stats

                # All windows are at least 10 nt long; verify this
                # Since here targets cannot overlap, some may be
                # shorter than 10, but all must be at least the
                # guide length (6)
                self.assertGreaterEqual(window_length, 6)

                # For up to the top 6 targets, only 1 primer on each
                # end is needed
                self.assertEqual(p1.num_primers, 1)
                self.assertEqual(p2.num_primers, 1)

                # The guides should cover all sequences
                self.assertEqual(guides_frac_bound, 1.0)

    def test_find_targets_with_cover_frac_minimize(self):
        b_seqs = ['ATCGAATGTACGGTCAACATTCTCACCTATGGATGCAGTGA',
                  'ATCGAATGTACGGTCAACATTCTCACCTATGGATGCAGTGA',
                  'GGGGAATGTACGGTCGGGGTTCTCACCTATGGCCCCAGTGA',
                  'CCCCAATGTACGGTCCCCCTTCTCACCTATGGGGGGAGTGA',
                  'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA']
        b_aln = alignment.Alignment.from_list_of_seqs(b_seqs)

        cover_frac = {0: 1.0, 1: 0.01}
        seq_groups = {0: {0}, 1: {1, 2, 3, 4}}
        b_ps = primer_search.PrimerSearcher(
            b_aln, 4, 0, cover_frac, (1, 1, 100), seq_groups=seq_groups)
        b_gs = guide_search.GuideSearcherMinimizeGuides(
            b_aln, 6, 0, cover_frac, (1, 1, 100), seq_groups=seq_groups)
        b = target_search.TargetSearcher(b_ps, b_gs, obj_type='min',
            max_primers_at_site=2)

        for best_n in [1, 2, 3, 4, 5, 6]:
            targets = b.find_targets(best_n=best_n, no_overlap=False)
            self.assertEqual(len(targets), best_n)

            for cost, target in targets:
                (p1, p2), (guides_stats, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start
                guides_frac_bound, _,  _, _ = guides_stats

                # Windows must be at least the guide length (6)
                self.assertGreaterEqual(window_length, 6)

                # For up to the top 6 targets, only 1 primer on each
                # end is needed
                self.assertEqual(p1.num_primers, 1)
                self.assertEqual(p2.num_primers, 1)

                # The guides should not cover the last sequence in b_seqs,
                # so the frac_bound should be <1.0
                self.assertLess(guides_frac_bound, 1.0)

    def find_targets_allowing_overlap_maximize(self, algo,
            soft_guide_constraint, hard_guide_constraint, penalty_strength,
            check_for_one_guide=False, check_for_no_targets=False):
        # Predict guides matching target to have activity 1, and
        # starting with 'A' to have activity 3 (otherwise, 0)
        class PredictorTest:
            def __init__(self):
                self.context_nt = 1
                self.rough_max_activity = 3
            def compute_activity(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    target_without_context = target[self.context_nt:len(target)-self.context_nt]
                    if guide == target_without_context:
                        if guide[0] == 'A':
                            y += [3]
                        else:
                            y += [1]
                    else:
                        y += [0]
                return y
            def cleanup_memoized(self, pos):
                pass
        predictor = PredictorTest()

        a_ps = primer_search.PrimerSearcher(
            self.a_aln, 4, 0, 1.0, (1, 1, 100))

        a_max_gs = guide_search.GuideSearcherMaximizeActivity(
            self.a_aln, 6,
            soft_guide_constraint,
            hard_guide_constraint,
            penalty_strength,
            (1, 1, 100),
            algorithm=algo,
            predictor=predictor)

        # Change the guide_clusterer to have a smaller k and thus be
        # more sensitive in clustering
        a_max_gs.guide_clusterer = alignment.SequenceClusterer(
                lsh.HammingDistanceFamily(6), k=6)

        a_max = target_search.TargetSearcher(a_ps, a_max_gs, obj_type='max',
            max_primers_at_site=2)

        for best_n in [1, 2, 3, 4, 5, 6]:
            targets = a_max.find_targets(best_n=best_n, no_overlap=False)

            if check_for_no_targets:
                self.assertEqual(len(targets), 0)
            else:
                self.assertEqual(len(targets), best_n)

            for rank, (obj_value, target) in enumerate(targets):
                (p1, p2), (guides_stats, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start
                guides_frac_bound, guides_activity_expected, guides_activity_median, guides_activity_5thpctile = guides_stats

                # All windows are at least 10 nt long; verify this
                # Since here targets cannot overlap, some may be
                # shorter than 10, but all must be at least the
                # guide length (6)
                self.assertGreaterEqual(window_length, 6)

                # For up to the top 6 targets, only 1 primer on each
                # end is needed
                self.assertEqual(p1.num_primers, 1)
                self.assertEqual(p2.num_primers, 1)

                # The hard constraint should be met
                self.assertGreaterEqual(len(guides), 1)
                self.assertLessEqual(len(guides), hard_guide_constraint)

                # The number of guides should be at least the soft constraint
                # But if the 5th percentile of activity across sequences is
                # already >0, it is possible that nothing can be gained by
                # adding more guides, in which case this condition would
                # not be met
                # Also, if the algorithm is random-greedy, this condition
                # might not be met due to randomness, so only check if
                # the algorithm is greedy
                if algo == 'greedy' and not guides_activity_5thpctile > 0:
                    self.assertGreaterEqual(len(guides), soft_guide_constraint)

                if check_for_one_guide:
                    # The parameters were set such that there should only
                    # be 1 guide
                    self.assertEqual(len(guides), 1)

                # The median activity should be at least 1
                # (unless there is only 1 guide, in which case the
                # median may be 0)
                if len(guides) > 1:
                    self.assertGreaterEqual(guides_activity_median, 1.0)

                # The guides should detect at least half the sequences
                self.assertGreaterEqual(guides_frac_bound, 0.5)

                # Since guides starting with 'A' have higher activity,
                # the guide set should contain at least one such guide
                # But only check this for the highest ranked target; for
                # lower ranked targets, there may not be an option for
                # such a guide
                if rank == 0:
                    num_with_A_start = sum(1 for g in guides if g[0] == 'A')
                    self.assertGreaterEqual(num_with_A_start, 1)

    def test_find_targets_allowing_overlap_maximize_greedy(self):
        for hard_guide_constraint in [1, 2, 3, 4, 5]:
            for soft_guide_constraint in range(1, hard_guide_constraint+1):
                self.find_targets_allowing_overlap_maximize(
                        'greedy', soft_guide_constraint,
                        hard_guide_constraint, 0.05)

        # Try with high penalty strengths
        self.find_targets_allowing_overlap_maximize(
                'greedy', 1, 3, 10, check_for_one_guide=True)
        self.find_targets_allowing_overlap_maximize(
                'greedy', 1, 3, 100, check_for_one_guide=True)

    def test_find_targets_allowing_overlap_maximize_random_greedy(self):
        for hard_guide_constraint in [1, 2, 3, 4, 5]:
            for soft_guide_constraint in range(1, hard_guide_constraint+1):
                self.find_targets_allowing_overlap_maximize(
                        'random-greedy', soft_guide_constraint,
                        hard_guide_constraint, 0.05)

        # Try with high penalty strengths
        # With random-greedy, there should be 0 targets because there
        # will no ground set at any site (no guide will meet the
        # non-negativity threshold)
        self.find_targets_allowing_overlap_maximize(
                'random-greedy', 1, 3, 10, check_for_one_guide=True,
                check_for_no_targets=True)
        self.find_targets_allowing_overlap_maximize(
                'random-greedy', 1, 3, 100, check_for_one_guide=True,
                check_for_no_targets=True)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET) 
