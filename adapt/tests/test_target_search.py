"""Tests for target_search module.
"""

import logging
import random
import unittest

from adapt import alignment
from adapt import guide_search
from adapt import primer_search
from adapt import target_search

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
        a_gs = guide_search.GuideSearcherMinimizeGuides(
            self.a_aln, 6, 0, 1.0, (1, 1, 100))
        self.a_min = target_search.TargetSearcher(a_ps, a_gs, obj_type='min',
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
                (p1, p2), (guides_frac_bound, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start

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
            self.assertEqual(len(targets), best_n)

            for cost, target in targets:
                (p1, p2), (guides_frac_bound, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start

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

    def test_find_targets_with_cover_frac_minmize(self):
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
                (p1, p2), (guides_frac_bound, guides) = target
                window_start = p1.start + p1.primer_length
                window_end = p2.start
                window_length = window_end - window_start

                # Windows must be at least the guide length (6)
                self.assertGreaterEqual(window_length, 6)

                # For up to the top 6 targets, only 1 primer on each
                # end is needed
                self.assertEqual(p1.num_primers, 1)
                self.assertEqual(p2.num_primers, 1)

                # The guides should not cover the last sequence in b_seqs,
                # so the frac_bound should be <1.0
                self.assertLess(guides_frac_bound, 1.0)

    # TODO test with maximization; add mock activity model to GuideSearcher
    # that returns 1 iff guide-target match

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET) 
