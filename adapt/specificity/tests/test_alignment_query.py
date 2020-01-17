"""Tests for alignment_query module.
"""

import random
import unittest

from adapt import alignment
from adapt.specificity import alignment_query

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignmentQuerier(unittest.TestCase):
    """Tests the AlignmentQuerier class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        aln_a_seqs = ['ATCGAAATAAACC',
                      'ATCGATATAATGG',
                      'ATGGTTATAATGG']
        aln_a = alignment.Alignment.from_list_of_seqs(aln_a_seqs)

        aln_b_seqs = ['GTTCCTTTACAGT',
                      'GCTCCTTCACAGT',
                      'GCTCCTTTCCCGT']
        aln_b = alignment.Alignment.from_list_of_seqs(aln_b_seqs)

        aln_c_seqs = ['ATCGACATAGTCG',
                      'ATCGACATAATGG',
                      'CTGGTCATAACCC']
        aln_c = alignment.Alignment.from_list_of_seqs(aln_c_seqs)

        alns = [aln_a, aln_b, aln_c]
        self.aq = alignment_query.AlignmentQuerier(alns, 5, 1, False, k=3,
            reporting_prob=0.95)
        self.aq.setup()

    def test_frac_of_aln_hit_by_guide(self):
        self.assertEqual(self.aq.frac_of_aln_hit_by_guide('ATCGA'),
                         [2.0/3, 0.0, 2.0/3])

        self.assertEqual(self.aq.frac_of_aln_hit_by_guide('GGGGG'),
                         [0.0, 0.0, 0.0])

        self.assertEqual(self.aq.frac_of_aln_hit_by_guide('CCTTC'),
                         [0.0, 1.0, 0.0])

        self.assertEqual(self.aq.frac_of_aln_hit_by_guide('TTACA'),
                         [1.0/3, 2.0/3, 0.0])

    def test_guide_is_specific_to_aln(self):
        def assert_is_specific(guide, thres, specific_to):
            not_specific_to = [i for i in [0, 1, 2] if i not in specific_to]
            for i in specific_to:
                self.assertTrue(self.aq.guide_is_specific_to_alns(guide, [i], thres))
            for i in not_specific_to:
                self.assertFalse(self.aq.guide_is_specific_to_alns(guide, [i], thres))

        assert_is_specific('ATCGA', 0, [])
        assert_is_specific('GGGGG', 0, [0, 1, 2])
        assert_is_specific('CCTTC', 0, [1])
        assert_is_specific('TTACA', 0, [])

        assert_is_specific('ATCGA', 0.5, [])
        assert_is_specific('GGGGG', 0.5, [0, 1, 2])
        assert_is_specific('CCTTC', 0.5, [1])
        assert_is_specific('TTACA', 0.5, [1])

    def test_hit_with_gaps(self):
        gappy_aln_seqs = ['ATCGACGTAAACC',
                          'ATCGA--TAATGG',
                          'ATGGA--TAATGG']
        gappy_aln = alignment.Alignment.from_list_of_seqs(gappy_aln_seqs)
        gappy_aq = alignment_query.AlignmentQuerier([gappy_aln], 5, 1, False, k=3,
            reporting_prob=0.95)
        gappy_aq.setup()

        # GGGGG should hit no sequences
        self.assertEqual(gappy_aq.frac_of_aln_hit_by_guide('GGGGG'),
                         [0.0])

        # GACGT should only hit the first sequence
        self.assertEqual(gappy_aq.frac_of_aln_hit_by_guide('GACGT'),
                         [1.0/3.0])

        # GATAA should hit the second and third sequence (since their
        # gaps are removed)
        self.assertEqual(gappy_aq.frac_of_aln_hit_by_guide('GATAA'),
                         [2.0/3.0])

        # CATAA should hit all 3 alignments (tolerating a mismatch
        # to CGTAA in the first, and with perfect match in the second
        # and third after their gaps are removed
        self.assertEqual(gappy_aq.frac_of_aln_hit_by_guide('CATAA'),
                         [1.0])
