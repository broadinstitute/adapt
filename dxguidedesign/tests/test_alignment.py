"""Tests for alignment module.
"""

import random
import unittest

from dxguidedesign import alignment

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignment(unittest.TestCase):
    """Tests methods in the Alignment class.
    """

    def setUp(self):
        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a = alignment.Alignment.from_list_of_seqs(self.a_seqs)

        self.b_seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.b = alignment.Alignment.from_list_of_seqs(self.b_seqs)

        self.c_seqs = ['ATCGAA', 'ATC-AA']
        self.c = alignment.Alignment.from_list_of_seqs(self.c_seqs)

    def test_make_list_of_seqs_a(self):
        self.assertEqual(self.a.make_list_of_seqs(),
                         self.a_seqs)
        self.assertEqual(self.a.make_list_of_seqs([0]),
                         ['ATCGAA'])
        self.assertEqual(self.a.make_list_of_seqs([0,3]),
                         ['ATCGAA', 'AYCGAT'])

    def test_make_list_of_seqs_b(self):
        self.assertEqual(self.b.make_list_of_seqs(),
                         self.b_seqs)

    def test_make_list_of_seqs_c(self):
        self.assertEqual(self.c.make_list_of_seqs(),
                         self.c_seqs)

    def test_determine_consensus_sequence_a(self):
        self.assertEqual(self.a.determine_consensus_sequence(), 'ATCGAA')
        self.assertEqual(self.a.determine_consensus_sequence([0]), 'ATCGAA')
        self.assertIn(self.a.determine_consensus_sequence([2]),
                      ['ATCGAA', 'ACCGAA'])
        self.assertIn(self.a.determine_consensus_sequence([0,1]),
                      ['ATCGAA', 'ATCGAT'])
        self.assertIn(self.a.determine_consensus_sequence([1,2]),
                      ['ATCGAT', 'ATCGAA'])

    def test_determine_consensus_sequence_b(self):
        self.assertEqual(self.b.determine_consensus_sequence(), 'ATCGAT')

    def test_determine_consensus_sequence_c(self):
        with self.assertRaises(ValueError):
            # Should fail when determining consensus sequence given an indel
            self.c.determine_consensus_sequence()
        self.assertIn(self.c.determine_consensus_sequence([0]), 'ATCGAA')

    def test_seqs_with_gap(self):
        self.assertCountEqual(self.a.seqs_with_gap(), [])
        self.assertCountEqual(self.b.seqs_with_gap(), [])
        self.assertCountEqual(self.c.seqs_with_gap(), [1])

    def test_construct_guide_a(self):
        self.assertEqual(self.a.construct_guide(0, 4, [0,1,2,3,4], 0),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a.construct_guide(0, 4, [0,1,2,3,4], 1),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_guide(0, 4, [4], 1),
                         ('AGCG', [4]))
        self.assertIn(self.a.construct_guide(0, 4, [2,3], 0),
                      [('ATCG', [2,3]), ('ACCG', [2,3])])
        self.assertEqual(self.a.construct_guide(1, 4, [0,1,2,3,4], 0),
                         ('TCGA', [0,1,2,3]))
        self.assertEqual(self.a.construct_guide(2, 4, [0,1,2,3,4], 0),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a.construct_guide(2, 4, [0,1,2,3,4], 1),
                         ('CGAA', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_guide(2, 4, [0,1,2,3,4], 2),
                         ('CGAA', [0,1,2,3,4]))
        self.assertIn(self.a.construct_guide(2, 4, [0,1,2,3], 0),
                      [('CGAA', [0,2]), ('CGAT', [1,3])])
        self.assertIn(self.a.construct_guide(2, 4, [0,1,2,3], 1),
                      [('CGAA', [0,1,2,3]), ('CGAT', [0,1,2,3])])

    def test_construct_guide_b(self):
        self.assertEqual(self.b.construct_guide(0, 4, [0,1,2,3,4], 0),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_guide(0, 4, [0,1,2,3,4], 1),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_guide(0, 4, [0,1,2,3,4], 2),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_guide(2, 4, [0,1,2,3,4], 0),
                         ('CGAA', [0]))
        self.assertEqual(self.b.construct_guide(2, 4, [0,1,2,3,4], 1),
                         ('CGAT', [0]))
        self.assertEqual(self.b.construct_guide(2, 4, [0,1,2,3,4], 2),
                         ('CGAT', [0,1,2,3]))
        self.assertEqual(self.b.construct_guide(2, 4, [0,1,2,3,4], 3),
                         ('CGAT', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_guide(2, 4, [2,4], 1),
                         ('CGAC', []))
        self.assertEqual(self.b.construct_guide(2, 4, [2,4], 2),
                         ('CGAC', [2,4]))
        self.assertIn(self.b.construct_guide(2, 4, [2,3,4], 2),
                      [('CGAC', [2,4]), ('CGAT', [2,3])])
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when 'N' is all that exists at a position
            self.b.construct_guide(0, 4, [1,3,4], 0)

    def test_construct_guide_c(self):
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when the only sequence given (1) has an indel
            self.c.construct_guide(0, 4, [1], 0)

    def test_sequences_bound_by_guide(self):
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 0),
                         [0,1,2,3])
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 1),
                         [0,1,2,3,4])


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
        self.aq = alignment.AlignmentQuerier(alns, 5, 1, k=3)
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
                self.assertTrue(self.aq.guide_is_specific_to_aln(guide, i, thres))
            for i in not_specific_to:
                self.assertFalse(self.aq.guide_is_specific_to_aln(guide, i, thres))

        assert_is_specific('ATCGA', 0, [])
        assert_is_specific('GGGGG', 0, [0, 1, 2])
        assert_is_specific('CCTTC', 0, [1])
        assert_is_specific('TTACA', 0, [])

        assert_is_specific('ATCGA', 0.5, [])
        assert_is_specific('GGGGG', 0.5, [0, 1, 2])
        assert_is_specific('CCTTC', 0.5, [1])
        assert_is_specific('TTACA', 0.5, [1])

