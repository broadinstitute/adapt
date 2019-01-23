"""Tests for alignment module.
"""

import random
import unittest

from dxguidedesign import alignment
from dxguidedesign.utils import guide
from dxguidedesign.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignment(unittest.TestCase):
    """Tests methods in the Alignment class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a = alignment.Alignment.from_list_of_seqs(self.a_seqs)

        self.b_seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.b = alignment.Alignment.from_list_of_seqs(self.b_seqs)

        self.c_seqs = ['ATCGAA', 'ATC-AA']
        self.c = alignment.Alignment.from_list_of_seqs(self.c_seqs)

        self.gc = alignment.SequenceClusterer(lsh.HammingDistanceFamily(4), k=2)

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

    def test_make_list_of_seqs_c_with_no_gaps(self):
        self.assertEqual(self.c.make_list_of_seqs(remove_gaps=True),
                         ['ATCGAA', 'ATCAA'])

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
        self.assertEqual(self.a.construct_guide(0, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a.construct_guide(0, 4, {0: {0,1,2,3,4}}, 1, False, self.gc),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_guide(0, 4, {0: {4}}, 1, False, self.gc),
                         ('AGCG', [4]))
        self.assertIn(self.a.construct_guide(0, 4, {0: {2,3}}, 0, False, self.gc),
                      [('ATCG', [2,3]), ('ACCG', [2,3])])
        self.assertEqual(self.a.construct_guide(1, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('TCGA', [0,1,2,3]))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 1, False, self.gc),
                         ('CGAA', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 2, False, self.gc),
                         ('CGAA', [0,1,2,3,4]))
        self.assertIn(self.a.construct_guide(2, 4, {0: {0,1,2,3}}, 0, False, self.gc),
                      [('CGAA', [0,2]), ('CGAT', [1,3])])
        self.assertIn(self.a.construct_guide(2, 4, {0: {0,1,2,3}}, 1, False, self.gc),
                      [('CGAA', [0,1,2,3]), ('CGAT', [0,1,2,3])])

    def test_construct_guide_b(self):
        # self.b has many Ns, which makes it difficult to write test cases
        # when clustering (the clusters tend to consist of guides in
        # which a position only has N); so pass None to guide_clusterer in
        # construct_guide() to skip clustering
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 0, False, None),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 1, False, None),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 2, False, None),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 0, False, None),
                         ('CGAA', [0]))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 1, False, None),
                         ('CGAT', [0]))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 2, False, None),
                         ('CGAT', [0,1,2,3]))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 3, False, None),
                         ('CGAT', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {2,4}}, 1, False, None),
                         ('CGAC', []))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {2,4}}, 2, False, None),
                         ('CGAC', [2,4]))
        self.assertIn(self.b.construct_guide(2, 4, {0: {2,3,4}}, 2, False, None),
                      [('CGAC', [2,4]), ('CGAT', [2,3])])
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when 'N' is all that exists at a position
            self.b.construct_guide(0, 4, {0: {1,3,4}}, 0, False, None)

    def test_construct_guide_c(self):
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when the only sequence given (1) has an indel
            self.c.construct_guide(0, 4, {0: {1}}, 0, False, self.gc)

    def test_construct_guide_with_large_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        seqs_aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 3, 1: 1}
        # 'ATCGAA' is most sequences, and let's construct a guide by
        # needing more from the group consisting of these sequences
        self.assertEqual(seqs_aln.construct_guide(0, 4, seqs_to_consider, 0,
                            False, self.gc, num_needed=num_needed),
                         ('ATCG', [0, 1, 3, 4, 5]))

    def test_construct_guide_with_small_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        seqs_aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 1, 1: 2}
        # 'ATCGAA' is most sequences, but let's construct a guide by
        # needing more from a group consisting of the 'GGGCCC' sequences
        self.assertEqual(seqs_aln.construct_guide(0, 4, seqs_to_consider, 0,
                            False, self.gc, num_needed=num_needed),
                         ('GGGC', [2, 6]))

    def test_construct_guide_with_suitable_fn(self):
        seqs = ['GTATCAAAT',
                'CTACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        seqs_to_consider = {0: {0, 1, 2, 3}}
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)

        # The best guide is 'GTATCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer)
        gd, covered_seqs = p
        self.assertEqual(gd, 'GTATCA')
        self.assertEqual(covered_seqs, [0, 2, 3])

        # Do not allow guides with 'TAT' in them
        def f(guide):
            if 'TAT' in guide:
                return False
            else:
                return True
        # Now the best guide is 'CTACCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
            guide_is_suitable_fn=f)
        gd, covered_seqs = p
        self.assertEqual(gd, 'CTACCA')
        self.assertEqual(covered_seqs, [1])

        # Do not allow guides with 'A' in them
        def f(guide):
            if 'A' in guide:
                return False
            else:
                return True
        # Now there is no suitable guide
        with self.assertRaises(alignment.CannotConstructGuideError):
            aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
                guide_is_suitable_fn=f)

    def test_sequences_bound_by_guide(self):
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 0, False),
                         [0,1,2,3])
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 1, False),
                         [0,1,2,3,4])

    def test_construct_from_0_seqs(self):
        with self.assertRaises(Exception):
            seqs = []
            alignment.Alignment.from_list_of_seqs(seqs)


class TestSequenceClusterer(unittest.TestCase):
    """Tests the SequenceClusterer class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        family = lsh.HammingDistanceFamily(10)
        self.sc = alignment.SequenceClusterer(family, k=3)

    def test_cluster(self):
        seqs = [('ATCGAAATAA', 4),
                ('ATCGAAATAA', 5),
                ('ATTGAAATAT', 1),
                ('CTGGTCATAA', 2),
                ('CTCGTCATAA', 3)]

        clusters = self.sc.cluster(seqs)
        self.assertCountEqual(clusters,
            {frozenset({1, 4, 5}), frozenset({2, 3})})

    def test_largest_cluster(self):
        seqs = [('ATCGAAATAA', 4),
                ('ATCGAAATAA', 5),
                ('ATTGAAATAT', 1),
                ('CTGGTCATAA', 2),
                ('CTCGTCATAA', 3)]

        largest_cluster = self.sc.largest_cluster(seqs)
        self.assertCountEqual(largest_cluster, {1, 4, 5})


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
        self.aq = alignment.AlignmentQuerier(alns, 5, 1, False, k=3,
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

    def test_hit_with_gaps(self):
        gappy_aln_seqs = ['ATCGACGTAAACC',
                          'ATCGA--TAATGG',
                          'ATGGA--TAATGG']
        gappy_aln = alignment.Alignment.from_list_of_seqs(gappy_aln_seqs)
        gappy_aq = alignment.AlignmentQuerier([gappy_aln], 5, 1, False, k=3,
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
