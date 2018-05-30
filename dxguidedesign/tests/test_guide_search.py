"""Tests for guide_search module.
"""

import random
import unittest

from dxguidedesign import alignment
from dxguidedesign import guide_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestGuideSearch(unittest.TestCase):
    """Tests methods in the GuideSearch class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        self.a = guide_search.GuideSearcher(self.a_aln, 4, 0, 5, 1.0, (1, 1, 100))

        self.b_seqs = ['ATCGAA', 'ATC-AA']
        self.b_aln = alignment.Alignment.from_list_of_seqs(self.b_seqs)
        self.b = guide_search.GuideSearcher(self.b_aln, 4, 0, 5, 1.0, (1, 1, 100))

        self.c_seqs = ['GTATCATCGGCCATGNAC',
                       'GTNNCATCGG-CATGNAC',
                       'GTNNGATAGGCCATGNAC',
                       'GAAAAAAAAAAAAAAAAC',
                       'GTATCATCGGCCATGNAC',
                       'GTATCAGCGGCCATGNAC']
        self.c_aln = alignment.Alignment.from_list_of_seqs(self.c_seqs)
        self.c = guide_search.GuideSearcher(self.c_aln, 5, 1, 14, 1.0, (1, 1, 100))
        self.c_partial = guide_search.GuideSearcher(self.c_aln, 5, 1, 14, 0.6,
            (1, 1, 100))

        self.d_seqs = ['GTATACGG',
                       'ACGTACGG',
                       'TACTACGG']
        self.d_aln = alignment.Alignment.from_list_of_seqs(self.d_seqs)
        self.d = guide_search.GuideSearcher(self.d_aln, 5, 0, 8, 1.0, (1, 1, 100))

        self.e_seqs = ['GTAGACGG',
                       'ACGTACGG',
                       'TACTTCGG']
        self.e_aln = alignment.Alignment.from_list_of_seqs(self.e_seqs)
        self.e = guide_search.GuideSearcher(self.e_aln, 5, 1, 8, 1.0, (1, 1, 100))

        self.f_seqs = ['GTNNACGN',
                       'ANNTACGN',
                       'TANTTNNN']
        self.f_aln = alignment.Alignment.from_list_of_seqs(self.f_seqs)
        self.f = guide_search.GuideSearcher(self.f_aln, 5, 1, 8, 1.0, (1, 1, 100))

        self.g_seqs = ['GTATCATCGGCCATCNAC',
                       'CTATCACCTGCTACGNAC',
                       'ATAGCACCGGCCATGNAC',
                       'TTAGGACCGACCATGNAC']
        self.g_aln = alignment.Alignment.from_list_of_seqs(self.g_seqs)
        self.g = guide_search.GuideSearcher(self.g_aln, 5, 0, 18, 1.0, (1, 1, 100))
        self.g_partial = guide_search.GuideSearcher(self.g_aln, 5, 0, 18, 0.5,
            (1, 1, 100))

        self.h_seqs = ['GTATCAGCGGCCATCNACAA',
                       'GTANCACCTGCTACGNACTT',
                       'GTATCAATGNCCATGNACCC',
                       'GTATCATCCACNATGNACGG']
        self.h_aln = alignment.Alignment.from_list_of_seqs(self.h_seqs)
        self.h = guide_search.GuideSearcher(self.h_aln, 5, 1, 18, 1.0, (0.5, 0, 1))

        # Skip guide clustering, which may not work well when the guides
        # are so short in these tests
        self.a.guide_clusterer = None
        self.b.guide_clusterer = None
        self.c.guide_clusterer = None
        self.d.guide_clusterer = None
        self.e.guide_clusterer = None
        self.f.guide_clusterer = None
        self.g.guide_clusterer = None
        self.h.guide_clusterer = None

    def test_construct_guide_memoized_a(self):
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', [0,1,2,3]))
        self.assertIn(self.a._construct_guide_memoized(0, {0: {2,3,4}}),
                      [('ATCG', [2,3]), ('ACCG', [2,3]), ('AGCG', [4])])
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {4}}),
                         ('AGCG', [4]))
        self.assertIn(0, self.a._memoized_guides)
        self.assertIn((frozenset({(0, frozenset({0,1,2,3,4}))}), None),
            self.a._memoized_guides[0])
        self.assertIn((frozenset({(0, frozenset({2,3,4}))}), None),
            self.a._memoized_guides[0])
        self.assertIn((frozenset({(0, frozenset({4}))}), None),
            self.a._memoized_guides[0])
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {3}}),
                         ('CGAT', [3]))
        self.assertIn(2, self.a._memoized_guides)
        self.assertIn((frozenset({(0, frozenset({0,1,2,3,4}))}), None),
            self.a._memoized_guides[2])
        self.assertIn((frozenset({(0, frozenset({3}))}), None),
            self.a._memoized_guides[2])

        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', [0,1,2,3]))
        self.assertIn(self.a._construct_guide_memoized(0, {0: {2,3,4}}),
                      [('ATCG', [2,3]), ('ACCG', [2,3]), ('AGCG', [4])])
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {4}}),
                         ('AGCG', [4]))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {3}}),
                         ('CGAT', [3]))

        self.a._cleanup_memoized_guides(2)
        self.assertNotIn(2, self.a._memoized_guides)
        self.a._cleanup_memoized_guides(100)
        self.assertNotIn(100, self.a._memoized_guides)

    def test_construct_guide_memoized_b(self):
        self.assertIsNone(self.b._construct_guide_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_guide_memoized(0, {0: {0,1}}),
                         ('ATCG', [0]))
        
        self.assertIsNone(self.b._construct_guide_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_guide_memoized(0, {0: {0,1}}),
                         ('ATCG', [0]))

    def test_construct_guide_memoized_a_with_needed(self):
        # Use the num_needed argument
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', [0,1,2,3]))

        self.a._cleanup_memoized_guides(0)
        self.assertNotIn(0, self.a._memoized_guides)
        self.a._cleanup_memoized_guides(100)
        self.assertNotIn(100, self.a._memoized_guides)

    def test_find_optimal_guide_in_window(self):
        self.assertEqual(self.c._find_optimal_guide_in_window(1,
                            {0: set([0,1,2,3,4,5])}, {0: 6}),
                         ('ATCGG', set([0,1,2,4,5]), 5, 5))

    def test_find_optimal_guide_in_window_at_end_boundary(self):
        self.assertEqual(self.d._find_optimal_guide_in_window(0,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))
        self.assertEqual(self.e._find_optimal_guide_in_window(0,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))

    def test_find_optimal_guide_in_window_none(self):
        self.assertEqual(self.f._find_optimal_guide_in_window(0,
                            {0: set([0,1,2])}, {0: 3}),
                         (None, set(), None, 0))

    def test_find_optimal_guide_in_window_with_groups_1(self):
        g_opt = self.g._find_optimal_guide_in_window(0,
            {2017: {0, 2}, 2018: {1, 3}}, {2017: 0, 2018: 1})
        gd, gd_covered, gd_start, gd_score = g_opt

        # We only need to cover 1 sequence from the 2018 group ({1, 3});
        # check that at least one of these is covered
        self.assertTrue(1 in gd_covered or 3 in gd_covered)

        # Since we only need to cover 1 sequence in total, the score
        # should only be 1
        self.assertEqual(gd_score, 1)

    def test_find_guides_that_cover_in_window(self):
        self.assertEqual(self.c._find_guides_that_cover_in_window(1),
                         set(['ATCGG', 'AAAAA']))
        self.assertIn(self.c_partial._find_guides_that_cover_in_window(1),
                      {frozenset(['ATCGG']), frozenset(['TCATC'])})

        self.assertIn(self.g._find_guides_that_cover_in_window(0),
                      [set(['TATCA', 'CCATG']), set(['CGGCC', 'TTAGG', 'CTATC'])])
        self.assertIn(self.g_partial._find_guides_that_cover_in_window(0),
                      [set(['TATCA']), set(['CCATG']), set(['CGGCC'])])

    def test_find_guides_with_missing_data(self):
        # The best guides are in regions with missing data, but the
        # alignment and thresholds on missing data are setup to avoid
        # guides in these regions
        self.assertEqual(self.h._find_guides_that_cover_in_window(0),
                         set(['CAACG', 'CACCC']))

    def test_guide_is_suitable_fn(self):
        seqs = ['GTATCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATAGCAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcher(aln, 5, 0, 28, 1.0, (1, 1, 100))

        # The best guide is 'AAAAA'
        self.assertEqual(gs._find_guides_that_cover_in_window(0),
                         set(['AAAAA']))

        # Do not allow guides with 'AAA' in them
        def f(guide):
            if 'AAA' in guide:
                return False
            else:
                return True
        gs = guide_search.GuideSearcher(aln, 5, 0, 28, 1.0, (1, 1, 100),
            guide_is_suitable_fn=f)
        self.assertEqual(gs._find_guides_that_cover_in_window(0),
                         set(['CCCCC', 'GGGGG']))

    def test_with_groups(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTTTACGATGCTCTGGTTAGCCATCT',
                'ATCTTATCGTTGGACTCGTAAGGCACCT',
                'ATCAGATCGCTGAGCTTGTGAGACAGCT',
                'TAGATCTAATCCCAGTATGGTACTTATC',
                'TAGAACTAATGGCAGTTTGGTCCTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcher(aln, 5, 0, 28, 1.0, (1, 1, 100))

        # 4 guides are needed (3 for the first 4 sequences, 1 for the last
        # 2 sequences)
        self.assertEqual(len(gs._find_guides_that_cover_in_window(0)), 4)

        # Divide into groups, wanting to cover more of group 2018; now
        # we only need 1 guide from group 2010 and 1 from group 2018, so just
        # 2 guides are needed
        seq_groups = {2010: {0, 1, 2, 3}, 2018: {4, 5}}
        cover_frac = {2010: 0.1, 2018: 1.0}
        gs = guide_search.GuideSearcher(aln, 5, 0, 28, cover_frac, (1, 1, 100),
            seq_groups=seq_groups)
        self.assertEqual(len(gs._find_guides_that_cover_in_window(0)), 2)
