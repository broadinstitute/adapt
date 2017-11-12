"""Tests for guide_search module.
"""

import unittest

from dxguidedesign import alignment
from dxguidedesign import guide_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestGuideSearch(unittest.TestCase):
    """Tests methods in the GuideSearch class.
    """

    def setUp(self):
        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        self.a = guide_search.GuideSearcher(self.a_aln, 4, 0, 5, 1.0)

        self.b_seqs = ['ATCGAA', 'ATC-AA']
        self.b_aln = alignment.Alignment.from_list_of_seqs(self.b_seqs)
        self.b = guide_search.GuideSearcher(self.b_aln, 4, 0, 5, 1.0)

        self.c_seqs = ['GTATCATCGGCCATGNAC',
                       'GTNNCATCGG-CATGNAC',
                       'GTNNGATAGGCCATGNAC',
                       'GAAAAAAAAAAAAAAAAC',
                       'GTATCATCGGCCATGNAC',
                       'GTATCAGCGGCCATGNAC']
        self.c_aln = alignment.Alignment.from_list_of_seqs(self.c_seqs)
        self.c = guide_search.GuideSearcher(self.c_aln, 5, 1, 14, 1.0)
        self.c_partial = guide_search.GuideSearcher(self.c_aln, 5, 1, 14, 0.5)

        self.d_seqs = ['GTATACGG',
                       'ACGTACGG',
                       'TACTACGG']
        self.d_aln = alignment.Alignment.from_list_of_seqs(self.d_seqs)
        self.d = guide_search.GuideSearcher(self.d_aln, 5, 0, 8, 1.0)

        self.e_seqs = ['GTAGACGG',
                       'ACGTACGG',
                       'TACTTCGG']
        self.e_aln = alignment.Alignment.from_list_of_seqs(self.e_seqs)
        self.e = guide_search.GuideSearcher(self.e_aln, 5, 1, 8, 1.0)

        self.f_seqs = ['GTNNACGN',
                       'ANNTACGN',
                       'TANTTNNN']
        self.f_aln = alignment.Alignment.from_list_of_seqs(self.f_seqs)
        self.f = guide_search.GuideSearcher(self.f_aln, 5, 1, 8, 1.0)

        self.g_seqs = ['GTATCATCGGCCATCNAC',
                       'CTATCACCTGCTACGNAC',
                       'ATAGCACCGGCCATGNAC',
                       'TTAGGACCGACCATGNAC']
        self.g_aln = alignment.Alignment.from_list_of_seqs(self.g_seqs)
        self.g = guide_search.GuideSearcher(self.g_aln, 5, 0, 18, 1.0)
        self.g_partial = guide_search.GuideSearcher(self.g_aln, 5, 0, 18, 0.5)

    def test_construct_guide_memoized_a(self):
        self.assertEqual(self.a._construct_guide_memoized(0, [0,1,2,3,4]),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, [0,1,2,3,4]),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a._construct_guide_memoized(0, set([0,1,2,3,4])),
                         ('ATCG', [0,1,2,3]))
        self.assertIn(self.a._construct_guide_memoized(0, [2,3,4]),
                      [('ATCG', [2,3]), ('ACCG', [2,3]), ('AGCG', [4])])
        self.assertEqual(self.a._construct_guide_memoized(0, [4]),
                         ('AGCG', [4]))
        self.assertIn(0, self.a._memoized_guides)
        self.assertIn(frozenset([0,1,2,3,4]), self.a._memoized_guides[0])
        self.assertIn(frozenset([2,3,4]), self.a._memoized_guides[0])
        self.assertIn(frozenset([4]), self.a._memoized_guides[0])
        self.assertEqual(self.a._construct_guide_memoized(2, [0,1,2,3,4]),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a._construct_guide_memoized(2, [3]),
                         ('CGAT', [3]))
        self.assertIn(2, self.a._memoized_guides)
        self.assertIn(frozenset([0,1,2,3,4]), self.a._memoized_guides[2])
        self.assertIn(frozenset([3]), self.a._memoized_guides[2])

        self.assertEqual(self.a._construct_guide_memoized(0, [0,1,2,3,4]),
                         ('ATCG', [0,1,2,3]))
        self.assertIn(self.a._construct_guide_memoized(0, [2,3,4]),
                      [('ATCG', [2,3]), ('ACCG', [2,3]), ('AGCG', [4])])
        self.assertEqual(self.a._construct_guide_memoized(0, [4]),
                         ('AGCG', [4]))
        self.assertEqual(self.a._construct_guide_memoized(2, [0,1,2,3,4]),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a._construct_guide_memoized(2, [3]),
                         ('CGAT', [3]))

        self.a._cleanup_memoized_guides(2)
        self.assertNotIn(2, self.a._memoized_guides)
        self.a._cleanup_memoized_guides(100)
        self.assertNotIn(100, self.a._memoized_guides)

    def test_construct_guide_memoized_b(self):
        self.assertIsNone(self.b._construct_guide_memoized(0, [0]))
        self.assertIsNone(self.b._construct_guide_memoized(0, [0,1]))
                         
        self.assertIsNone(self.b._construct_guide_memoized(0, [0]))
        self.assertIsNone(self.b._construct_guide_memoized(0, [0,1]))

    def test_find_optimal_guide_in_window(self):
        self.assertEqual(self.c._find_optimal_guide_in_window(1,
                            set([0,1,2,3,4,5])),
                         ('ATCGG', set([0,1,2,4,5])))

    def test_find_optimal_guide_in_window_at_end_boundary(self):
        self.assertEqual(self.d._find_optimal_guide_in_window(0, set([0,1,2])),
                         ('TACGG', set([0,1,2]), 3))
        self.assertEqual(self.e._find_optimal_guide_in_window(0, set([0,1,2])),
                         ('TACGG', set([0,1,2]), 3))

    def test_find_optimal_guide_in_window(self):
        self.assertEqual(self.f._find_optimal_guide_in_window(0, set([0,1,2])),
                         (None, set(), None))

    def test_find_guides_that_cover_in_window(self):
        self.assertEqual(self.c._find_guides_that_cover_in_window(1),
                         set(['ATCGG', 'AAAAA']))
        self.assertEqual(self.c_partial._find_guides_that_cover_in_window(1),
                         set(['ATCGG']))

        self.assertIn(self.g._find_guides_that_cover_in_window(0),
                      [set(['TATCA', 'CCATG']), set(['CGGCC', 'TTAGG', 'CTATC'])])
        self.assertIn(self.g_partial._find_guides_that_cover_in_window(0),
                      [set(['TATCA']), set(['CCATG']), set(['CGGCC'])])