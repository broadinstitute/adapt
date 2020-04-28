"""Tests for guide_search module.
"""

import logging
import random
import unittest

import numpy as np

from adapt import alignment
from adapt import guide_search
from adapt.utils import guide
from adapt.utils import index_compress

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestGuideSearcherMinimizeGuides(unittest.TestCase):
    """Tests methods in the GuideSearcherMinimizeGuides class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        self.a_window_size = 5
        self.a = guide_search.GuideSearcherMinimizeGuides(self.a_aln, 4, 0, 1.0, (1, 1, 100))

        self.b_seqs = ['ATCGAA', 'ATC-AA']
        self.b_aln = alignment.Alignment.from_list_of_seqs(self.b_seqs)
        self.b_window_size = 5
        self.b = guide_search.GuideSearcherMinimizeGuides(self.b_aln, 4, 0, 1.0, (1, 1, 100))

        self.c_seqs = ['GTATCATCGGCCATGNAC',
                       'GTNNCATCGG-CATGNAC',
                       'GTNNGATAGGCCATGNAC',
                       'GAAAAAAAAAAAAAAAAC',
                       'GTATCATCGGCCATGNAC',
                       'GTATCAGCGGCCATGNAC']
        self.c_aln = alignment.Alignment.from_list_of_seqs(self.c_seqs)
        self.c_window_size = 14
        self.c = guide_search.GuideSearcherMinimizeGuides(self.c_aln, 5, 1, 1.0, (1, 1, 100))
        self.c_partial = guide_search.GuideSearcherMinimizeGuides(self.c_aln, 5, 1, 0.6,
            (1, 1, 100))

        self.d_seqs = ['GTATACGG',
                       'ACGTACGG',
                       'TACTACGG']
        self.d_aln = alignment.Alignment.from_list_of_seqs(self.d_seqs)
        self.d_window_size = 8
        self.d = guide_search.GuideSearcherMinimizeGuides(self.d_aln, 5, 0, 1.0, (1, 1, 100))

        self.e_seqs = ['GTAGACGG',
                       'ACGTACGG',
                       'TACTTCGG']
        self.e_aln = alignment.Alignment.from_list_of_seqs(self.e_seqs)
        self.e_window_size = 8
        self.e = guide_search.GuideSearcherMinimizeGuides(self.e_aln, 5, 1, 1.0, (1, 1, 100))

        self.f_seqs = ['GTNNACGN',
                       'ANNTACGN',
                       'TANTTNNN']
        self.f_aln = alignment.Alignment.from_list_of_seqs(self.f_seqs)
        self.f_window_size = 8
        self.f = guide_search.GuideSearcherMinimizeGuides(self.f_aln, 5, 1, 1.0, (1, 1, 100))

        self.g_seqs = ['GTATCATCGGCCATCNAC',
                       'CTATCACCTGCTACGNAC',
                       'ATAGCACCGGCCATGNAC',
                       'TTAGGACCGACCATGNAC']
        self.g_aln = alignment.Alignment.from_list_of_seqs(self.g_seqs)
        self.g_window_size = 18
        self.g = guide_search.GuideSearcherMinimizeGuides(self.g_aln, 5, 0, 1.0, (1, 1, 100))
        self.g_partial = guide_search.GuideSearcherMinimizeGuides(self.g_aln, 5, 0, 0.5,
            (1, 1, 100))

        self.h_seqs = ['GTATCAGCGGCCATCNACAA',
                       'GTANCACCTGCTACGNACTT',
                       'GTATCAATGNCCATGNACCC',
                       'GTATCATCCACNATGNACGG']
        self.h_aln = alignment.Alignment.from_list_of_seqs(self.h_seqs)
        self.h_window_size = 18
        self.h = guide_search.GuideSearcherMinimizeGuides(self.h_aln, 5, 1, 1.0, (0.5, 0, 1))

        self.i_seqs = ['GTATCAGCGGCCATCAACAA',
                       'GT-TCACCTGCTACGAACTT',
                       'GT-TCAATGCCCATGAACCC',
                       'GTATCATCCACCATGAACGG']
        self.i_aln = alignment.Alignment.from_list_of_seqs(self.i_seqs)
        self.i_window_size = 5
        self.i = guide_search.GuideSearcherMinimizeGuides(self.i_aln, 5, 1, 1.0, (0.5, 0, 1))

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
        self.i.guide_clusterer = None

    def test_construct_guide_memoized_a(self):
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}))
        self.assertIn(self.a._construct_guide_memoized(0, {0: {2,3,4}}),
                      [('ATCG', {2,3}), ('ACCG', {2,3}), ('AGCG', {4})])
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {4}}),
                         ('AGCG', {4}))

        def ic(idx):
            return index_compress.compress_mostly_contiguous(idx)

        key = (frozenset({(0, frozenset(ic({0,1,2,3,4})))}), None)
        self.assertIn(key, self.a._memoized_guides)
        self.assertIn(0, self.a._memoized_guides[key])

        key = (frozenset({(0, frozenset(ic({2,3,4})))}), None)
        self.assertIn(key, self.a._memoized_guides)
        self.assertIn(0, self.a._memoized_guides[key])

        key = (frozenset({(0, frozenset(ic({4})))}), None)
        self.assertIn(key, self.a._memoized_guides)
        self.assertIn(0, self.a._memoized_guides[key])

        self.assertEqual(self.a._construct_guide_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,2,4}))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {3}}),
                         ('CGAT', {3}))

        key = (frozenset({(0, frozenset(ic({0,1,2,3,4})))}), None)
        self.assertIn(key, self.a._memoized_guides)
        self.assertIn(2, self.a._memoized_guides[key])

        key = (frozenset({(0, frozenset(ic({3})))}), None)
        self.assertIn(key, self.a._memoized_guides)
        self.assertIn(2, self.a._memoized_guides[key])

        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}))
        self.assertIn(self.a._construct_guide_memoized(0, {0: {2,3,4}}),
                      [('ATCG', {2,3}), ('ACCG', {2,3}), ('AGCG', {4})])
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {4}}),
                         ('AGCG', {4}))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,2,4}))
        self.assertEqual(self.a._construct_guide_memoized(2, {0: {3}}),
                         ('CGAT', {3}))

        self.a._cleanup_memoized_guides(2)
        for key in self.a._memoized_guides.keys():
            self.assertNotIn(2, self.a._memoized_guides[key])

        self.a._cleanup_memoized_guides(100)
        for key in self.a._memoized_guides.keys():
            self.assertNotIn(100, self.a._memoized_guides[key])

    def test_construct_guide_memoized_b(self):
        self.assertIsNone(self.b._construct_guide_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_guide_memoized(0, {0: {0,1}}),
                         ('ATCG', {0}))
        
        self.assertIsNone(self.b._construct_guide_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_guide_memoized(0, {0: {0,1}}),
                         ('ATCG', {0}))

    def test_construct_guide_memoized_a_with_needed(self):
        # Use the num_needed argument
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', {0,1,2,3}))

        self.a._cleanup_memoized_guides(0)
        for key in self.a._memoized_guides.keys():
            self.assertNotIn(0, self.a._memoized_guides[key])

        self.a._cleanup_memoized_guides(100)
        for key in self.a._memoized_guides.keys():
            self.assertNotIn(100, self.a._memoized_guides[key])

    def test_construct_guide_memoized_a_use_last(self):
        # Use the use_last argument
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}, use_last=False),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=False),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=True),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=False),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a._construct_guide_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=True),
                         ('ATCG', {0,1,2,3}))

        self.a._cleanup_memoized_guides(0)
        for key in self.a._memoized_guides.keys():
            self.assertNotIn(0, self.a._memoized_guides[key])

    def test_find_optimal_guide_in_window(self):
        self.assertEqual(self.c._find_optimal_guide_in_window(
                            1, 1 + self.c_window_size,
                            {0: set([0,1,2,3,4,5])}, {0: 6}),
                         ('ATCGG', set([0,1,2,4,5]), 5, 5))

    def test_find_optimal_guide_in_window_at_end_boundary(self):
        self.assertEqual(self.d._find_optimal_guide_in_window(
                            0, 0 + self.d_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))
        self.assertEqual(self.e._find_optimal_guide_in_window(
                            0, 0 + self.e_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))

    def test_find_optimal_guide_in_window_none(self):
        self.assertEqual(self.f._find_optimal_guide_in_window(
                            0, 0 + self.f_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         (None, set(), None, 0))

    def test_find_optimal_guide_in_window_with_groups_1(self):
        g_opt = self.g._find_optimal_guide_in_window(0, 0 + self.g_window_size,
            {2017: {0, 2}, 2018: {1, 3}}, {2017: 0, 2018: 1})
        gd, gd_covered, gd_start, gd_score = g_opt

        # We only need to cover 1 sequence from the 2018 group ({1, 3});
        # check that at least one of these is covered
        self.assertTrue(1 in gd_covered or 3 in gd_covered)

        # Since we only need to cover 1 sequence in total, the score
        # should only be 1
        self.assertEqual(gd_score, 1)

    def test_find_guides_in_window(self):
        self.assertEqual(self.c._find_guides_in_window(
                            1, 1 + self.c_window_size),
                         set(['ATCGG', 'AAAAA']))
        self.assertIn(self.c_partial._find_guides_in_window(
                        1, 1 + self.c_window_size),
                      {frozenset(['ATCGG']), frozenset(['TCATC'])})

        self.assertIn(self.g._find_guides_in_window(
                        0, 0 + self.g_window_size),
                      [set(['TATCA', 'CCATG']), set(['CGGCC', 'TTAGG', 'CTATC'])])
        self.assertIn(self.g_partial._find_guides_in_window(
                        0, 0 + self.g_window_size),
                      [set(['TATCA']), set(['CCATG']), set(['CGGCC'])])

    def test_find_guides_with_missing_data(self):
        # The best guides are in regions with missing data, but the
        # alignment and thresholds on missing data are setup to avoid
        # guides in these regions
        self.assertEqual(self.h._find_guides_in_window(
                            0, 0 + self.h_window_size),
                         set(['CAACG', 'CACCC']))

    def test_find_guides_with_gap(self):
        # It should not be able to find a guide in a window where the only
        # possible guides overlap sequences with a gap
        with self.assertRaises(guide_search.CannotAchieveDesiredCoverageError):
            self.i._find_guides_in_window(1, 1 + self.i_window_size)

        # It should be able to find a guide in a window without a gap
        self.i._find_guides_in_window(10, 10 + self.i_window_size)

    def test_guide_is_suitable_fn(self):
        seqs = ['GTATCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATAGCAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guide is 'AAAAA'
        self.assertEqual(gs._find_guides_in_window(0, 28),
                         set(['AAAAA']))

        # Do not allow guides with 'AAA' in them
        def f(guide):
            if 'AAA' in guide:
                return False
            else:
                return True
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            guide_is_suitable_fn=f)
        self.assertEqual(gs._find_guides_in_window(0, 28),
                         set(['CCCCC', 'GGGGG']))

    def test_with_groups(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTTTACGATGCTCTGGTTAGCCATCT',
                'ATCTTATCGTTGGACTCGTAAGGCACCT',
                'ATCAGATCGCTGAGCTTGTGAGACAGCT',
                'TAGATCTAATCCCAGTATGGTACTTATC',
                'TAGAACTAATGGCAGTTTGGTCCTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # 4 guides are needed (3 for the first 4 sequences, 1 for the last
        # 2 sequences)
        self.assertEqual(len(gs._find_guides_in_window(0, 28)), 4)

        # Divide into groups, wanting to cover more of group 2018; now
        # we only need 1 guide from group 2010 and 1 from group 2018, so just
        # 2 guides are needed
        seq_groups = {2010: {0, 1, 2, 3}, 2018: {4, 5}}
        cover_frac = {2010: 0.1, 2018: 1.0}
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, cover_frac, (1, 1, 100),
            seq_groups=seq_groups)
        self.assertEqual(len(gs._find_guides_in_window(0, 28)), 2)

    def test_score_collection_of_guides_without_groups(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTTTACGATGCTCTGGTTAGCCATCT',
                'ATCTTATCGTTGGACTCGTAAGGCACCT',
                'ATCAGATCGCTGAGCTTGTGAGACAGCT',
                'TAGATCTAATCCCAGTATGGTACTTATC',
                'TAGAACTAATGGCAGTTTGGTACTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The function _score_collection_of_guides() will need to know
        # positions of guides (in _selected_guide_positions), so insert
        # these
        gs._selected_guide_positions = {'TCGAT': {6}, 'GGTAC': {18}}

        guides = ['TCGAT', 'GGTAC']
        # TCGAT covers 1 sequence (1/6) and GGTAC covers 2 sequences (2/6),
        # so the average is 0.25
        self.assertEqual(gs._score_collection_of_guides(guides), 0.25)

    def test_score_collection_of_guides_with_groups(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTTTTCGATGCTCTGGTTAGCCATCT',
                'ATCTTATCGTTGGACTCGTAAGGCACCT',
                'ATCAGATCGCTGAGCTTGTGAGACAGCT',
                'TAGATCTAATCCCAGTATGGTACTTATC',
                'TAGAACTAATGGCAGTTTGGTTCTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        seq_groups = {2010: {0, 1, 2, 3}, 2018: {4, 5}}
        cover_frac = {2010: 0.1, 2018: 1.0}
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, cover_frac, (1, 1, 100),
            seq_groups=seq_groups)

        # The function _score_collection_of_guides() will need to know
        # positions of guides (in _selected_guide_positions), so insert
        # these
        gs._selected_guide_positions = {'TCGAT': {6}, 'GGTAC': {18}}

        guides = ['TCGAT', 'GGTAC']
        # 3 sequences are needed in total (1 from 2010 and 2 from 2018)
        # TCGAT covers 1 needed sequence from 2010 and 0 needed sequences
        # from 2018: so it covers 1/3 needed sequences
        # GGTAC covers 0 needed sequences from 2010 and 1 needed sequence
        # from 2018: so it covers 1/3 needed sequences
        # The average of these fractions (the score) is 1/3
        self.assertEqual(gs._score_collection_of_guides(guides), 1/3.0)

    def test_find_optimal_guide_with_gu_pairing(self):
        seqs = ['GTATTAACACTTCGGCTACCCCCTCTAC',
                'CTACCAACACACCTGCTAGGGGGCGTAC',
                'ATAGCAACACAACGTCCTCCCCCTGTAC',
                'TTAGGGGTGTGGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        # Two guides are needed for coverage
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            allow_gu_pairs=False)
        self.assertEqual(len(gs._find_guides_in_window(0, 28)), 2)

        # Only one guide is needed for coverage: 'AACAC'
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            allow_gu_pairs=True)
        self.assertEqual(gs._find_guides_in_window(0, 28),
                         set(['AACAC']))

    def test_with_required_guides_full_coverage(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTAATCGATGCTCTGGTTAGCCATCT',
                'ATCCAATCGCAGTACTCGTAAGGCACCT',
                'ATCAAATCGGTGAGCTTGTGAGACAGCT',
                'TAGAAATCGAACTAGTATGGTACTTATC',
                'TAGAAATCGTGGCAGTTTGGTTCTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        # First 5 listed guides are from the alignment and the
        # positions given are correct; last guide ('AAAAA') is made up
        # but should still be in the output
        required_guides = {'ATGCC': 9, 'ATGCT': 9, 'TCGAA': 6, 'ATCGT': 5,
                           'TTGTC': 23, 'AAAAA': 5}

        # Search with 0 mismatches and 100% coverage
        window_size = 11
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            required_guides=required_guides)

        # Search in window starting at position 3
        guides_in_cover = gs._find_guides_in_window(
            3, 3 + window_size)
        # required_guides account for all sequences except the 3rd and 4th,
        # which can be both covered by 'AATCG'; the position of 'TTGTC' is
        # outside the window, so it should not be in the output
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'ATGCT', 'TCGAA', 'ATCGT', 'AAAAA', 'AATCG'})

        # Search in window starting at position 4; results should be the
        # same as above, but now use memoized values
        guides_in_cover = gs._find_guides_in_window(
            4, 4 + window_size)
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'ATGCT', 'TCGAA', 'ATCGT', 'AAAAA', 'AATCG'})

    def test_with_required_guides_partial_coverage(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTAATCGATGCTCTGGTTAGCCATCT',
                'ATCCAATCGCAGTACTCGTAAGGCACCT',
                'ATCAAATCGGTGAGCTTGTGAGACAGCT',
                'TAGAAATCGAACTAGTATGGTACTTATC',
                'TAGAAATCGTGGCAGTTTGGTTCTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        # First 3 listed guides are from the alignment and the
        # positions given are correct; last guide ('AAAAA') is made up
        # but should still be in the output
        required_guides = {'ATGCC': 9, 'TCGAA': 6, 'TTGTC': 23,
                           'AAAAA': 5}

        # Search with 1 mismatch and 50% coverage
        window_size = 11
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 1, 0.5, (1, 1, 100),
            required_guides=required_guides)

        # Search in window starting at position 3
        guides_in_cover = gs._find_guides_in_window(
            3, 3 + window_size)
        # required_guides can account for 3 of the 6 sequences
        # the position of 'TTGTC' is outside the window, so it should not be
        # in the output
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'TCGAA', 'AAAAA'})

    def test_guide_overlaps_blacklisted_range(self):
        seqs = ['AAAAAAAAAAAAAAAAAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        blacklisted_ranges = {(3, 8), (14, 18)}

        gs = guide_search.GuideSearcherMinimizeGuides(aln, 3, 0, 1.0, (1, 1, 100),
            blacklisted_ranges=blacklisted_ranges)

        # For each position i, encode 1 if the guide (of length 3)
        # starting at i overlaps a blacklisted range, and 0 otherwise
        does_overlap = '01111111000011111100'
        for i in range(len(does_overlap)):
            if does_overlap[i] == '0':
                self.assertFalse(gs._guide_overlaps_blacklisted_range(i))
            elif does_overlap[i] == '1':
                self.assertTrue(gs._guide_overlaps_blacklisted_range(i))

    def test_optimal_guide_with_blacklisted_range(self):
        seqs = ['GTATCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATAGCAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guide is 'AAAAA'
        self.assertEqual(gs._find_guides_in_window(0, 28),
                         set(['AAAAA']))

        # Do not allow guides overlapping (5, 9)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            blacklisted_ranges={(5, 9)})
        self.assertEqual(gs._find_guides_in_window(0, 28),
                         set(['CCCCC', 'GGGGG']))

    def test_with_windows_of_varying_size(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        self.assertEqual(gs._find_guides_in_window(1, 8),
                         set(['CCAAA', 'GGAAA']))

        # The best guide in [1, 15) is 'AAAAA'
        self.assertEqual(gs._find_guides_in_window(1, 15),
                         set(['AAAAA']))

    def test_obj_value(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        guides = gs._find_guides_in_window(1, 8)
        self.assertEqual(gs.obj_value(guides), 2)

    def test_total_frac_bound_by_guides(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        guides = gs._find_guides_in_window(1, 8)

        # All sequences are bound by a guide
        self.assertEqual(gs.total_frac_bound_by_guides(guides), 1.0)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)


class TestGuideSearcherMaximizeActivity(unittest.TestCase):
    """Tests methods in the GuideSearcherMaximizeActivity class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

    def make_gs(self, seqs, soft_guide_constraint=1, hard_guide_constraint=3,
            penalty_strength=0.1, algorithm='random-greedy',
            guide_is_suitable_fn=None):
        # Predict guides matching target to have activity 1, and
        # starting with 'A' to have activity 2 (otherwise, 0)
        class PredictorTest:
            def __init__(self):
                self.context_nt = 1
            def compute_activity(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    target_without_context = target[self.context_nt:len(target)-self.context_nt]
                    if guide == target_without_context:
                        if guide[0] == 'A':
                            y += [2]
                        else:
                            y += [1]
                    else:
                        y += [0]
                return y
            def cleanup_memoized(self, pos):
                pass
        predictor = PredictorTest()

        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMaximizeActivity(aln, 4,
                soft_guide_constraint, hard_guide_constraint,
                penalty_strength, (1, 1, 100), algorithm=algorithm,
                predictor=predictor,
                guide_is_suitable_fn=guide_is_suitable_fn)
        return gs

    def test_obj_value_from_params(self):
        gs = self.make_gs(['ATCGATCG'])
        self.assertEqual(gs._obj_value_from_params(10, 2),
                10 - 0.1*1)

    def test_ground_set_with_activities_memoized(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')

        o = gs._ground_set_with_activities_memoized(4)
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        np.testing.assert_equal(o['AATT'], np.array([2, 0, 0, 2]))
        np.testing.assert_equal(o['GGGG'], np.array([0, 1, 0, 0]))
        np.testing.assert_equal(o['CCCC'], np.array([0, 0, 1, 0]))

        o = gs._ground_set_with_activities_memoized(2)
        self.assertEqual(o.keys(), {'CGAA', 'GGGG', 'CCCC'})
        np.testing.assert_equal(o['CGAA'], np.array([1, 0, 0, 1]))
        np.testing.assert_equal(o['GGGG'], np.array([0, 1, 0, 0]))
        np.testing.assert_equal(o['CCCC'], np.array([0, 0, 1, 0]))

        # Try position 4 again
        o = gs._ground_set_with_activities_memoized(4)
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        np.testing.assert_equal(o['AATT'], np.array([2, 0, 0, 2]))
        np.testing.assert_equal(o['GGGG'], np.array([0, 1, 0, 0]))
        np.testing.assert_equal(o['CCCC'], np.array([0, 0, 1, 0]))

        gs._cleanup_memoized_ground_sets(4)

        # Try position 4 again
        o = gs._ground_set_with_activities_memoized(4)
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        np.testing.assert_equal(o['AATT'], np.array([2, 0, 0, 2]))
        np.testing.assert_equal(o['GGGG'], np.array([0, 1, 0, 0]))
        np.testing.assert_equal(o['CCCC'], np.array([0, 0, 1, 0]))

    def test_ground_set_with_activities_memoized_with_insufficient_guides(self):
        seqs = ['ATCGAATTCG']*100 + ['GGGGGGGGGG'] + ['ATCGAATTCG']*100
        gs = self.make_gs(seqs, algorithm='random-greedy')
        o = gs._ground_set_with_activities_memoized(4)
        self.assertEqual(o.keys(), {'AATT'})
        np.testing.assert_equal(o['AATT'], np.array([2]*100 + [0] + [2]*100))

    def test_activities_after_adding_guide(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')
        o = gs._activities_after_adding_guide(
                [1, 1, 1, 2, 0],
                [1, 0, 2, 2, 1])
        np.testing.assert_equal(o, np.array([1, 1, 2, 2, 1]))

    def test_analyze_guides(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')
        o = gs._analyze_guides(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

    def test_analyze_guides_memoized(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')

        o = gs._analyze_guides(4, [3, 0, 1, 1])
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

        o = gs._analyze_guides(2, [10, 0, 1, 1])
        self.assertEqual(o, {'CGAA': 12/4.0, 'GGGG': 13/4.0, 'CCCC': 12/4.0})

        # Try position 4 again
        o = gs._analyze_guides(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

        gs._cleanup_memoized_guides(4)

        # Try position 4 after cleanup
        o = gs._analyze_guides(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

    def test_find_optimal_guide_in_window_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='greedy')

        o = gs._find_optimal_guide_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        o = gs._find_optimal_guide_in_window(2, 8, {'AATT'},
                np.array([0, 0, 0, 0]))
        self.assertNotEqual(o[0], 'AATT')

        o = gs._find_optimal_guide_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 3]))
        self.assertIn(o, [('CCCC', 2), ('CCCC', 3), ('CCCC', 4), ('CCCC', 5)])

        o = gs._find_optimal_guide_in_window(2, 8, {'ATCG'},
                np.array([2, 1, 0, 2]))
        self.assertIn(o, [('CCCC', 2), ('CCCC', 3), ('CCCC', 4), ('CCCC', 5)])

    def test_find_optimal_guide_in_window_greedy_with_high_penalty(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           penalty_strength=1,
                           algorithm='greedy')

        o = gs._find_optimal_guide_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        # Adding any additional guide will create a negative marginal
        # contribution
        with self.assertRaises(guide_search.CannotFindPositiveMarginalContributionError):
            gs._find_optimal_guide_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 1]))

        # With a higher soft constraint, another guide is ok
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           soft_guide_constraint=2,
                           penalty_strength=1,
                           algorithm='greedy')
        o = gs._find_optimal_guide_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 1]))
        self.assertEqual(o[0], 'CCCC')

    def test_find_optimal_guide_in_window_random_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=1,
                           algorithm='random-greedy')
        o = gs._find_optimal_guide_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_guide_in_window(2, 8, {'AATT'},
                np.array([0, 1, 1, 0]))
        self.assertIn(o[0], ['CGAA', 'GAAT'])

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_guide_in_window(2, 8, {'ATCG'},
                np.array([4, 0, 0, 2]))
        self.assertIn(o[0], ['GGGG', 'CCCC'])

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_guide_in_window(2, 8, {'GGGG'},
                np.array([0, 1, 1, 1]))
        self.assertIn(o[0], ['CGAA', 'GAAT', 'AATT'])
        o = gs._find_optimal_guide_in_window(2, 8, {'GGGG'},
                np.array([3, 3, 3, 3]))
        self.assertIsNone(o)

    def test_find_optimal_guide_in_window_random_greedy_with_high_penalty(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           penalty_strength=0.5,
                           algorithm='random-greedy')

        # Adding any additional guide will create a negative marginal
        # contribution
        o = gs._find_optimal_guide_in_window(2, 8, {'AAAA'},
                np.array([10, 0, 0, 0]))
        self.assertIsNone(o)

    def find_guides_in_window_different_hard_constraints(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=3,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(o, {'AATT', 'GGGG', 'CCCC'})

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertIn('AATT', o)

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=1,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def find_guides_in_window_high_penalty(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=3,
                           penalty_strength=0.5,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def test_find_guides_in_window_different_hard_constraints_greedy(self):
        self.find_guides_in_window_different_hard_constraints('greedy')

    def test_find_guides_in_window_different_hard_constraints_random_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(len(o), 2) # many possibilities for o

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=1,
                           algorithm='random-greedy')
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def test_find_guides_in_window_high_penalty_greedy(self):
        self.find_guides_in_window_high_penalty('greedy')

    def test_find_guides_in_window_high_penalty_random_greedy(self):
        self.find_guides_in_window_high_penalty('random-greedy')

    def find_guides_in_window_with_guide_is_suitable_fn(self, algo):
        def f(guide):
            if 'AA' in guide:
                return False
            else:
                return True
        
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=1,
                           algorithm=algo,
                           guide_is_suitable_fn=f)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'CGAA', 'GAAT', 'GGGG', 'CCCC'})

    def test_find_guides_in_window_with_guide_is_suitable_fn_greedy(self):
        self.find_guides_in_window_with_guide_is_suitable_fn('greedy')

    def test_find_guides_in_window_with_guide_is_suitable_fn_random_greedy(self):
        self.find_guides_in_window_with_guide_is_suitable_fn('random-greedy')

    def find_guides_in_window_with_gaps_and_missing_data(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'ATCGA-TTCG',
                           'GGGGGNNNGG',
                           'AACGAATTCG'],
                           hard_guide_constraint=1,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'ATCGA-TTCG',
                           'GGGGGNNNGG',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm=algo)
        o = gs._find_guides_in_window(2, 8)
        self.assertEqual(len(o), 2)
        if algo == 'greedy':
            self.assertIn('AATT', o)

    def test_find_guides_in_window_with_gaps_and_missing_data_greedy(self):
        self.find_guides_in_window_with_gaps_and_missing_data('greedy')

    def test_find_guides_in_window_with_gaps_and_missing_data_random_greedy(self):
        self.find_guides_in_window_with_gaps_and_missing_data('random-greedy')

    def test_find_guides_in_window_at_endpoint(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'AGGGGGGGGA',
                           'ACCCCCCCCA',
                           'AACGAATTCG'],
                           hard_guide_constraint=1)

        # No guides should start at position 0, since it does not
        # permit a context_nt of 1 for evaluating activity
        o = gs._find_guides_in_window(0, 5)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'TCGA', 'GGGG', 'CCCC', 'ACGA'})
        
        # No guides should end at the end of the alignment, since it
        # does not permit a context_nt of 1 for evaluating activity
        o = gs._find_guides_in_window(5, 10)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'ATTC', 'GGGG', 'CCCC'})

        # The ranges below should not work because the test predictor
        # uses a context_nt of 1, which these do not permit
        with self.assertRaises(guide_search.CannotFindAnyGuidesError):
            gs._find_guides_in_window(0, 4)
        with self.assertRaises(guide_search.CannotFindAnyGuidesError):
            gs._find_guides_in_window(6, 10)

    def test_guide_set_activities(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_guides_in_window(2, 8)

        activities = gs.guide_set_activities(2, 8, guides)
        np.testing.assert_equal(activities, np.array([2, 2, 0, 2]))

        activities_percentile = gs.guide_set_activities_percentile(2, 8,
                guides, [5, 50])
        self.assertEqual(activities_percentile, [0, 2])

    def test_guide_activities_expected_value(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_guides_in_window(2, 8)

        aatt = gs.guide_activities_expected_value(2, 8, 'AATT')
        self.assertEqual(aatt, 1.0)

        aggg = gs.guide_activities_expected_value(2, 8, 'AGGG')
        self.assertEqual(aggg, 0.5)

    def test_obj_value(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           soft_guide_constraint=1,
                           penalty_strength=0.1,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_guides_in_window(2, 8)

        expected_activity = (2 + 2 + 0 + 2) / 4.0
        true_obj_value = expected_activity - 0.1*1
        self.assertEqual(gs.obj_value(2, 8, guides), true_obj_value)

    def test_total_frac_bound_by_guides(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_guide_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_guides_in_window(2, 8)

        # 3 of the 4 sequences should be bound
        self.assertEqual(gs.total_frac_bound_by_guides(2, 8, guides), 0.75)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
