"""Tests for guide_search module.
"""

import logging
import random
import unittest

import numpy as np

from adapt import alignment
from adapt import guide_search
from adapt.utils import search
from adapt.utils import index_compress
from adapt.utils import lsh

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
        self.a = guide_search.GuideSearcherMinimizeGuides(self.a_aln, 4, 0, 1.0, (1, 1, 100))


        self.b_seqs = ['ATCGAA', 'ATC-AA']
        self.b_aln = alignment.Alignment.from_list_of_seqs(self.b_seqs)
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

        # Create a generic guide clusterer that can have its alignment overriden
        # (for testing construct_oligo)
        gen_seqs = ['AAAA']
        gen_aln = alignment.Alignment.from_list_of_seqs(gen_seqs)
        self.gen_gs = guide_search.GuideSearcherMinimizeGuides(gen_aln, 4, 0, 1.0, (1, 1, 100))

        # Skip guide clustering, which may not work well when the guides
        # are so short in these tests
        self.a.clusterer = None
        self.b.clusterer = None
        self.c.clusterer = None
        self.d.clusterer = None
        self.e.clusterer = None
        self.f.clusterer = None
        self.g.clusterer = None
        self.h.clusterer = None
        self.i.clusterer = None
        self.gen_gs.clusterer = None

        # Some tests do require a guide clusterer, so store here
        self.gc = alignment.SequenceClusterer(lsh.HammingDistanceFamily(4), k=2)

    def test_construct_memoized_a(self):
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertIn(self.a._construct_memoized(0, {0: {2,3,4}}),
                      [('ATCG', {2,3}, 2), ('ACCG', {2,3}, 2), ('AGCG', {4}, 1)])
        self.assertEqual(self.a._construct_memoized(0, {0: {4}}),
                         ('AGCG', {4}, 1))

        def ic(idx):
            return index_compress.compress_mostly_contiguous(idx)

        key = (frozenset({(0, frozenset(ic({0,1,2,3,4})))}), None)
        self.assertIn(key, self.a._memo)
        self.assertIn(0, self.a._memo[key])

        key = (frozenset({(0, frozenset(ic({2,3,4})))}), None)
        self.assertIn(key, self.a._memo)
        self.assertIn(0, self.a._memo[key])

        key = (frozenset({(0, frozenset(ic({4})))}), None)
        self.assertIn(key, self.a._memo)
        self.assertIn(0, self.a._memo[key])

        self.assertEqual(self.a._construct_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,2,4}, 3))
        self.assertEqual(self.a._construct_memoized(2, {0: {3}}),
                         ('CGAT', {3}, 1))

        key = (frozenset({(0, frozenset(ic({0,1,2,3,4})))}), None)
        self.assertIn(key, self.a._memo)
        self.assertIn(2, self.a._memo[key])

        key = (frozenset({(0, frozenset(ic({3})))}), None)
        self.assertIn(key, self.a._memo)
        self.assertIn(2, self.a._memo[key])

        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertIn(self.a._construct_memoized(0, {0: {2,3,4}}),
                      [('ATCG', {2,3}, 2), ('ACCG', {2,3}, 2), ('AGCG', {4}, 1)])
        self.assertEqual(self.a._construct_memoized(0, {0: {4}}),
                         ('AGCG', {4}, 1))
        self.assertEqual(self.a._construct_memoized(2, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,2,4}, 3))
        self.assertEqual(self.a._construct_memoized(2, {0: {3}}),
                         ('CGAT', {3}, 1))

        self.a._cleanup_memo(2)
        for key in self.a._memo.keys():
            self.assertNotIn(2, self.a._memo[key])

        self.a._cleanup_memo(100)
        for key in self.a._memo.keys():
            self.assertNotIn(100, self.a._memo[key])

    def test_construct_memoized_b(self):
        self.assertIsNone(self.b._construct_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_memoized(0, {0: {0,1}}),
                         ('ATCG', {0}, 1))

        self.assertIsNone(self.b._construct_memoized(0, {0: {1}}))
        self.assertEqual(self.b._construct_memoized(0, {0: {0,1}}),
                         ('ATCG', {0}, 1))

    def test_construct_memoized_a_with_needed(self):
        # Use the num_needed argument
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', {0,1,2,3}, 3))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}),
                         ('ATCG', {0,1,2,3}, 3))

        self.a._cleanup_memo(0)
        for key in self.a._memo.keys():
            self.assertNotIn(0, self.a._memo[key])

        self.a._cleanup_memo(100)
        for key in self.a._memo.keys():
            self.assertNotIn(100, self.a._memo[key])

    def test_construct_memoized_a_use_last(self):
        # Use the use_last argument
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 5}, use_last=False),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=False),
                         ('ATCG', {0,1,2,3}, 3))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=True),
                         ('ATCG', {0,1,2,3}, 3))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=False),
                         ('ATCG', {0,1,2,3}, 3))
        self.assertEqual(self.a._construct_memoized(0, {0: {0,1,2,3,4}},
                            {0: 3}, use_last=True),
                         ('ATCG', {0,1,2,3}, 3))

        self.a._cleanup_memo(0)
        for key in self.a._memo.keys():
            self.assertNotIn(0, self.a._memo[key])

    def test_find_optimal_oligo_in_window(self):
        self.assertEqual(self.c._find_optimal_oligo_in_window(
                            1, 1 + self.c_window_size,
                            {0: set([0,1,2,3,4,5])}, {0: 6}),
                         ('ATCGG', set([0,1,2,4,5]), 5, 5))

    def test_find_optimal_oligo_in_window_at_end_boundary(self):
        self.assertEqual(self.d._find_optimal_oligo_in_window(
                            0, 0 + self.d_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))
        self.assertEqual(self.e._find_optimal_oligo_in_window(
                            0, 0 + self.e_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         ('TACGG', set([0,1,2]), 3, 3))

    def test_find_optimal_oligo_in_window_none(self):
        self.assertEqual(self.f._find_optimal_oligo_in_window(
                            0, 0 + self.f_window_size,
                            {0: set([0,1,2])}, {0: 3}),
                         (None, set(), None, 0))

    def test_find_optimal_oligo_in_window_with_groups_1(self):
        g_opt = self.g._find_optimal_oligo_in_window(0, 0 + self.g_window_size,
            {2017: {0, 2}, 2018: {1, 3}}, {2017: 0, 2018: 1})
        gd, gd_covered, gd_start, gd_score = g_opt

        # We only need to cover 1 sequence from the 2018 group ({1, 3});
        # check that at least one of these is covered
        self.assertTrue(1 in gd_covered or 3 in gd_covered)

        # Since we only need to cover 1 sequence in total, the score
        # should only be 1
        self.assertEqual(gd_score, 1)

    def test_find_oligos_in_window(self):
        # self.assertEqual(self.c._find_oligos_in_window(
        #                     1, 1 + self.c_window_size),
        #                  set(['ATCGG', 'AAAAA']))
        # self.assertIn(self.c_partial._find_oligos_in_window(
        #                 1, 1 + self.c_window_size),
        #               {frozenset(['ATCGG']), frozenset(['TCATC'])})

        self.assertIn(self.g._find_oligos_in_window(
                        0, 0 + self.g_window_size),
                      [set(['TATCA', 'CCATG']), set(['CGGCC', 'TTAGG', 'CTATC'])])
        self.assertIn(self.g_partial._find_oligos_in_window(
                        0, 0 + self.g_window_size),
                      [set(['TATCA']), set(['CCATG']), set(['CGGCC'])])

    def test_find_oligos_with_missing_data(self):
        # The best guides are in regions with missing data, but the
        # alignment and thresholds on missing data are setup to avoid
        # guides in these regions
        self.assertEqual(self.h._find_oligos_in_window(
                            0, 0 + self.h_window_size),
                         set(['CAACG', 'CACCC']))

    def test_find_oligos_with_gap(self):
        # It should not be able to find a guide in a window where the only
        # possible guides overlap sequences with a gap
        with self.assertRaises(search.CannotAchieveDesiredCoverageError):
            self.i._find_oligos_in_window(1, 1 + self.i_window_size)

        # It should be able to find a guide in a window without a gap
        self.i._find_oligos_in_window(10, 10 + self.i_window_size)

    def test_is_suitable_fns(self):
        seqs = ['GTATCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATAGCAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guide is 'AAAAA'
        self.assertEqual(gs._find_oligos_in_window(0, 28),
                         set(['AAAAA']))

        # Do not allow guides with 'AAA' in them
        def f(guide):
            if 'AAA' in guide:
                return False
            else:
                return True
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            is_suitable_fns=[f])
        self.assertEqual(gs._find_oligos_in_window(0, 28),
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
        self.assertEqual(len(gs._find_oligos_in_window(0, 28)), 4)

        # Divide into groups, wanting to cover more of group 2018; now
        # we only need 1 guide from group 2010 and 1 from group 2018, so just
        # 2 guides are needed
        seq_groups = {2010: {0, 1, 2, 3}, 2018: {4, 5}}
        cover_frac = {2010: 0.1, 2018: 1.0}
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, cover_frac, (1, 1, 100),
            seq_groups=seq_groups)
        self.assertEqual(len(gs._find_oligos_in_window(0, 28)), 2)

    def test_score_collection_without_groups(self):
        seqs = ['ATCAAATCGATGCCCTAGTCAGTCAACT',
                'ATCTTTACGATGCTCTGGTTAGCCATCT',
                'ATCTTATCGTTGGACTCGTAAGGCACCT',
                'ATCAGATCGCTGAGCTTGTGAGACAGCT',
                'TAGATCTAATCCCAGTATGGTACTTATC',
                'TAGAACTAATGGCAGTTTGGTACTTGTC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The function _score_collection() will need to know
        # positions of guides (in _selected_guide_positions), so insert
        # these
        gs._selected_positions = {'TCGAT': {6}, 'GGTAC': {18}}

        guides = ['TCGAT', 'GGTAC']
        # TCGAT covers 1 sequence (1/6) and GGTAC covers 2 sequences (2/6),
        # so the average is 0.25
        self.assertEqual(gs._score_collection(guides), 0.25)

    def test_score_collection_with_groups(self):
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

        # The function _score_collection() will need to know
        # positions of guides (in _selected_guide_positions), so insert
        # these
        gs._selected_positions = {'TCGAT': {6}, 'GGTAC': {18}}

        guides = ['TCGAT', 'GGTAC']
        # 3 sequences are needed in total (1 from 2010 and 2 from 2018)
        # TCGAT covers 1 needed sequence from 2010 and 0 needed sequences
        # from 2018: so it covers 1/3 needed sequences
        # GGTAC covers 0 needed sequences from 2010 and 1 needed sequence
        # from 2018: so it covers 1/3 needed sequences
        # The average of these fractions (the score) is 1/3
        self.assertEqual(gs._score_collection(guides), 1/3.0)

    def test_find_optimal_oligo_with_gu_pairing(self):
        seqs = ['GTATTAACACTTCGGCTACCCCCTCTAC',
                'CTACCAACACACCTGCTAGGGGGCGTAC',
                'ATAGCAACACAACGTCCTCCCCCTGTAC',
                'TTAGGGGTGTGGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        # Two guides are needed for coverage
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            allow_gu_pairs=False)
        self.assertEqual(len(gs._find_oligos_in_window(0, 28)), 2)

        # Only one guide is needed for coverage: 'AACAC'
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            allow_gu_pairs=True)
        self.assertEqual(gs._find_oligos_in_window(0, 28),
                         set(['AACAC']))

    def test_with_required_oligos_full_coverage(self):
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
        required_oligos = {'ATGCC': 9, 'ATGCT': 9, 'TCGAA': 6, 'ATCGT': 5,
                           'TTGTC': 23, 'AAAAA': 5}

        # Search with 0 mismatches and 100% coverage
        window_size = 11
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            required_oligos=required_oligos)

        # Search in window starting at position 3
        guides_in_cover = gs._find_oligos_in_window(
            3, 3 + window_size)
        # required_oligos account for all sequences except the 3rd and 4th,
        # which can be both covered by 'AATCG'; the position of 'TTGTC' is
        # outside the window, so it should not be in the output
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'ATGCT', 'TCGAA', 'ATCGT', 'AAAAA', 'AATCG'})

        # Search in window starting at position 4; results should be the
        # same as above, but now use memoized values
        guides_in_cover = gs._find_oligos_in_window(
            4, 4 + window_size)
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'ATGCT', 'TCGAA', 'ATCGT', 'AAAAA', 'AATCG'})

    def test_with_required_oligos_partial_coverage(self):
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
        required_oligos = {'ATGCC': 9, 'TCGAA': 6, 'TTGTC': 23,
                           'AAAAA': 5}

        # Search with 1 mismatch and 50% coverage
        window_size = 11
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 1, 0.5, (1, 1, 100),
            required_oligos=required_oligos)

        # Search in window starting at position 3
        guides_in_cover = gs._find_oligos_in_window(
            3, 3 + window_size)
        # required_oligos can account for 3 of the 6 sequences
        # the position of 'TTGTC' is outside the window, so it should not be
        # in the output
        self.assertEqual(guides_in_cover,
                         {'ATGCC', 'TCGAA', 'AAAAA'})

    def test_overlaps_ignored_range(self):
        seqs = ['AAAAAAAAAAAAAAAAAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        ignored_ranges = {(3, 8), (14, 18)}

        gs = guide_search.GuideSearcherMinimizeGuides(aln, 3, 0, 1.0, (1, 1, 100),
            ignored_ranges=ignored_ranges)

        # For each position i, encode 1 if the guide (of length 3)
        # starting at i overlaps a ignored range, and 0 otherwise
        does_overlap = '01111111000011111100'
        for i in range(len(does_overlap)):
            if does_overlap[i] == '0':
                self.assertFalse(gs._overlaps_ignored_range(i))
            elif does_overlap[i] == '1':
                self.assertTrue(gs._overlaps_ignored_range(i))

    def test_optimal_guide_with_ignored_range(self):
        seqs = ['GTATCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATAGCAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guide is 'AAAAA'
        self.assertEqual(gs._find_oligos_in_window(0, 28),
                         set(['AAAAA']))

        # Do not allow guides overlapping (5, 9)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100),
            ignored_ranges={(5, 9)})
        self.assertEqual(gs._find_oligos_in_window(0, 28),
                         set(['CCCCC', 'GGGGG']))

    def test_with_windows_of_varying_size(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        self.assertEqual(gs._find_oligos_in_window(1, 8),
                         set(['CCAAA', 'GGAAA']))

        # The best guide in [1, 15) is 'AAAAA'
        self.assertEqual(gs._find_oligos_in_window(1, 15),
                         set(['AAAAA']))

    def test_obj_value(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        guides = gs._find_oligos_in_window(1, 8)
        self.assertEqual(gs.obj_value(guides), 2)

    def test_total_frac_bound(self):
        seqs = ['GTTCCAAAAAATCGGCTACCCCCTCTAC',
                'CTACCAAAAAACCTGCTAGGGGGCGTAC',
                'ATCGGAAAAAAACGTCCTCCCCCTGTAC',
                'TTAGGAAAAAAGCGACCGGGGGGTCTAC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        gs = guide_search.GuideSearcherMinimizeGuides(aln, 5, 0, 1.0, (1, 1, 100))

        # The best guides in [1, 8) are 'CCAAA' and 'GGAAA'
        guides = gs._find_oligos_in_window(1, 8)

        # All sequences are bound by a guide
        self.assertEqual(gs.total_frac_bound(guides), 1.0)

    def test_construct_oligo_a(self):
        self.a.clusterer = self.gc
        self.assertEqual(self.a.construct_oligo(0, 4, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3}, 4))
        self.assertIn(self.a.construct_oligo(0, 4, {0: {2,3}}),
                      [('ATCG', {2,3}, 2), ('ACCG', {2,3}, 2)])
        self.assertEqual(self.a.construct_oligo(1, 4, {0: {0,1,2,3,4}}),
                         ('TCGA', {0,1,2,3}, 4))
        self.assertEqual(self.a.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,2,4}, 3))
        self.assertIn(self.a.construct_oligo(2, 4, {0: {0,1,2,3}}),
                      [('CGAA', {0,2}, 2), ('CGAT', {1,3}, 2)])
        self.a.mismatches = 1
        self.assertEqual(self.a.construct_oligo(0, 4, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3,4}, 5))
        self.assertEqual(self.a.construct_oligo(0, 4, {0: {4}}),
                         ('AGCG', {4}, 1))
        self.assertEqual(self.a.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,1,2,3,4}, 5))
        self.assertIn(self.a.construct_oligo(2, 4, {0: {0,1,2,3}}),
                      [('CGAA', {0,1,2,3}, 4), ('CGAT', {0,1,2,3}, 4)])
        self.a.mismatches = 2
        self.assertEqual(self.a.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAA', {0,1,2,3,4}, 5))

    def test_construct_oligo_b(self):
        self.b.clusterer = self.gc
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when the only sequence given (1) has an indel
            self.b.construct_oligo(0, 4, {0: {1}})

    def test_construct_oligo_ambiguous(self):
        # Alignment has many Ns, which makes it difficult to write test cases
        # when clustering (the clusters tend to consist of guides in
        # which a position only has N); so pass None to guide_clusterer in
        # construct_oligo() to skip clustering
        seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        self.assertEqual(self.gen_gs.construct_oligo(0, 4, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,2}, 2))
        self.assertEqual(self.gen_gs.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAA', {0}, 1))
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when 'N' is all that exists at a position
            self.gen_gs.construct_oligo(0, 4, {0: {1,3,4}})
        self.gen_gs.mismatches = 1
        self.assertEqual(self.gen_gs.construct_oligo(0, 4, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,2}, 2))
        self.assertEqual(self.gen_gs.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAT', {0}, 1))
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when a potential guide (here, 'CGAC') cannot
            # bind to any sequence because they all have 'N' somewhere
            self.gen_gs.construct_oligo(2, 4, {0: {2,4}})
        self.gen_gs.mismatches = 2
        self.assertEqual(self.gen_gs.construct_oligo(0, 4, {0: {0,1,2,3,4}}),
                         ('ATCG', {0,1,2,3,4}, 5))
        self.assertEqual(self.gen_gs.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAT', {0,1,2,3}, 4))
        self.assertIn(self.gen_gs.construct_oligo(2, 4, {0: {2,3,4}}),
                      [('CGAC', {2,4}, 2), ('CGAT', {2,3}, 2)])
        self.assertEqual(self.gen_gs.construct_oligo(2, 4, {0: {2,4}}),
                         ('CGAC', {2,4}, 2))
        self.gen_gs.mismatches = 3
        self.assertEqual(self.gen_gs.construct_oligo(2, 4, {0: {0,1,2,3,4}}),
                         ('CGAT', {0,1,2,3,4}, 5))

    def test_construct_oligo_with_large_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        self.gen_gs.clusterer = self.gc

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 3, 1: 1}
        # 'ATCGAA' is most sequences, and let's construct a guide by
        # needing more from the group consisting of these sequences
        self.assertEqual(self.gen_gs.construct_oligo(0, 4, seqs_to_consider,
                                                     num_needed=num_needed),
                         ('ATCG', {0, 1, 3, 4, 5}, 3))

    def test_construct_oligo_with_small_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        self.gen_gs.clusterer = self.gc

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 1, 1: 2}
        # 'ATCGAA' is most sequences, but let's construct a guide by
        # needing more from a group consisting of the 'GGGCCC' sequences
        self.assertEqual(self.gen_gs.construct_oligo(0, 4, seqs_to_consider,
                                                     num_needed=num_needed),
                         ('GGGC', {2, 6}, 2))

    def test_construct_oligo_with_suitable_fn(self):
        seqs = ['GTATCAAAT',
                'CTACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        self.gen_gs.min_oligo_length = guide_length
        self.gen_gs.max_oligo_length = guide_length
        seqs_to_consider = {0: {0, 1, 2, 3}}
        self.gen_gs.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length), k=3)
        self.gen_gs.mismatches = 1

        # The best guide is 'GTATCA'
        self.assertEqual(self.gen_gs.construct_oligo(0, guide_length,
                                                     seqs_to_consider),
                         ('GTATCA', {0, 2, 3}, 3))

        # Do not allow guides with 'TAT' in them
        def f(guide):
            if 'TAT' in guide:
                return False
            else:
                return True
        prev_suitable_fns = self.gen_gs.is_suitable_fns
        self.gen_gs.is_suitable_fns = prev_suitable_fns + [f]

        # Now the best guide is 'CTACCA'
        self.assertEqual(self.gen_gs.construct_oligo(0, guide_length,
                                                     seqs_to_consider),
                         ('CTACCA', {1}, 1))

        # Do not allow guides with 'A' in them
        def f(guide):
            if 'A' in guide:
                return False
            else:
                return True
        self.gen_gs.is_suitable_fns = prev_suitable_fns + [f]

        # Now there is no suitable guide
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_gs.construct_oligo(0, guide_length, seqs_to_consider)

    def test_construct_oligo_with_predictor(self):
        seqs = ['GTATCAAAT',
                'ATACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        self.gen_gs.min_oligo_length = guide_length
        self.gen_gs.max_oligo_length = guide_length
        seqs_to_consider = {0: {0, 1, 2, 3}}
        self.gen_gs.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length), k=3)
        self.gen_gs.mismatches = 1

        self.assertEqual(self.gen_gs.construct_oligo(0, guide_length, seqs_to_consider),
                         ('GTATCA', {0, 2, 3}, 3))

        # Only predict guides starting with 'A' to be active
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'A']
                return y
        self.gen_gs.predictor = PredictorTest()
        # Now the best guide is 'ATACCA'
        self.assertEqual(self.gen_gs.construct_oligo(0, guide_length,
                                                     seqs_to_consider,
                                                     stop_early=False),
                         ('ATACCA', {1}, 1))

        # Only predict guides starting with 'A' to be active, and impose an
        # early stopping criterion
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'A']
                return y
        self.gen_gs.predictor = PredictorTest()
        # With early stopping, it will not find a guide
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_gs.construct_oligo(0, guide_length, seqs_to_consider)

        # Only predictor guides starting with 'C' to be active
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'C']
                return y
        self.gen_gs.predictor = PredictorTest()
        # Now there is no suitable guide
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_gs.construct_oligo(0, guide_length, seqs_to_consider)

    def test_construct_oligo_with_required_flanking(self):
        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'TCAAAT',
                'TCAAAT']
        self.gen_gs.aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 2
        self.gen_gs.min_oligo_length = guide_length
        self.gen_gs.max_oligo_length = guide_length
        self.gen_gs.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=2)
        seqs_to_consider = {0: set(range(len(seqs)))}
        self.gen_gs.mismatches = 1

        # The best guide at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        self.gen_gs.required_flanking_seqs = ('C', None)
        self.assertEqual(self.gen_gs.construct_oligo(2, guide_length,
                                                     seqs_to_consider),
                         ('AA', {0,1,9,10}, 4))

        # The best guide at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        # Now if we require 'M' on the 5' end, 'TT' will be the best guide
        self.gen_gs.required_flanking_seqs = ('M', None)
        self.assertEqual(self.gen_gs.construct_oligo(2, guide_length,
                                                     seqs_to_consider),
                         ('TT', {2,3,4,5,6,7,8}, 7))

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

    def make_gs(self, seqs, soft_constraint=1, hard_constraint=3,
            penalty_strength=0.1, algorithm='random-greedy',
            is_suitable_fns=[]):
        # Predict guides matching target to have activity 1, and
        # starting with 'A' to have activity 2 (otherwise, 0)
        class PredictorTest:
            def __init__(self):
                self.context_nt = 1
            def compute_activity(self, start_pos, pairs):
                y = []
                for target, guide_seq in pairs:
                    target_without_context = target[self.context_nt:len(target)-self.context_nt]
                    if guide_seq == target_without_context:
                        if guide_seq[0] == 'A':
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
                soft_constraint, hard_constraint,
                penalty_strength, (1, 1, 100), algorithm=algorithm,
                predictor=predictor,
                is_suitable_fns=is_suitable_fns)
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

    def test_activities_after_adding_oligo(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')
        o = gs._activities_after_adding_oligo(
                [1, 1, 1, 2, 0],
                [1, 0, 2, 2, 1])
        np.testing.assert_equal(o, np.array([1, 1, 2, 2, 1]))

    def test_analyze_oligos(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')
        o = gs._analyze_oligos(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

    def test_analyze_oligos_memoized(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='random-greedy')

        o = gs._analyze_oligos(4, [3, 0, 1, 1])
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

        o = gs._analyze_oligos(2, [10, 0, 1, 1])
        self.assertEqual(o, {'CGAA': 12/4.0, 'GGGG': 13/4.0, 'CCCC': 12/4.0})

        # Try position 4 again
        o = gs._analyze_oligos(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

        gs._cleanup_memo(4)

        # Try position 4 after cleanup
        o = gs._analyze_oligos(4, [3, 0, 1, 1])
        self.assertEqual(o.keys(), {'AATT', 'GGGG', 'CCCC'})
        self.assertEqual(o, {'AATT': 6/4.0, 'GGGG': 6/4.0, 'CCCC': 5/4.0})

    def test_find_optimal_oligo_in_window_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           algorithm='greedy')

        o = gs._find_optimal_oligo_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        o = gs._find_optimal_oligo_in_window(2, 8, {'AATT'},
                np.array([0, 0, 0, 0]))
        self.assertNotEqual(o[0], 'AATT')

        o = gs._find_optimal_oligo_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 3]))
        self.assertIn(o, [('CCCC', 2), ('CCCC', 3), ('CCCC', 4), ('CCCC', 5)])

        o = gs._find_optimal_oligo_in_window(2, 8, {'ATCG'},
                np.array([2, 1, 0, 2]))
        self.assertIn(o, [('CCCC', 2), ('CCCC', 3), ('CCCC', 4), ('CCCC', 5)])

    def test_find_optimal_oligo_in_window_greedy_with_high_penalty(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           penalty_strength=1,
                           algorithm='greedy')

        o = gs._find_optimal_oligo_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        # Adding any additional guide will create a negative marginal
        # contribution
        with self.assertRaises(search.CannotFindPositiveMarginalContributionError):
            gs._find_optimal_oligo_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 1]))

        # With a higher soft constraint, another guide is ok
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           soft_constraint=2,
                           penalty_strength=1,
                           algorithm='greedy')
        o = gs._find_optimal_oligo_in_window(2, 8, {'AAAA'},
                np.array([10, 1, 0, 1]))
        self.assertEqual(o[0], 'CCCC')

    def test_find_optimal_oligo_in_window_random_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=1,
                           algorithm='random-greedy')
        o = gs._find_optimal_oligo_in_window(2, 8, {}, np.array([0, 0, 0, 0]))
        self.assertEqual(o, ('AATT', 4))

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_oligo_in_window(2, 8, {'AATT'},
                np.array([0, 1, 1, 0]))
        self.assertIn(o[0], ['CGAA', 'GAAT'])

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_oligo_in_window(2, 8, {'ATCG'},
                np.array([4, 0, 0, 2]))
        self.assertIn(o[0], ['GGGG', 'CCCC'])

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_optimal_oligo_in_window(2, 8, {'GGGG'},
                np.array([0, 1, 1, 1]))
        self.assertIn(o[0], ['CGAA', 'GAAT', 'AATT'])
        o = gs._find_optimal_oligo_in_window(2, 8, {'GGGG'},
                np.array([3, 3, 3, 3]))
        self.assertIsNone(o)

    def test_find_optimal_oligo_in_window_random_greedy_with_high_penalty(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           penalty_strength=0.5,
                           algorithm='random-greedy')

        # Adding any additional guide will create a negative marginal
        # contribution
        o = gs._find_optimal_oligo_in_window(2, 8, {'AAAA'},
                np.array([10, 0, 0, 0]))
        self.assertIsNone(o)

    def find_oligos_in_window_different_hard_constraints(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=3,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(o, {'AATT', 'GGGG', 'CCCC'})

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertIn('AATT', o)

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=1,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def find_oligos_in_window_high_penalty(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=3,
                           penalty_strength=0.5,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def test_find_oligos_in_window_different_hard_constraints_greedy(self):
        self.find_oligos_in_window_different_hard_constraints('greedy')

    def test_find_oligos_in_window_different_hard_constraints_random_greedy(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='random-greedy')
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(len(o), 2) # many possibilities for o

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=1,
                           algorithm='random-greedy')
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

    def test_find_oligos_in_window_high_penalty_greedy(self):
        self.find_oligos_in_window_high_penalty('greedy')

    def test_find_oligos_in_window_high_penalty_random_greedy(self):
        self.find_oligos_in_window_high_penalty('random-greedy')

    def find_oligos_in_window_with_is_suitable_fns(self, algo):
        def f(guide):
            if 'AA' in guide:
                return False
            else:
                return True

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=1,
                           algorithm=algo,
                           is_suitable_fns=[f])
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'CGAA', 'GAAT', 'GGGG', 'CCCC'})

    def test_find_oligos_in_window_with_is_suitable_fns_greedy(self):
        self.find_oligos_in_window_with_is_suitable_fns('greedy')

    def test_find_oligos_in_window_with_is_suitable_fns_random_greedy(self):
        self.find_oligos_in_window_with_is_suitable_fns('random-greedy')

    def find_oligos_in_window_with_gaps_and_missing_data(self, algo):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'ATCGA-TTCG',
                           'GGGGGNNNGG',
                           'AACGAATTCG'],
                           hard_constraint=1,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(o, {'AATT'})

        gs = self.make_gs(['ATCGAATTCG',
                           'GGGGGGGGGG',
                           'CCCCCCCCCC',
                           'ATCGA-TTCG',
                           'GGGGGNNNGG',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm=algo)
        o = gs._find_oligos_in_window(2, 8)
        self.assertEqual(len(o), 2)
        if algo == 'greedy':
            self.assertIn('AATT', o)

    def test_find_oligos_in_window_with_gaps_and_missing_data_greedy(self):
        self.find_oligos_in_window_with_gaps_and_missing_data('greedy')

    def test_find_oligos_in_window_with_gaps_and_missing_data_random_greedy(self):
        self.find_oligos_in_window_with_gaps_and_missing_data('random-greedy')

    def test_find_oligos_in_window_at_endpoint(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'AGGGGGGGGA',
                           'ACCCCCCCCA',
                           'AACGAATTCG'],
                           hard_constraint=1)

        # No guides should start at position 0, since it does not
        # permit a context_nt of 1 for evaluating activity
        o = gs._find_oligos_in_window(0, 5)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'TCGA', 'GGGG', 'CCCC', 'ACGA'})

        # No guides should end at the end of the alignment, since it
        # does not permit a context_nt of 1 for evaluating activity
        o = gs._find_oligos_in_window(5, 10)
        self.assertEqual(len(o), 1)
        self.assertIn(list(o)[0], {'ATTC', 'GGGG', 'CCCC'})

        # The ranges below should not work because the test predictor
        # uses a context_nt of 1, which these do not permit
        with self.assertRaises(search.CannotFindAnyOligosError):
            gs._find_oligos_in_window(0, 4)
        with self.assertRaises(search.CannotFindAnyOligosError):
            gs._find_oligos_in_window(6, 10)

    def test_oligo_set_activities(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_oligos_in_window(2, 8)

        activities = gs.oligo_set_activities(2, 8, guides)
        np.testing.assert_equal(activities, np.array([2, 2, 0, 2]))

        activities_percentile = gs.oligo_set_activities_percentile(2, 8,
                guides, [5, 50])
        self.assertEqual(activities_percentile, [0, 2])

    def test_oligo_activities_expected_value(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_oligos_in_window(2, 8)

        aatt = gs.oligo_activities_expected_value(2, 8, 'AATT')
        self.assertEqual(aatt, 1.0)

        aggg = gs.oligo_activities_expected_value(2, 8, 'AGGG')
        self.assertEqual(aggg, 0.5)

    def test_obj_value(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           soft_constraint=1,
                           penalty_strength=0.1,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_oligos_in_window(2, 8)

        expected_activity = (2 + 2 + 0 + 2) / 4.0
        true_obj_value = expected_activity - 0.1*1
        self.assertEqual(gs.obj_value(2, 8, guides), true_obj_value)

    def test_total_frac_bound(self):
        gs = self.make_gs(['ATCGAATTCG',
                           'GGGAGGGGGG',
                           'CCCCCCCCCC',
                           'AACGAATTCG'],
                           hard_constraint=2,
                           algorithm='greedy')
        # Guides should be 'AATT' and 'AGGG'
        guides = gs._find_oligos_in_window(2, 8)

        # 3 of the 4 sequences should be bound
        self.assertEqual(gs.total_frac_bound(2, 8, guides), 0.75)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
