"""Tests for search module.
"""

import logging
import random
import unittest

import numpy as np

from adapt import alignment
from adapt.utils import search
from adapt.utils import index_compress
from adapt.utils import lsh

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'


class TestOligoSearcher(unittest.TestCase):
    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.seqs = ['ATCGAATTCG',
                     'GGGAGGGGGG',
                     'CCCCCCCCCC',
                     'AACGAATTCG']
        self.oligos = {'AATT', 'AGGG'}
        self.aln = alignment.Alignment.from_list_of_seqs(self.seqs)

        self.s = search.OligoSearcher(self.aln, 3, 5, (1, 1, 100),
                predictor=PredictorTest(), obj_type='max')

        # Create a generic searcher that can have its alignment overriden
        # (for testing construct_oligo)
        gen_seqs = ['AAAA']
        gen_aln = alignment.Alignment.from_list_of_seqs(gen_seqs)
        self.gen_s = search.OligoSearcherMinimizeNumber(1.0, 0,
            missing_data_params=(1, 1, 100), aln=gen_aln,
            min_oligo_length=4, max_oligo_length=4)

    def test_compute_memoized_and_cleanup_memo(self):
        def call_fn():
            return ('CGA', {0, 3}, 2)

        seq_needed = index_compress.compress_mostly_contiguous({0,1,2,3})
        key = (frozenset({(0, frozenset(seq_needed))}), None)
        def key_fn():
            return key

        self.assertEqual(len(self.s._memo), 0)
        first_call = self.s._compute_memoized(3, call_fn, key_fn)
        self.assertEqual(first_call, ('CGA', {0, 3}, 2))
        self.assertIn(key, self.s._memo)
        self.assertEqual(len(self.s._memo), 1)
        second_call = self.s._compute_memoized(3, call_fn, key_fn)
        self.assertIs(first_call, second_call)
        self.assertIn(key, self.s._memo)
        self.assertEqual(len(self.s._memo), 1)

        def call_fn():
            return ('GAA', {0, 3}, 2)

        def key_fn():
            raise Exception("_compute_memoized should not call key_fn if "
                            "use_last is True")

        third_call = self.s._compute_memoized(4, call_fn, key_fn, use_last=True)
        self.assertEqual(third_call, ('GAA', {0, 3}, 2))
        self.assertIn(key, self.s._memo)
        self.assertEqual(len(self.s._memo), 1)

        self.s._cleanup_memo(3)
        self.assertIn(key, self.s._memo)
        self.s._cleanup_memo(4)
        self.assertNotIn(key, self.s._memo)

    def test_overlaps_ignored_range(self):
        seqs = ['AAAAAAAAAAAAAAAAAAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        ignored_ranges = {(3, 8), (15, 19)}

        s = search.OligoSearcher(aln, 3, 5, (1, 1, 100),
            ignored_ranges=ignored_ranges, obj_type='max')

        # For each position i, encode 1 if the oligo (length 3, 4, or 5
        # respectively) starting at i overlaps a ignored range, and 0 otherwise
        does_overlap_3 = '011111110000011111100'
        does_overlap_4 = '111111110000111111100'
        does_overlap_5 = '111111110001111111100'

        for i in range(len(seqs[0])):
            self.assertEqual(s._overlaps_ignored_range(i),
                             (does_overlap_3[i] == '1'))
            self.assertEqual(s._overlaps_ignored_range(i, olg_len=3),
                             (does_overlap_3[i] == '1'))
            self.assertEqual(s._overlaps_ignored_range(i, olg_len=4),
                             (does_overlap_4[i] == '1'))
            self.assertEqual(s._overlaps_ignored_range(i, olg_len=5),
                             (does_overlap_5[i] == '1'))

    def test_oligo_set_activities_per_oligo(self):
        self.s._selected_positions = {'AATT': [4], 'AGGG': [3]}

        per_olg_activities, activities = self.s.oligo_set_activities_per_oligo(
            2, 8, self.oligos)
        np.testing.assert_equal(activities, np.array([2, 2, 0, 2],))
        self.assertIn('AATT', per_olg_activities)
        self.assertIn('AGGG', per_olg_activities)
        np.testing.assert_equal(per_olg_activities['AATT'], np.array([2, 0, 0, 2]))
        np.testing.assert_equal(per_olg_activities['AGGG'], np.array([0, 2, 0, 0]))

    def test_oligo_set_activities(self):
        self.s._selected_positions = {'AATT': [4], 'AGGG': [3]}

        activities = self.s.oligo_set_activities(2, 8, self.oligos)
        np.testing.assert_equal(activities, np.array([2, 2, 0, 2]))

        activities_percentile = self.s.oligo_set_activities_percentile(2, 8,
                self.oligos, [5, 50])
        self.assertEqual(activities_percentile, [0, 2])

        activities_expected = self.s.oligo_set_activities_expected_value(2, 8,
                self.oligos)
        self.assertEqual(activities_expected, 1.5)

        activities_expected = self.s.oligo_set_activities_expected_value(2, 8,
                self.oligos, activities=[0, 2, 0, 2])
        self.assertEqual(activities_expected, 1)

    def test_oligo_activities_expected_value(self):
        self.s._selected_positions = {'AATT': [4], 'AGGG': [3]}

        aatt = self.s.oligo_activities_expected_value(2, 8, 'AATT')
        self.assertEqual(aatt, 1.0)

        aggg = self.s.oligo_activities_expected_value(2, 8, 'AGGG')
        self.assertEqual(aggg, 0.5)

    def test_oligo_set_activities_expected_value_per_oligo(self):
        self.s._selected_positions = {'AATT': [4], 'AGGG': [3]}

        per_olg_expected = self.s.oligo_set_activities_expected_value_per_oligo(
            2, 8, self.oligos)

        self.assertIn('AATT', per_olg_expected)
        self.assertIn('AGGG', per_olg_expected)
        self.assertEqual(per_olg_expected['AATT'], 4/3)
        self.assertEqual(per_olg_expected['AGGG'], 1)

    def test_find_oligos_for_each_window(self):
        s = OligoSearcherTest(self.aln, 3, 5, (1, 1, 100),
                predictor=PredictorTest(), obj_type='max')
        olgs_win =  [(0, 7, {'CGAAT', 'ATC'}),
                     (1, 8, {'GAATT', 'TCG'}),
                     (2, 9, {'AATTC', 'CGA'}),
                     (3, 10, {'ATTCG', 'GAA'})]

        self.assertEqual(len(s._memo), 0)
        for i, p in enumerate(s._find_oligos_for_each_window(7)):
            self.assertIn(0, s._memo)
            self.assertIn(i, s._memo[0])
            self.assertNotIn(i-1, s._memo[0])
            self.assertEqual(p, olgs_win[i])
        self.assertNotIn(3, s._memo[0])


    def test_construct_oligo_a(self):
        seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,1,2,3})
        self.assertAlmostEqual(score, .8)
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {2,3}})
        self.assertIn(olg, ['ATCG','ACCG'])
        self.assertEqual(olg_seqs, {2,3})
        self.assertAlmostEqual(score, .4)
        olg, olg_seqs, score = self.gen_s.construct_oligo(1, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'TCGA')
        self.assertEqual(olg_seqs, {0,1,2,3})
        self.assertAlmostEqual(score, .8)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAA')
        self.assertEqual(olg_seqs, {0,2,4})
        self.assertAlmostEqual(score, .6)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3}})
        self.assertIn(olg, ['CGAA','CGAT'])
        if olg == 'CGAA':
            self.assertEqual(olg_seqs, {0,2})
        else:
            self.assertEqual(olg_seqs, {1,3})
        self.assertAlmostEqual(score, .4)
        self.gen_s.mismatches = 1
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,1,2,3,4})
        self.assertAlmostEqual(score, 1)
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {4}})
        self.assertEqual(olg, 'AGCG')
        self.assertEqual(olg_seqs, {4})
        self.assertAlmostEqual(score, .2)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAA')
        self.assertEqual(olg_seqs, {0,1,2,3,4})
        self.assertAlmostEqual(score, 1)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3}})
        self.assertIn(olg, ['CGAA','CGAT'])
        self.assertEqual(olg_seqs, {0,1,2,3})
        self.assertAlmostEqual(score, .8)
        self.gen_s.mismatches = 2
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAA')
        self.assertEqual(olg_seqs, {0,1,2,3,4})
        self.assertAlmostEqual(score, 1)

    def test_construct_oligo_b(self):
        seqs = ['ATCGAA', 'ATC-AA']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when the only sequence given (1) has an indel
            self.gen_s.construct_oligo(0, 4, {0: {1}})

    def test_construct_oligo_ambiguous(self):
        # Alignment has many Ns, which makes it difficult to write test cases
        # when clustering (the clusters tend to consist of oligos in
        # which a position only has N); so pass None to clusterer in
        # construct_oligo() to skip clustering
        self.gen_s.clusterer = None
        seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,2})
        self.assertAlmostEqual(score, .4)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAA')
        self.assertEqual(olg_seqs, {0})
        self.assertAlmostEqual(score, .2)
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when 'N' is all that exists at a position
            self.gen_s.construct_oligo(0, 4, {0: {1,3,4}})
        self.gen_s.mismatches = 1
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,2})
        self.assertAlmostEqual(score, .4)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAT')
        self.assertEqual(olg_seqs, {0})
        self.assertAlmostEqual(score, .2)
        with self.assertRaises(search.CannotConstructOligoError):
            # Should fail when a potential oligo (here, 'CGAC') cannot
            # bind to any sequence because they all have 'N' somewhere
            self.gen_s.construct_oligo(2, 4, {0: {2,4}})
        self.gen_s.mismatches = 2
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,1,2,3,4})
        self.assertAlmostEqual(score, 1)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAT')
        self.assertEqual(olg_seqs, {0,1,2,3})
        self.assertAlmostEqual(score, .8)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {2,3,4}})
        self.assertIn(olg, ['CGAC','CGAT'])
        if olg == 'CGAC':
            self.assertEqual(olg_seqs, {2,4})
        else:
            self.assertEqual(olg_seqs, {2,3})
        self.assertAlmostEqual(score, .4)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {2,4}})
        self.assertEqual(olg, 'CGAC')
        self.assertEqual(olg_seqs, {2,4})
        self.assertAlmostEqual(score, .4)
        self.gen_s.mismatches = 3
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, 4, {0: {0,1,2,3,4}})
        self.assertEqual(olg, 'CGAT')
        self.assertEqual(olg_seqs, {0,1,2,3,4})
        self.assertAlmostEqual(score, 1)

    def test_construct_oligo_with_large_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        percent_needed = {0: 3/7, 1: 1/7}
        # 'ATCGAA' is most sequences, and let's construct a oligo by
        # needing more from the group consisting of these sequences
        olg, olg_seqs, score = self.gen_s.construct_oligo(
            0, 4, seqs_to_consider, percent_needed=percent_needed)
        self.assertEqual(olg, 'ATCG')
        self.assertEqual(olg_seqs, {0,1,3,4,5})
        self.assertAlmostEqual(score, 3/7)

    def test_construct_oligo_with_small_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        percent_needed = {0: 1/7, 1: 2/7}
        # 'ATCGAA' is most sequences, but let's construct a oligo by
        # needing more from a group consisting of the 'GGGCCC' sequences
        olg, olg_seqs, score = self.gen_s.construct_oligo(
            0, 4, seqs_to_consider, percent_needed=percent_needed)
        self.assertEqual(olg, 'GGGC')
        self.assertEqual(olg_seqs, {2,6})
        self.assertAlmostEqual(score, 2/7)

    def test_construct_oligo_with_suitable_fn(self):
        seqs = ['GTATCAAAT',
                'CTACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        oligo_length = 6
        self.gen_s.min_oligo_length = oligo_length
        self.gen_s.max_oligo_length = oligo_length
        seqs_to_consider = {0: {0, 1, 2, 3}}
        self.gen_s.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(oligo_length), k=3)
        self.gen_s.mismatches = 1

        # The best oligo is 'GTATCA'
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, oligo_length,
                                                          seqs_to_consider)
        self.assertEqual(olg, 'GTATCA')
        self.assertEqual(olg_seqs, {0,2,3})
        self.assertAlmostEqual(score, .75)

        # Do not allow oligos with 'TAT' in them
        def f(oligo):
            if 'TAT' in oligo:
                return False
            else:
                return True
        prev_suitable_fns = self.gen_s.pre_filter_fns
        self.gen_s.pre_filter_fns = prev_suitable_fns + [f]

        # Now the best oligo is 'CTACCA'
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, oligo_length,
                                                          seqs_to_consider)
        self.assertEqual(olg, 'CTACCA')
        self.assertEqual(olg_seqs, {1})
        self.assertAlmostEqual(score, .25)

        # Do not allow oligos with 'A' in them
        def f(oligo):
            if 'A' in oligo:
                return False
            else:
                return True
        self.gen_s.pre_filter_fns = prev_suitable_fns + [f]

        # Now there is no suitable oligo
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_s.construct_oligo(0, oligo_length, seqs_to_consider)

    def test_construct_oligo_with_predictor(self):
        seqs = ['GTATCAAAT',
                'ATACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        oligo_length = 6
        self.gen_s.min_oligo_length = oligo_length
        self.gen_s.max_oligo_length = oligo_length
        seqs_to_consider = {0: {0, 1, 2, 3}}
        self.gen_s.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(oligo_length), k=3)
        self.gen_s.mismatches = 1

        olg, olg_seqs, score = self.gen_s.construct_oligo(0, oligo_length,
                                                          seqs_to_consider)
        self.assertEqual(olg, 'GTATCA')
        self.assertEqual(olg_seqs, {0,2,3})
        self.assertAlmostEqual(score, .75)

        # Only predict oligos starting with start_base to be active
        class PredictorTest:
            def __init__(self, start_base):
                self.context_nt = 0
                self.min_activity = 0
                self.start_base = start_base
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, oligo in pairs:
                    y += [oligo[0] == self.start_base]
                return y
        self.gen_s.predictor = PredictorTest('A')
        # Now the best oligo is 'ATACCA'
        olg, olg_seqs, score = self.gen_s.construct_oligo(0, oligo_length,
                                                          seqs_to_consider,
                                                          stop_early=False)
        self.assertEqual(olg, 'ATACCA')
        self.assertEqual(olg_seqs, {1})
        self.assertAlmostEqual(score, .25)

        # Only predict oligos starting with 'A' to be active, and impose an
        # early stopping criterion
        # With early stopping, it will not find a oligo
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_s.construct_oligo(0, oligo_length, seqs_to_consider)

        self.gen_s.predictor = PredictorTest('C')
        # Now there is no suitable oligo
        with self.assertRaises(search.CannotConstructOligoError):
            self.gen_s.construct_oligo(0, oligo_length, seqs_to_consider)

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
        self.gen_s.aln = alignment.Alignment.from_list_of_seqs(seqs)
        oligo_length = 2
        self.gen_s.min_oligo_length = oligo_length
        self.gen_s.max_oligo_length = oligo_length
        self.gen_s.clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(oligo_length),
            k=2)
        seqs_to_consider = {0: set(range(len(seqs)))}
        self.gen_s.mismatches = 1

        # The best oligo at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        self.gen_s.required_flanking_seqs = ('C', None)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, oligo_length,
                                                          seqs_to_consider)
        self.assertEqual(olg, 'AA')
        self.assertEqual(olg_seqs, {0,1,9,10})
        self.assertAlmostEqual(score, 4/11)

        # The best oligo at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        # Now if we require 'M' on the 5' end, 'TT' will be the best oligo
        self.gen_s.required_flanking_seqs = ('M', None)
        olg, olg_seqs, score = self.gen_s.construct_oligo(2, oligo_length,
                                                          seqs_to_consider)

        self.assertEqual(olg, 'TT')
        self.assertEqual(olg_seqs, {2,3,4,5,6,7,8})
        self.assertAlmostEqual(score, 7/11)


class PredictorTest:
    def __init__(self):
        self.context_nt = 1
        self.min_activity = 0

    def compute_activity(self, start_pos, pairs):
        y = []
        for target, oligo_seq in pairs:
            target_without_context = target[self.context_nt:len(target) -
                                            self.context_nt]
            if oligo_seq == target_without_context:
                if oligo_seq[0] == 'A':
                    y += [2]
                else:
                    y += [1]
            else:
                y += [0]
        return y

    def cleanup_memoized(self, pos):
        pass


class OligoSearcherTest(search.OligoSearcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _find_oligos_in_window(self, start, end):
        if 0 not in self._memo:
            self._memo[0] = {}
        for i in range(start, end):
            self._memo[0][i] = 0

        seqs = self.aln.make_list_of_seqs(seqs_to_consider=[0, -1])
        return {seqs[0][start:start+self.min_oligo_length],
                seqs[1][end-self.max_oligo_length:end]}

