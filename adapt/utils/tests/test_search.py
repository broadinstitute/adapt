"""Tests for search module.
"""

import logging
import random
import unittest

import numpy as np

from adapt import alignment
from adapt.utils import search
from adapt.utils import index_compress

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
        self.aln = alignment.Alignment.from_list_of_seqs(self.seqs)

        self.s = search.OligoSearcher(self.aln, 3, 5, (1, 1, 100),
                                 predictor=PredictorTest())

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
            ignored_ranges=ignored_ranges)

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

    def test_oligo_set_activities(self):
        oligos = {'AATT', 'AGGG'}
        self.s._selected_positions = {'AATT': [4], 'AGGG': [3]}

        activities = self.s.oligo_set_activities(2, 8, oligos)
        np.testing.assert_equal(activities, np.array([2, 2, 0, 2]))

        activities_percentile = self.s.oligo_set_activities_percentile(2, 8,
                oligos, [5, 50])
        self.assertEqual(activities_percentile, [0, 2])

        activities_expected = self.s.oligo_set_activities_expected_value(2, 8,
                oligos)
        self.assertEqual(activities_expected, 1.5)

        activities_expected = self.s.oligo_set_activities_expected_value(2, 8,
                oligos, activities=[0, 2, 0, 2])
        self.assertEqual(activities_expected, 1)

    def test_find_oligos_for_each_window(self):
        s = OligoSearcherTest(self.aln, 3, 5, (1, 1, 100),
                                     predictor=PredictorTest())
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


class PredictorTest:
    def __init__(self):
        self.context_nt = 1

    def compute_activity(self, start_pos, pairs):
        y = []
        for target, guide_seq in pairs:
            target_without_context = target[self.context_nt:len(target) -
                                            self.context_nt]
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

