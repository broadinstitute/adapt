"""Tests for guide module.
"""

import unittest

from dxguidedesign.utils import guide

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestGuideBinds(unittest.TestCase):
    """Tests the guide_binds function and its helper(s).
    """

    def test_equal(self):
        self.assertEqual(guide.seq_mismatches('ATCG', 'ATCG'), 0)

    def test_equal_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches('AYCG', 'ATCG'), 0)
        self.assertEqual(guide.seq_mismatches('AYCG', 'AYCG'), 0)
        self.assertEqual(guide.seq_mismatches('AYCG', 'AYCB'), 0)
        self.assertEqual(guide.seq_mismatches('AYCG', 'AYSB'), 0)

    def test_one_mismatch(self):
        self.assertEqual(guide.seq_mismatches('ATCG', 'AACG'), 1)
        self.assertEqual(guide.seq_mismatches('ATCG', 'ATCA'), 1)
        self.assertEqual(guide.seq_mismatches('ATCG', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches('ATCN', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches('ATCN', 'ATCG'), 1)

    def test_one_mismatch_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches('AYCG', 'ATCA'), 1)
        self.assertEqual(guide.seq_mismatches('AYCG', 'ARCG'), 1)
        self.assertEqual(guide.seq_mismatches('AYCG', 'ADCA'), 1)
        self.assertEqual(guide.seq_mismatches('AYCG', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches('AYCN', 'ATCG'), 1)

    def test_multiple_mismatches(self):
        self.assertEqual(guide.seq_mismatches('ATCG', 'AAAA'), 3)
        self.assertEqual(guide.seq_mismatches('ATCG', 'ANNN'), 3)

    def test_multiple_mismatches_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches('AYCG', 'ARCA'), 2)
        self.assertEqual(guide.seq_mismatches('AYCG', 'ATAA'), 2)
        self.assertEqual(guide.seq_mismatches('AYCG', 'ATNN'), 2)
        self.assertEqual(guide.seq_mismatches('AYNN', 'ATCG'), 2)

    def test_guide_does_bind(self):
        self.assertTrue(guide.guide_binds('ATCG', 'ATCG', 0))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCG', 0))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCB', 0))
        self.assertTrue(guide.guide_binds('ATCG', 'ATCA', 1))
        self.assertTrue(guide.guide_binds('ATCG', 'ARCS', 1))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCA', 1))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCN', 1))
        self.assertTrue(guide.guide_binds('ATCG', 'AYNN', 2))

    def test_guide_does_not_bind(self):
        self.assertFalse(guide.guide_binds('ATCG', 'AACG', 0))
        self.assertFalse(guide.guide_binds('ATCG', 'ASCG', 0))
        self.assertFalse(guide.guide_binds('ATCG', 'AYNG', 0))
        self.assertFalse(guide.guide_binds('ATCG', 'ATAA', 1))
        self.assertFalse(guide.guide_binds('ATCG', 'ATNN', 1))
