"""Tests for guide module.
"""

import unittest

from adapt.utils import guide

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestSeqMismatches(unittest.TestCase):
    """Tests the seq_mismatches function.
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


class TestSeqMismatchesWithGUPairing(unittest.TestCase):
    """Tests the seq_mismatches_with_gu_pairs function.
    """

    def test_equal(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ATCG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATAG', 'ATGG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ATTG'), 0)

    def test_equal_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATYG', 'ATCG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATYG', 'ATBG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATRG', 'ATGG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATRG', 'ATSG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ATDG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATMG', 'ATDG'), 0)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATSG', 'ATWG'), 0)

    def test_one_mismatch(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'AACG'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ATCA'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCN', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCN', 'ATCG'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATAN', 'ATAG'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('CTAC', 'TTAG'), 1)

    def test_one_mismatch_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ATCA'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ARCG'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ADCA'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ATCN'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCN', 'ATCG'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCA', 'ATCT'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYAC', 'AGGT'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYAC', 'ARGT'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYAY', 'ARGT'), 1)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYAY', 'AYNT'), 1)

    def test_multiple_mismatches(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'AAAA'), 3)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'ANNN'), 3)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'GNNN'), 3)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ATCG', 'GATA'), 2)

    def test_multiple_mismatches_with_ambiguity(self):
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ARCA'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ATAA'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYCG', 'ATNN'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYNN', 'ATCG'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYNN', 'ATCG'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('AYTG', 'ATCA'), 2)
        self.assertEqual(guide.seq_mismatches_with_gu_pairs('ACTG', 'ATCA'), 2)


class TestQueryTargetEq(unittest.TestCase):
    """Tests the query_target_eq function.
    """

    def test_is_equal(self):
        self.assertTrue(guide.query_target_eq('N', 'A'))
        self.assertTrue(guide.query_target_eq('H', 'A'))
        self.assertTrue(guide.query_target_eq('A', 'A'))
        self.assertTrue(guide.query_target_eq('ANN', 'AAA'))
        self.assertTrue(guide.query_target_eq('HNN', 'AAA'))
        self.assertTrue(guide.query_target_eq('HNA', 'TAA'))
        self.assertTrue(guide.query_target_eq('HNA', 'TAR'))
        self.assertTrue(guide.query_target_eq('HNT', 'TAH'))
        self.assertTrue(guide.query_target_eq('HNT', 'YAH'))

    def test_is_not_equal(self):
        self.assertFalse(guide.query_target_eq('N', 'N'))
        self.assertFalse(guide.query_target_eq('A', 'N'))
        self.assertFalse(guide.query_target_eq('A', 'T'))
        self.assertFalse(guide.query_target_eq('GAA', 'GAT'))
        self.assertFalse(guide.query_target_eq('HNN', 'GAA'))
        self.assertFalse(guide.query_target_eq('HNN', 'GAA'))
        self.assertFalse(guide.query_target_eq('HNN', 'AAN'))
        self.assertFalse(guide.query_target_eq('AAA', 'NNN'))
        self.assertFalse(guide.query_target_eq('AAA', 'YAA'))


class TestGuideBindsWithoutGUPairing(unittest.TestCase):
    """Tests the guide_binds function without allowing G-U pairs.
    """

    def test_guide_does_bind(self):
        self.assertTrue(guide.guide_binds('ATCG', 'ATCG', 0, False))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCG', 0, False))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCB', 0, False))
        self.assertTrue(guide.guide_binds('ATCG', 'ATCA', 1, False))
        self.assertTrue(guide.guide_binds('ATCG', 'ARCS', 1, False))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCA', 1, False))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCN', 1, False))
        self.assertTrue(guide.guide_binds('ATCG', 'AYNN', 2, False))

    def test_guide_does_not_bind(self):
        self.assertFalse(guide.guide_binds('ATCG', 'AACG', 0, False))
        self.assertFalse(guide.guide_binds('ATCG', 'ASCG', 0, False))
        self.assertFalse(guide.guide_binds('ATCG', 'AYNG', 0, False))
        self.assertFalse(guide.guide_binds('ATCG', 'ATTG', 0, False))
        self.assertFalse(guide.guide_binds('ATCG', 'ATAA', 1, False))
        self.assertFalse(guide.guide_binds('ATCG', 'ATNN', 1, False))


class TestGuideBindsWithGUPairing(unittest.TestCase):
    """Tests the guide_binds function while allowing G-U pairs.
    """

    def test_guide_does_bind(self):
        self.assertTrue(guide.guide_binds('ATCG', 'ATCG', 0, True))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCG', 0, True))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCB', 0, True))
        self.assertTrue(guide.guide_binds('ATCG', 'ATCA', 1, True))
        self.assertTrue(guide.guide_binds('ATCG', 'ARCS', 1, True))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCA', 1, True))
        self.assertTrue(guide.guide_binds('ATCG', 'AYCN', 1, True))
        self.assertTrue(guide.guide_binds('ATCG', 'AYNN', 2, True))
        self.assertTrue(guide.guide_binds('AACG', 'AGCG', 0, True))
        self.assertTrue(guide.guide_binds('AACG', 'AGCT', 1, True))
        self.assertTrue(guide.guide_binds('ACGG', 'ATGG', 0, True))
        self.assertTrue(guide.guide_binds('ACGG', 'ATGT', 1, True))
        self.assertTrue(guide.guide_binds('AACC', 'GGTT', 0, True))
        self.assertTrue(guide.guide_binds('ATCG', 'GCTA', 2, True))
        self.assertTrue(guide.guide_binds('MTCG', 'RCTA', 2, True))

    def test_guide_does_not_bind(self):
        self.assertFalse(guide.guide_binds('ATCG', 'AACG', 0, True))
        self.assertFalse(guide.guide_binds('ATCG', 'ASCG', 0, True))
        self.assertFalse(guide.guide_binds('ATCG', 'AYNG', 0, True))
        self.assertFalse(guide.guide_binds('WTCG', 'CTCG', 0, True))
        self.assertFalse(guide.guide_binds('ATCG', 'ATAA', 1, True))
        self.assertFalse(guide.guide_binds('ATCG', 'ATNN', 1, True))
        self.assertFalse(guide.guide_binds('ATCG', 'GCTA', 1, True))
        self.assertFalse(guide.guide_binds('CTCG', 'GCTA', 2, True))
        self.assertFalse(guide.guide_binds('WTCG', 'CTCG', 0, True))


class TestGCFrac(unittest.TestCase):
    """Tests the gc_frac function.
    """

    def test_gc_frac(self):
        self.assertEqual(guide.gc_frac('ATTA'), 0)
        self.assertEqual(guide.gc_frac('GGCC'), 1)
        self.assertEqual(guide.gc_frac('AGTC'), 0.5)


class TestGuideOverlapInSeq(unittest.TestCase):
    """Tests the guide_overlap_in_seq() function.
    """

    def test_disjoint(self):
        self.assertEqual(guide.guide_overlap_in_seq(
            ['ATC'], 'AATTCCATCGG', 0, False),
            {6,7,8})
        self.assertEqual(guide.guide_overlap_in_seq(
            ['ATC', 'ATT'], 'AATTCCATCGG', 0, False),
            {1,2,3,6,7,8})

    def test_overlapping(self):
        self.assertEqual(guide.guide_overlap_in_seq(
            ['ATT', 'TCC', 'TCG'], 'AATTCCATCGG', 0, False),
            {1,2,3,4,5,7,8,9})

    def test_at_ends(self):
        self.assertEqual(guide.guide_overlap_in_seq(
            ['AAT', 'CGG'], 'AATTCCATCGG', 0, False),
            {0,1,2,8,9,10})

    def test_with_mismatches(self):
        self.assertEqual(guide.guide_overlap_in_seq(
            ['ATA'], 'AATTCCATCGG', 1, False),
            {1,2,3,6,7,8})
