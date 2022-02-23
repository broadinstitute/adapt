"""Tests for oligo module.
"""

import unittest

from adapt.utils import oligo

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestSeqMismatches(unittest.TestCase):
    """Tests the seq_mismatches function.
    """

    def test_equal(self):
        self.assertEqual(oligo.seq_mismatches('ATCG', 'ATCG'), 0)

    def test_equal_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ATCG'), 0)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'AYCG'), 0)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'AYCB'), 0)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'AYSB'), 0)

    def test_one_mismatch(self):
        self.assertEqual(oligo.seq_mismatches('ATCG', 'AACG'), 1)
        self.assertEqual(oligo.seq_mismatches('ATCG', 'ATCA'), 1)
        self.assertEqual(oligo.seq_mismatches('ATCG', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches('ATCN', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches('ATCN', 'ATCG'), 1)

    def test_one_mismatch_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ATCA'), 1)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ARCG'), 1)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ADCA'), 1)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches('AYCN', 'ATCG'), 1)

    def test_multiple_mismatches(self):
        self.assertEqual(oligo.seq_mismatches('ATCG', 'AAAA'), 3)
        self.assertEqual(oligo.seq_mismatches('ATCG', 'ANNN'), 3)

    def test_multiple_mismatches_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ARCA'), 2)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ATAA'), 2)
        self.assertEqual(oligo.seq_mismatches('AYCG', 'ATNN'), 2)
        self.assertEqual(oligo.seq_mismatches('AYNN', 'ATCG'), 2)


class TestSeqMismatchesWithGUPairing(unittest.TestCase):
    """Tests the seq_mismatches_with_gu_pairs function.
    """

    def test_equal(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ATCG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATAG', 'ATGG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ATTG'), 0)

    def test_equal_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATYG', 'ATCG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATYG', 'ATBG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATRG', 'ATGG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATRG', 'ATSG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ATDG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATMG', 'ATDG'), 0)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATSG', 'ATWG'), 0)

    def test_one_mismatch(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'AACG'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ATCA'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCN', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCN', 'ATCG'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATAN', 'ATAG'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('CTAC', 'TTAG'), 1)

    def test_one_mismatch_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ATCA'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ARCG'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ADCA'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ATCN'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCN', 'ATCG'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCA', 'ATCT'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYAC', 'AGGT'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYAC', 'ARGT'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYAY', 'ARGT'), 1)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYAY', 'AYNT'), 1)

    def test_multiple_mismatches(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'AAAA'), 3)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'ANNN'), 3)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'GNNN'), 3)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ATCG', 'GATA'), 2)

    def test_multiple_mismatches_with_ambiguity(self):
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ARCA'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ATAA'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYCG', 'ATNN'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYNN', 'ATCG'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYNN', 'ATCG'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('AYTG', 'ATCA'), 2)
        self.assertEqual(oligo.seq_mismatches_with_gu_pairs('ACTG', 'ATCA'), 2)


class TestMakeComplement(unittest.TestCase):
    """Tests the make_complement function"""
    def test_make_complement(self):
        self.assertEqual(oligo.make_complement(''), '')
        self.assertEqual(oligo.make_complement('T'), 'A')
        self.assertEqual(oligo.make_complement('TGCA'), 'ACGT')
        self.assertEqual(oligo.make_complement('KMRYSWBVHDN-'), 'MKYRSWVBDHN-')


class TestIsComplement(unittest.TestCase):
    """Tests the is_complement function"""
    def test_is_complement(self):
        self.assertEqual(oligo.is_complement('', ''), 1)
        self.assertEqual(oligo.is_complement('A', 'T'), 1)
        self.assertEqual(oligo.is_complement('AT', 'TA'), 1)
        self.assertEqual(oligo.is_complement('AA', 'TA'), 0)
        self.assertEqual(oligo.is_complement('WA', 'TT'), 0.5)
        self.assertEqual(oligo.is_complement('WW', 'TT'), 0.25)
        self.assertEqual(oligo.is_complement('WT', 'TW'), 0.25)
        self.assertEqual(oligo.is_complement('WW', 'WW'), 0.25)


class TestIsSymmetric(unittest.TestCase):
    """Tests the is_symmetric function"""
    def test_is_symmetric(self):
        self.assertEqual(oligo.is_symmetric('AT'), 1)
        self.assertEqual(oligo.is_symmetric('ATAT'), 1)
        self.assertEqual(oligo.is_symmetric('AATA'), 0)
        self.assertEqual(oligo.is_symmetric('ATGCAT'), 1)


class TestQueryTargetEq(unittest.TestCase):
    """Tests the query_target_eq function.
    """

    def test_is_equal(self):
        self.assertTrue(oligo.query_target_eq('N', 'A'))
        self.assertTrue(oligo.query_target_eq('H', 'A'))
        self.assertTrue(oligo.query_target_eq('A', 'A'))
        self.assertTrue(oligo.query_target_eq('ANN', 'AAA'))
        self.assertTrue(oligo.query_target_eq('HNN', 'AAA'))
        self.assertTrue(oligo.query_target_eq('HNA', 'TAA'))
        self.assertTrue(oligo.query_target_eq('HNA', 'TAR'))
        self.assertTrue(oligo.query_target_eq('HNT', 'TAH'))
        self.assertTrue(oligo.query_target_eq('HNT', 'YAH'))

    def test_is_not_equal(self):
        self.assertFalse(oligo.query_target_eq('N', 'N'))
        self.assertFalse(oligo.query_target_eq('A', 'N'))
        self.assertFalse(oligo.query_target_eq('A', 'T'))
        self.assertFalse(oligo.query_target_eq('GAA', 'GAT'))
        self.assertFalse(oligo.query_target_eq('HNN', 'GAA'))
        self.assertFalse(oligo.query_target_eq('HNN', 'GAA'))
        self.assertFalse(oligo.query_target_eq('HNN', 'AAN'))
        self.assertFalse(oligo.query_target_eq('AAA', 'NNN'))
        self.assertFalse(oligo.query_target_eq('AAA', 'YAA'))


class TestOligoBindsWithoutGUPairing(unittest.TestCase):
    """Tests the binds function without allowing G-U pairs.
    """

    def test_oligo_does_bind(self):
        self.assertTrue(oligo.binds('ATCG', 'ATCG', 0, False))
        self.assertTrue(oligo.binds('ATCG', 'AYCG', 0, False))
        self.assertTrue(oligo.binds('ATCG', 'AYCB', 0, False))
        self.assertTrue(oligo.binds('ATCG', 'ATCA', 1, False))
        self.assertTrue(oligo.binds('ATCG', 'ARCS', 1, False))
        self.assertTrue(oligo.binds('ATCG', 'AYCA', 1, False))
        self.assertTrue(oligo.binds('ATCG', 'AYCN', 1, False))
        self.assertTrue(oligo.binds('ATCG', 'AYNN', 2, False))

    def test_oligo_does_not_bind(self):
        self.assertFalse(oligo.binds('ATCG', 'AACG', 0, False))
        self.assertFalse(oligo.binds('ATCG', 'ASCG', 0, False))
        self.assertFalse(oligo.binds('ATCG', 'AYNG', 0, False))
        self.assertFalse(oligo.binds('ATCG', 'ATTG', 0, False))
        self.assertFalse(oligo.binds('ATCG', 'ATAA', 1, False))
        self.assertFalse(oligo.binds('ATCG', 'ATNN', 1, False))


class TestOligoBindsWithGUPairing(unittest.TestCase):
    """Tests the binds function while allowing G-U pairs.
    """

    def test_oligo_does_bind(self):
        self.assertTrue(oligo.binds('ATCG', 'ATCG', 0, True))
        self.assertTrue(oligo.binds('ATCG', 'AYCG', 0, True))
        self.assertTrue(oligo.binds('ATCG', 'AYCB', 0, True))
        self.assertTrue(oligo.binds('ATCG', 'ATCA', 1, True))
        self.assertTrue(oligo.binds('ATCG', 'ARCS', 1, True))
        self.assertTrue(oligo.binds('ATCG', 'AYCA', 1, True))
        self.assertTrue(oligo.binds('ATCG', 'AYCN', 1, True))
        self.assertTrue(oligo.binds('ATCG', 'AYNN', 2, True))
        self.assertTrue(oligo.binds('AACG', 'AGCG', 0, True))
        self.assertTrue(oligo.binds('AACG', 'AGCT', 1, True))
        self.assertTrue(oligo.binds('ACGG', 'ATGG', 0, True))
        self.assertTrue(oligo.binds('ACGG', 'ATGT', 1, True))
        self.assertTrue(oligo.binds('AACC', 'GGTT', 0, True))
        self.assertTrue(oligo.binds('ATCG', 'GCTA', 2, True))
        self.assertTrue(oligo.binds('MTCG', 'RCTA', 2, True))

    def test_oligo_does_not_bind(self):
        self.assertFalse(oligo.binds('ATCG', 'AACG', 0, True))
        self.assertFalse(oligo.binds('ATCG', 'ASCG', 0, True))
        self.assertFalse(oligo.binds('ATCG', 'AYNG', 0, True))
        self.assertFalse(oligo.binds('WTCG', 'CTCG', 0, True))
        self.assertFalse(oligo.binds('ATCG', 'ATAA', 1, True))
        self.assertFalse(oligo.binds('ATCG', 'ATNN', 1, True))
        self.assertFalse(oligo.binds('ATCG', 'GCTA', 1, True))
        self.assertFalse(oligo.binds('CTCG', 'GCTA', 2, True))
        self.assertFalse(oligo.binds('WTCG', 'CTCG', 0, True))


class TestGCFrac(unittest.TestCase):
    """Tests the gc_frac function.
    """

    def test_gc_frac(self):
        self.assertEqual(oligo.gc_frac('ATTA'), 0)
        self.assertEqual(oligo.gc_frac('GGCC'), 1)
        self.assertEqual(oligo.gc_frac('AGTC'), 0.5)


class TestOverlapInSeq(unittest.TestCase):
    """Tests the overlap_in_seq() function.
    """

    def test_disjoint(self):
        self.assertEqual(oligo.overlap_in_seq(
            ['ATC'], 'AATTCCATCGG', 0, False),
            {6,7,8})
        self.assertEqual(oligo.overlap_in_seq(
            ['ATC', 'ATT'], 'AATTCCATCGG', 0, False),
            {1,2,3,6,7,8})

    def test_overlapping(self):
        self.assertEqual(oligo.overlap_in_seq(
            ['ATT', 'TCC', 'TCG'], 'AATTCCATCGG', 0, False),
            {1,2,3,4,5,7,8,9})

    def test_at_ends(self):
        self.assertEqual(oligo.overlap_in_seq(
            ['AAT', 'CGG'], 'AATTCCATCGG', 0, False),
            {0,1,2,8,9,10})

    def test_with_mismatches(self):
        self.assertEqual(oligo.overlap_in_seq(
            ['ATA'], 'AATTCCATCGG', 1, False),
            {1,2,3,6,7,8})
