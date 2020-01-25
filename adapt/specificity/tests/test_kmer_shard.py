"""Tests for kmer_shard module.
"""

import unittest

from adapt.specificity import kmer_shard

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestTrieSpace(unittest.TestCase):

    def test_get(self):
        # Build and fetch from a trie space
        ts = kmer_shard.TrieSpace()
        t = ts.get('abc', make=True)
        t.insert([('A', 4)])
        t = ts.get('abc', make=False)
        self.assertEqual(t.query('A') , [4])
        self.assertIsNone(ts.get('def', make=False))


class TestSignatureFunctions(unittest.TestCase):

    def test_signatures_with_mismatches(self):
        self.assertCountEqual(kmer_shard._signatures_with_mismatches('GTG', 0),
                ['GTG'])
        self.assertCountEqual(kmer_shard._signatures_with_mismatches('GTG', 1),
                ['GTG', 'TTG', 'GGG', 'GTT'])
        self.assertCountEqual(kmer_shard._signatures_with_mismatches('GTG', 2),
                ['GTG', 'TTG', 'GGG', 'GTT', 'TGG', 'TTT', 'GGT'])

    def test_full_signature(self):
        self.assertEqual(kmer_shard._full_signature('ATCGATCG'),
                'GTTGGTTG')

    def test_full_signatures_with_mismatches(self):
        self.assertCountEqual(
                kmer_shard._full_signatures_with_mismatches('ATG', 1),
                ['GTG', 'TTG', 'GGG', 'GTT'])

    def test_split_signature(self):
        self.assertEqual(kmer_shard._split_signature('ATCG', 0),
                'GT')
        self.assertEqual(kmer_shard._split_signature('ATCG', 1),
                'TG')
        self.assertEqual(kmer_shard._split_signature('ATCGA', 0),
                'GTT')
        self.assertEqual(kmer_shard._split_signature('ATCGA', 1),
                'GG')

    def test_split_signatures_with_mismatches(self):
        self.assertCountEqual(
                kmer_shard._split_signatures_with_mismatches('ATCG', 0, 1),
                ['GT', 'TT', 'GG'])
        self.assertCountEqual(
                kmer_shard._split_signatures_with_mismatches('ATCG', 1, 1),
                ['TG', 'GG', 'TT'])


class TestTrieSpaceOfKmersEvenK(unittest.TestCase):
    """Test TrieSpaceOfKmersFullSig and TrieSpaceOfKmersSplitSig using
    k-mers with even length (i.e., k is even).

    This is useful to split up because the methods treat even and odd lengths
    slightly differently.
    """

    def build(self, cls):
        # cls is the class to instantiate
        tsok = cls()
        tsok.add([
            ('ATCGAT', {(0,1), (0,2), (0,4), (1,1)}),
            ('ATAAAT', {(0,3), (2,1)})
        ])
        tsok.add([
            ('GGGGGG', {(3,1), (3,2)}),
            ('TTCGAT', {(0,5), (1,2)})
        ])
        return tsok

    def query_tests(self, tsok):
        # Exact matches, no G-U pairing
        self.assertEqual(tsok.query('ATCGAT', m=0, gu_pairing=False),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('ATAAAT', m=0, gu_pairing=False),
                {0: {3}, 2: {1}})
        self.assertEqual(tsok.query('GGGGGG', m=0, gu_pairing=False),
                {3: {1,2}})
        self.assertEqual(tsok.query('TTCGAT', m=0, gu_pairing=False),
                {0: {5}, 1: {2}})
        self.assertEqual(tsok.query('CCCCCC', m=0, gu_pairing=False),
                {})

        # Queries 1 mismatch away, no G-U pairing
        self.assertEqual(tsok.query('ATCGAT', m=1, gu_pairing=False),
                {0: {1,2,4,5}, 1: {1,2}})
        self.assertEqual(tsok.query('GGTGGG', m=1, gu_pairing=False),
                {3: {1,2}})
        self.assertEqual(tsok.query('GGTTGG', m=1, gu_pairing=False),
                {})

        # Exact matches, with G-U pairing
        self.assertEqual(tsok.query('ACCGAT', m=0, gu_pairing=True),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('ACCAAT', m=0, gu_pairing=True),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('AAAAAA', m=0, gu_pairing=True),
                {3: {1,2}})

        # Queries 1 mismatch away, with G-U pairing
        self.assertEqual(tsok.query('ACCGAT', m=1, gu_pairing=True),
                {0: {1,2,4,5}, 1: {1,2}})
        self.assertEqual(tsok.query('GGTGGA', m=1, gu_pairing=True),
                {3: {1,2}})
        self.assertEqual(tsok.query('GGTTGG', m=1, gu_pairing=True),
                {})
        self.assertEqual(tsok.query('ACATAT', m=1, gu_pairing=True),
                {0: {3}, 2: {1}})
        self.assertEqual(tsok.query('AAATAT', m=1, gu_pairing=True),
                {})

        # 6 mismatches away should yield all results
        self.assertEqual(tsok.query('AAAAAA', m=6, gu_pairing=False),
                {0: {1,2,3,4,5}, 1: {1,2}, 2: {1}, 3: {1,2}})
        self.assertEqual(tsok.query('AAAAAA', m=6, gu_pairing=True),
                {0: {1,2,3,4,5}, 1: {1,2}, 2: {1}, 3: {1,2}})

    def test_with_full_signatures(self):
        tsok = self.build(kmer_shard.TrieSpaceOfKmersFullSig)
        self.query_tests(tsok)

    def test_with_split_signatures(self):
        tsok = self.build(kmer_shard.TrieSpaceOfKmersSplitSig)
        self.query_tests(tsok)


class TestTrieSpaceOfKmersOddK(unittest.TestCase):
    """Test TrieSpaceOfKmersFullSig and TrieSpaceOfKmersSplitSig using
    k-mers with odd length (i.e., k is odd).

    This is useful to split up because the methods treat even and odd lengths
    slightly differently.
    """

    # TODO make these k-mers be odd length

    def build(self, cls):
        # cls is the class to instantiate
        tsok = cls()
        tsok.add([
            ('ATCGAT', {(0,1), (0,2), (0,4), (1,1)}),
            ('ATAAAT', {(0,3), (2,1)})
        ])
        tsok.add([
            ('GGGGGG', {(3,1), (3,2)}),
            ('TTCGAT', {(0,5), (1,2)})
        ])
        return tsok

    def query_tests(self, tsok):
        # Exact matches, no G-U pairing
        self.assertEqual(tsok.query('ATCGAT', m=0, gu_pairing=False),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('ATAAAT', m=0, gu_pairing=False),
                {0: {3}, 2: {1}})
        self.assertEqual(tsok.query('GGGGGG', m=0, gu_pairing=False),
                {3: {1,2}})
        self.assertEqual(tsok.query('TTCGAT', m=0, gu_pairing=False),
                {0: {5}, 1: {2}})
        self.assertEqual(tsok.query('CCCCCC', m=0, gu_pairing=False),
                {})

        # Queries 1 mismatch away, no G-U pairing
        self.assertEqual(tsok.query('ATCGAT', m=1, gu_pairing=False),
                {0: {1,2,4,5}, 1: {1,2}})
        self.assertEqual(tsok.query('GGTGGG', m=1, gu_pairing=False),
                {3: {1,2}})
        self.assertEqual(tsok.query('GGTTGG', m=1, gu_pairing=False),
                {})

        # Exact matches, with G-U pairing
        self.assertEqual(tsok.query('ACCGAT', m=0, gu_pairing=True),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('ACCAAT', m=0, gu_pairing=True),
                {0: {1,2,4}, 1: {1}})
        self.assertEqual(tsok.query('AAAAAA', m=0, gu_pairing=True),
                {3: {1,2}})

        # Queries 1 mismatch away, with G-U pairing
        self.assertEqual(tsok.query('ACCGAT', m=1, gu_pairing=True),
                {0: {1,2,4,5}, 1: {1,2}})
        self.assertEqual(tsok.query('GGTGGA', m=1, gu_pairing=True),
                {3: {1,2}})
        self.assertEqual(tsok.query('GGTTGG', m=1, gu_pairing=True),
                {})
        self.assertEqual(tsok.query('ACATAT', m=1, gu_pairing=True),
                {0: {3}, 2: {1}})
        self.assertEqual(tsok.query('AAATAT', m=1, gu_pairing=True),
                {})

        # 6 mismatches away should yield all results
        self.assertEqual(tsok.query('AAAAAA', m=6, gu_pairing=False),
                {0: {1,2,3,4,5}, 1: {1,2}, 2: {1}, 3: {1,2}})
        self.assertEqual(tsok.query('AAAAAA', m=6, gu_pairing=True),
                {0: {1,2,3,4,5}, 1: {1,2}, 2: {1}, 3: {1,2}})

    def test_with_full_signatures(self):
        tsok = self.build(kmer_shard.TrieSpaceOfKmersFullSig)
        self.query_tests(tsok)

    def test_with_split_signatures(self):
        tsok = self.build(kmer_shard.TrieSpaceOfKmersSplitSig)
        self.query_tests(tsok)
