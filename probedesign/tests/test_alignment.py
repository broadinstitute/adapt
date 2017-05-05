"""Tests for alignment module.
"""

import unittest

from probedesign import alignment

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignment(unittest.TestCase):
    """Tests methods in the Alignment class.
    """

    def setUp(self):
        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a = alignment.Alignment.from_list_of_seqs(self.a_seqs)

        self.b_seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.b = alignment.Alignment.from_list_of_seqs(self.b_seqs)

        self.c_seqs = ['ATCGAA', 'ATC-AA']
        self.c = alignment.Alignment.from_list_of_seqs(self.c_seqs)

    def test_make_list_of_seqs_a(self):
        self.assertEqual(self.a.make_list_of_seqs(),
                         self.a_seqs)
        self.assertEqual(self.a.make_list_of_seqs([0]),
                         ['ATCGAA'])
        self.assertEqual(self.a.make_list_of_seqs([0,3]),
                         ['ATCGAA', 'AYCGAT'])

    def test_make_list_of_seqs_b(self):
        self.assertEqual(self.b.make_list_of_seqs(),
                         self.b_seqs)

    def test_make_list_of_seqs_c(self):
        self.assertEqual(self.c.make_list_of_seqs(),
                         self.c_seqs)

    def test_determine_consensus_sequence_a(self):
        self.assertEqual(self.a.determine_consensus_sequence(), 'ATCGAA')
        self.assertEqual(self.a.determine_consensus_sequence([0]), 'ATCGAA')
        self.assertIn(self.a.determine_consensus_sequence([2]),
                      ['ATCGAA', 'ACCGAA'])
        self.assertIn(self.a.determine_consensus_sequence([0,1]),
                      ['ATCGAA', 'ATCGAT'])
        self.assertIn(self.a.determine_consensus_sequence([1,2]),
                      ['ATCGAT', 'ATCGAA'])

    def test_determine_consensus_sequence_b(self):
        self.assertEqual(self.b.determine_consensus_sequence(), 'ATCGAT')

    def test_determine_consensus_sequence_c(self):
        with self.assertRaises(ValueError):
            # Should fail when determining consensus sequence given an indel
            self.c.determine_consensus_sequence()
        self.assertIn(self.c.determine_consensus_sequence([0]), 'ATCGAA')

    def test_has_indel(self):
        self.assertFalse(self.a.has_indel())
        self.assertFalse(self.b.has_indel())
        self.assertTrue(self.c.has_indel())

    def test_construct_probe_a(self):
        self.assertEqual(self.a.construct_probe(0, 4, [0,1,2,3,4], 0),
                         ('ATCG', [0,1,2,3]))
        self.assertEqual(self.a.construct_probe(0, 4, [0,1,2,3,4], 1),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_probe(0, 4, [4], 1),
                         ('AGCG', [4]))
        self.assertIn(self.a.construct_probe(0, 4, [2,3], 0),
                      [('ATCG', [2,3]), ('ACCG', [2,3])])
        self.assertEqual(self.a.construct_probe(1, 4, [0,1,2,3,4], 0),
                         ('TCGA', [0,1,2,3]))
        self.assertEqual(self.a.construct_probe(2, 4, [0,1,2,3,4], 0),
                         ('CGAA', [0,2,4]))
        self.assertEqual(self.a.construct_probe(2, 4, [0,1,2,3,4], 1),
                         ('CGAA', [0,1,2,3,4]))
        self.assertEqual(self.a.construct_probe(2, 4, [0,1,2,3,4], 2),
                         ('CGAA', [0,1,2,3,4]))
        self.assertIn(self.a.construct_probe(2, 4, [0,1,2,3], 0),
                      [('CGAA', [0,2]), ('CGAT', [1,3])])
        self.assertIn(self.a.construct_probe(2, 4, [0,1,2,3], 1),
                      [('CGAA', [0,1,2,3]), ('CGAT', [0,1,2,3])])

    def test_construct_probe_b(self):
        self.assertEqual(self.b.construct_probe(0, 4, [0,1,2,3,4], 0),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_probe(0, 4, [0,1,2,3,4], 1),
                         ('ATCG', [0,2]))
        self.assertEqual(self.b.construct_probe(0, 4, [0,1,2,3,4], 2),
                         ('ATCG', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_probe(2, 4, [0,1,2,3,4], 0),
                         ('CGAA', [0]))
        self.assertEqual(self.b.construct_probe(2, 4, [0,1,2,3,4], 1),
                         ('CGAT', [0]))
        self.assertEqual(self.b.construct_probe(2, 4, [0,1,2,3,4], 2),
                         ('CGAT', [0,1,2,3]))
        self.assertEqual(self.b.construct_probe(2, 4, [0,1,2,3,4], 3),
                         ('CGAT', [0,1,2,3,4]))
        self.assertEqual(self.b.construct_probe(2, 4, [2,4], 1),
                         ('CGAC', []))
        self.assertEqual(self.b.construct_probe(2, 4, [2,4], 2),
                         ('CGAC', [2,4]))
        self.assertIn(self.b.construct_probe(2, 4, [2,3,4], 2),
                      [('CGAC', [2,4]), ('CGAT', [2,3])])
        with self.assertRaises(alignment.CannotConstructProbeError):
            # Should fail when 'N' is all that exists at a position
            self.b.construct_probe(0, 4, [1,3,4], 0)

    def test_construct_probe_c(self):
        with self.assertRaises(alignment.CannotConstructProbeError):
            # Should fail when alignment has an indel
            self.c.construct_probe(0, 4, [0,1], 0)

    def test_sequences_bound_by_probe(self):
        self.assertEqual(self.a.sequences_bound_by_probe('ATCG', 0, 0),
                         [0,1,2,3])
        self.assertEqual(self.a.sequences_bound_by_probe('ATCG', 0, 1),
                         [0,1,2,3,4])
