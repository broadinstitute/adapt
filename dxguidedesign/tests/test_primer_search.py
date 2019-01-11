"""Tests for primer_search module.
"""

import logging
import random
import unittest

from dxguidedesign import alignment
from dxguidedesign import primer_search
from dxguidedesign.utils import guide

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestPrimerSearch(unittest.TestCase):
    """Tests methods in the PrimerSearch class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

        # For these tests, do not allow G-U pairing
        guide.set_allow_gu_pairs_to_no()

        self.a_seqs = ['ATCGAA',
                       'ATCGAT',
                       'TTCGAA',
                       'CTCGAT',
                       'GGCGAA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        self.a = primer_search.PrimerSearcher(self.a_aln, 4, 0, 1.0, (1, 1, 100))

    def test_find_primers_simple(self):
        covers = list(self.a.find_primers())
        expected = [(0, 4, 1.0, set(['ATCG', 'TTCG', 'CTCG', 'GGCG'])),
                    (1, 2, 1.0, set(['TCGA', 'GCGA'])),
                    (2, 2, 1.0, set(['CGAA', 'CGAT']))]
        self.assertEqual(covers, expected)

    def test_find_primers_with_max(self):
        covers = list(self.a.find_primers(max_at_site=2))
        expected = [(1, 2, 1.0, set(['TCGA', 'GCGA'])),
                    (2, 2, 1.0, set(['CGAA', 'CGAT']))]
        self.assertEqual(covers, expected)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
        
        # Return G-U pairing setting to default
        guide.set_allow_gu_pairs_to_default()
