"""Tests for primer_search module.
"""

import logging
import random
import unittest

from dxguidedesign import alignment
from dxguidedesign import primer_search
from dxguidedesign.utils import guide

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestPrimerResult(unittest.TestCase):
    """Tests methods in the PrimerResult class.
    """

    def test_does_overlap(self):
        a = primer_search.PrimerResult(10, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(a))

        b = primer_search.PrimerResult(12, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

    def test_does_not_overlap(self):
        a = primer_search.PrimerResult(10, 1, 5, 1.0, {'AAAAA'})
        b = primer_search.PrimerResult(15, 1, 5, 1.0, {'AAAAA'})
        c = primer_search.PrimerResult(16, 1, 5, 1.0, {'AAAAA'})
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))
        self.assertFalse(a.overlaps(c))
        self.assertFalse(c.overlaps(a))

    def test_does_overlap_with_expand(self):
        a = primer_search.PrimerResult(10, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(a, expand=5))

        b = primer_search.PrimerResult(12, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(b, expand=5))
        self.assertTrue(b.overlaps(a, expand=5))

        b = primer_search.PrimerResult(17, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(b, expand=5))
        self.assertTrue(b.overlaps(a, expand=5))

        c = primer_search.PrimerResult(2, 1, 5, 1.0, {'AAAAA'})
        self.assertTrue(a.overlaps(c, expand=5))
        self.assertTrue(c.overlaps(a, expand=5))

    def test_does_not_overlap_with_expand(self):
        a = primer_search.PrimerResult(10, 1, 5, 1.0, {'AAAAA'})
        b = primer_search.PrimerResult(20, 1, 5, 1.0, {'AAAAA'})
        c = primer_search.PrimerResult(0, 1, 5, 1.0, {'AAAAA'})
        self.assertFalse(a.overlaps(b, expand=5))
        self.assertFalse(b.overlaps(a, expand=5))
        self.assertFalse(a.overlaps(c, expand=5))
        self.assertFalse(c.overlaps(a, expand=5))


class TestPrimerSearch(unittest.TestCase):
    """Tests methods in the PrimerSearch class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAA',
                       'ATCGAT',
                       'TTCGAA',
                       'CTCGAT',
                       'GGCGAA']
        self.a_aln = alignment.Alignment.from_list_of_seqs(self.a_seqs)
        self.a = primer_search.PrimerSearcher(self.a_aln, 4, 0, 1.0, (1, 1, 100))

    def _pr(self, start, num_primers, frac_bound, primers_in_cover):
        # Construct a PrimerResult object
        return primer_search.PrimerResult(start, num_primers, 4, frac_bound,
            primers_in_cover)

    def test_find_primers_simple(self):
        covers = list(self.a.find_primers())
        expected = [self._pr(0, 4, 1.0, set(['ATCG', 'TTCG', 'CTCG', 'GGCG'])),
                    self._pr(1, 2, 1.0, set(['TCGA', 'GCGA'])),
                    self._pr(2, 2, 1.0, set(['CGAA', 'CGAT']))]
        self.assertEqual(covers, expected)

    def test_find_primers_with_max(self):
        covers = list(self.a.find_primers(max_at_site=2))
        expected = [self._pr(1, 2, 1.0, set(['TCGA', 'GCGA'])),
                    self._pr(2, 2, 1.0, set(['CGAA', 'CGAT']))]
        self.assertEqual(covers, expected)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
