"""Tests for kmer_leaf module.
"""

import unittest

from adapt.specificity import kmer_leaf

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestKmerLeaf(unittest.TestCase):
    """Tests KmerLeaf objects.
    """
    
    def test_create(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        kl2 = kmer_leaf.KmerLeaf([(0,1),(0,2),(3,5)])
        self.assertEqual(len(kl), 3)
        self.assertTrue(kl == kl2)

    def test_extend(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        kl2 = kmer_leaf.KmerLeaf([(0,1),(3,6),(4,7)])
        self.assertEqual(len(kl), 3)
        kl.extend(kl2)
        self.assertEqual(len(kl), 5)
        self.assertEqual(len(kl2), 3)

    def test_contains(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        self.assertTrue(0 in kl)
        self.assertTrue(3 in kl)
        self.assertFalse(1 in kl)
    
    def test_union(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        kl2 = kmer_leaf.KmerLeaf([(0,1),(3,6),(4,7)])
        self.assertEqual(len(kl), 3)
        self.assertEqual(len(kl2), 3)
        kl_union = kmer_leaf.KmerLeaf.union([kl, kl2])
        self.assertEqual(len(kl), 3)
        self.assertEqual(len(kl2), 3)
        self.assertEqual(len(kl_union), 5)

    def test_remove(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        self.assertEqual(len(kl), 3)
        kl.remove(0)
        self.assertEqual(len(kl), 1)


    def test_copy(self):
        kl = kmer_leaf.KmerLeaf([(0,1),(0,2),(0,2),(3,5)])
        kl_copy = kl.copy()
        self.assertTrue(kl == kl_copy)
        kl.remove(0)
        self.assertEqual(len(kl), 1)
        self.assertEqual(len(kl_copy), 3)

