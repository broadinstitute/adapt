"""Tests for index_compress module.
"""

import unittest

from adapt.utils import index_compress

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestIndexCompress(unittest.TestCase):
    """Tests functions for compressing/de-compressing.
    """

    def run_compress_and_decompress(self, idx):
        compressed = index_compress.compress_mostly_contiguous(idx)
        decompressed = index_compress.decompress_ranges(compressed)
        self.assertEqual(decompressed, idx)

    def test_contiguous(self):
        self.run_compress_and_decompress({0, 1, 2, 3})
        self.run_compress_and_decompress({1, 2, 3})

    def test_uncontinguous(self):
        self.run_compress_and_decompress({0, 1, 2, 4, 5})
        self.run_compress_and_decompress({0, 1, 2, 4, 5, 7})
        self.run_compress_and_decompress({0, 2, 4, 5, 7, 8, 10})
        self.run_compress_and_decompress({0, 2, 4, 5, 7, 8, 10, 12})

    def test_duplicate(self):
        self.run_compress_and_decompress({0, 1, 2, 2, 3})
        self.run_compress_and_decompress({0, 1, 2, 2, 3, 5, 5})

