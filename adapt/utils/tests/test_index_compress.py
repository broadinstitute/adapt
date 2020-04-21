"""Tests for index_compress module.
"""

import unittest

from adapt.utils import index_compress

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestIndexCompress(unittest.TestCase):
    """Tests functions for compressing/de-compressing integers.
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


class TestIndexCompressForDicts(unittest.TestCase):
    """Tests functions for compressing/de-compressing dicts.
    """

    def run_compress_and_decompress(self, d):
        compressed = index_compress.compress_mostly_contiguous_keys_with_identical_vals(d)
        decompressed = index_compress.decompress_ranges_for_dict(compressed)
        self.assertEqual(decompressed, d)

    def test_contiguous(self):
        self.run_compress_and_decompress({0:300, 1:100, 2:100, 3:200})
        self.run_compress_and_decompress({1:100, 2:200, 3:200})
        self.run_compress_and_decompress({1:100, 2:200, 3:100})

    def test_uncontinguous(self):
        self.run_compress_and_decompress({0:100, 1:100, 2:100, 4:100, 5:100})
        self.run_compress_and_decompress({0:200, 1:100, 2:200, 4:200, 5:200,
            7:200})
        self.run_compress_and_decompress({0:200, 2:200, 4:300, 5:300, 7:400,
            8:500, 10:500})
        self.run_compress_and_decompress({0:200, 2:300, 4:400, 5:500, 7:700,
            8:800, 10:1000, 12:1000})

