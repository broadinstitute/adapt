"""Tests for formatting module.
"""

import logging
import unittest

from adapt.utils import formatting

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestTagSeqOverlap(unittest.TestCase):
    """Tests the tag_seq_overlap() function.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

    def t(self, overlap_labels, expect, target_seq='AATTCCATCGG'):
        f = formatting.tag_seq_overlap(overlap_labels, target_seq)
        self.assertEqual(f, expect)

    def test_disjoint(self):
        self.t({'pr': {6,7,8}}, 'AATTCC<pr>ATC</pr>GG')
        self.t({'pr': {1,2,3,6,7,8}}, 'A<pr>ATT</pr>CC<pr>ATC</pr>GG')
        self.t({'g': {1,2,3}, 'pr': {6,7,8}}, 'A<g>ATT</g>CC<pr>ATC</pr>GG')

    def test_adjacent(self):
        self.t({'pr': {1,2,3}, 'g': {4,5,6}}, 'A<pr>ATT</pr><g>CCA</g>TCGG')
        self.t({'pr': {1,2,3,7,8,9}, 'g': {4,5,6}},
                'A<pr>ATT</pr><g>CCA</g><pr>TCG</pr>G')

    def test_overlapping_different_labels(self):
        f = formatting.tag_seq_overlap({'pr': {1,2,3}, 'g': {3,4,5}},
                'AATTCCATCGG')
        self.assertIn(f, ['A<pr>ATT</pr><g>CC</g>ATCGG',
            'A<pr>AT</pr><g>TCC</g>ATCGG'])

    def test_at_ends(self):
        self.t({'pr': {0,1,2,8,9,10}}, '<pr>AAT</pr>TCCAT<pr>CGG</pr>')
        self.t({'pr': {0,1,2,8,9,10}, 'g': {4,5,6}},
                '<pr>AAT</pr>T<g>CCA</g>T<pr>CGG</pr>')

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
