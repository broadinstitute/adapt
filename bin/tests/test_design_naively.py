"""Tests for design_naively.py
"""

import random
import logging
import copy
import unittest

from argparse import Namespace
from adapt import alignment
from bin import design_naively

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'


class TestDesignNaively(unittest.TestCase):
    """Test design_naively.py
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

    def test_construct_guide_naively_at_each_pos_simple(self):
        ref_seq = 'ACAAA'
        seqs = [ref_seq,
                'AAAAA',
                'AAAAA',
                'AAAAA',
                'GGATA',
                'GGATA',
                'GGTAA',
                'GGTAA']

        def test_seqs(seqs, args):
            aln = alignment.Alignment.from_list_of_seqs(seqs)
            guides = design_naively.construct_guide_naively_at_each_pos(aln, 
                    args, ref_seq=ref_seq)
            guides = [list(guide.values())[0] for guide in guides]
            return guides

        args = baseArgs()
        args.consensus = True
        self.assertEqual(test_seqs(seqs, args), [('AG', 0.0), ('GA', 0.25), ('AA', 0.5), ('AA', 0.75)])

        args.consensus = False
        args.mode = True
        self.assertEqual(test_seqs(seqs, args), [('GG', 0.5), ('AA', 0.375), ('AA', 0.5), ('AA', 0.75)])

        args.mode = False
        args.diversity = 'entropy'
        test_guides = test_seqs(seqs, args)
        test_guides = [guide[0] for guide in test_guides]
        self.assertEqual(test_guides, [('AC', 0.125), ('CA', 0.125), ('AA', 0.5), ('AA', 0.75)])

    def test_construct_guide_naively_at_each_pos_flanking(self):
        ref_seq = 'ACAAA'
        seqs = [ref_seq,
                'AAAAA',
                'AAAAA',
                'AAAAA',
                'GGATA',
                'GGATA',
                'GGTAA',
                'GGTAA']

        def test_seqs(seqs, args):
            aln = alignment.Alignment.from_list_of_seqs(seqs)
            guides = design_naively.construct_guide_naively_at_each_pos(aln, 
                    args, ref_seq=ref_seq)
            guides = [list(guide.values())[0] for guide in guides]
            return guides

        args = baseArgs()
        args.required_flanking_seqs = ('B', 'V')
        args.consensus = True
        self.assertEqual(test_seqs(seqs, args), [('None', 0), ('GT', 0.25), ('AA', 0.125), ('None', 0)])

        args.consensus = False
        args.mode = True
        self.assertEqual(test_seqs(seqs, args), [('None', 0), ('GT', 0.25), ('AT', 0.25), ('None', 0)])

        args.mode = False
        args.diversity = 'entropy'
        test_guides = test_seqs(seqs, args)
        test_guides = [guide[0] for guide in test_guides]
        self.assertEqual(test_guides, [('None', 0), ('None', 0), ('AA', 0.125), ('None', 0)])

    def test_find_guide_in_each_window_simple(self):
        guides = [('TT', 50),
                  ('TG', 30),
                  ('GG', 10),
                  ('GC', 40),
                  ('CC', 60),
                  ('CA', 70),
                  ('AA', 20)]

        args = baseArgs()
        test_max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        self.assertEqual(test_max_windows, [[('CC', 60), ('TT', 50), ('GC', 40)], 
                                            [('CA', 70), ('CC', 60), ('GC', 40)], 
                                            [('CA', 70), ('CC', 60), ('GC', 40)]])
        test_min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')
        self.assertEqual(test_min_windows, [[('GG', 10), ('TG', 30), ('GC', 40)], 
                                            [('GG', 10), ('TG', 30), ('GC', 40)], 
                                            [('GG', 10), ('AA', 20), ('GC', 40)]])

    def test_find_guide_in_each_window_ties(self):
        guides = [('TT', 10),
                  ('TG', 30),
                  ('GG', 10),
                  ('GC', 10),
                  ('CC', 10),
                  ('CA', 30),
                  ('AA', 30)]

        args = baseArgs()
        test_max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        self.assertEqual(test_max_windows, [[('TG', 30), ('TT', 10), ('GG', 10)], 
                                            [('TG', 30), ('CA', 30), ('GG', 10)], 
                                            [('CA', 30), ('AA', 30), ('GG', 10)]])
        test_min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')
        self.assertEqual(test_min_windows, [[('TT', 10), ('GG', 10), ('GC', 10)], 
                                            [('GG', 10), ('GC', 10), ('CC', 10)], 
                                            [('GG', 10), ('GC', 10), ('CC', 10)]])

    def test_find_guide_in_each_window_negative(self):
        guides = [('TT', 30),
                  ('TG', 0),
                  ('GG', -20),
                  ('GC', 10),
                  ('CC', 20),
                  ('CA', -30),
                  ('AA', -10)]

        args = baseArgs()
        test_max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        self.assertEqual(test_max_windows, [[('TT', 30), ('CC', 20), ('GC', 10)], 
                                            [('CC', 20), ('GC', 10), ('TG', 0)], 
                                            [('CC', 20), ('GC', 10), ('AA', -10)]])
        test_min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')
        self.assertEqual(test_min_windows, [[('GG', -20), ('TG', 0), ('GC', 10)], 
                                            [('CA', -30), ('GG', -20), ('TG', 0)], 
                                            [('CA', -30), ('GG', -20), ('AA', -10)]])

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)


def baseArgs():
    args = Namespace()

    args.window_size = 6
    args.guide_length = 2
    args.skip_gaps = 0.5
    args.consensus = False
    args.mode = False
    args.diversity = None
    args.guide_mismatches = 0
    args.allow_gu_pairs = False
    args.best_n = 3
    args.required_flanking_seqs = (None, None)

    return args

