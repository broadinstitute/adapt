"""Tests for design_naively.py
"""

import random
import copy
import unittest

from argparse import Namespace
from adapt import alignment
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
import os.path
import types

file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "design_naively.py"))

loader = SourceFileLoader("design_naively", file_path)
spec = spec_from_loader(loader.name, loader)
design_naively = module_from_spec(spec)
loader.exec_module(design_naively)

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'


class TestDesignNaively(unittest.TestCase):
    """Test design_naively.py
    """

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
                  ('GT', 30),
                  ('GG', 10),
                  ('CG', 40),
                  ('CC', 60),
                  ('CA', 70),
                  ('AA', 20)]

        args = baseArgs()
        max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')

    def test_find_guide_in_each_window_ties(self):
        guides = [('TT', 30),
                  ('GT', 30),
                  ('GG', 10),
                  ('CG', 10),
                  ('CC', 10),
                  ('CA', 10),
                  ('AA', 30)]

        args = baseArgs()
        max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')

    def test_find_guide_in_each_window_negative(self):
        guides = [('TT', 30),
                  ('GT', 0),
                  ('GG', -20),
                  ('CG', 10),
                  ('CC', 20),
                  ('CA', -30),
                  ('AA', -10)]

        args = baseArgs()
        max_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args)
        min_windows = design_naively.find_guide_in_each_window(guides, len(guides) + 1, args, obj_type='min')


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

