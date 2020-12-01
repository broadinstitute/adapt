"""Tests for design_naively.py
"""

import random
import os
import copy
import tempfile
import unittest

from collections import OrderedDict
from argparse import Namespace
from adapt import alignment
from adapt.utils import seq_io
from bin import design

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'


class TestDesignFasta(unittest.TestCase):
    """Test design.py given an input FASTA
    """

    def setUp(self):
        # Create a temporary fasta file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()

        seqs = OrderedDict()
        seqs["genome_1"] = "AACTA"
        seqs["genome_2"] = "AAACT"
        seqs["genome_3"] = "GGCTA"
        seqs["genome_4"] = "GGCTT"

        seq_io.write_fasta(seqs, self.fasta.name)

        # Create a temporary fasta file for specificity
        self.sp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.sp_fasta.close()

        sp_seqs = OrderedDict()
        sp_seqs["genome_5"] = "AA---"

        seq_io.write_fasta(sp_seqs, self.sp_fasta.name)

        # Create a temporary output file 
        self.output_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.output_file.close()

        self.files = [self.fasta.name, self.sp_fasta.name, self.output_file.name]

    def check_results(self, file, expected):
        with open(file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                self.assertLess(i, len(expected) + 1)
                guide_line = line.split('\t')[-2]
                guides = guide_line.split(' ')
                for guide in guides: 
                    self.assertIn(guide, expected[i-1])
                self.assertEqual(len(guides), len(expected[i-1]))
            self.assertEqual(i, len(expected))

    def test_min_guides(self):
        argv = baseArgv(input_file=self.fasta.name, output_file=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AA"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)

    def test_max_activity(self):
        argv = baseArgv(objective='maximize-activity', input_file=self.fasta.name, 
                        output_file=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AA"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)

    def test_complete_targets(self):
        argv = baseArgv(search_type='complete-targets', input_file=self.fasta.name, 
                        output_file=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["CT"]]
        self.check_results(self.output_file.name, expected)

    def test_specific_fastas(self):
        argv = baseArgv(input_file=self.fasta.name, output_file=self.output_file.name,
                        specific='fasta', specificity_file=self.sp_fasta.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AC", "GG"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)

    def tearDown(self):
        for file in self.files:
            if os.path.isfile(file):
                os.unlink(file)


class TestDesignAutos(unittest.TestCase):
    """Test design.py given arguments to automatically download FASTAs
    """
    def setUp(self):
        self.files = []
        # Create a temporary input file
        self.input_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.input_file.write("Zika virus\t64320\tNone\tNC_035889\n")
        # Closes the file so that it can be reopened on Windows
        self.input_file.close()

        # Create a temporary input file
        self.sp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.sp_file.write("11083\tNone\n")
        # Closes the file so that it can be reopened on Windows
        self.sp_file.close()

        # Create a temporary output file 
        self.output_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.output_file.close()

        self.files = [self.input_file.name, self.sp_file.name, self.output_file.name]

    def test_auto_from_file(self):
        argv = baseArgv(input_type='auto-from-file', 
                        input_file=self.input_file.name, 
                        output_file=self.output_file.name)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_auto_from_args(self):
        argv = baseArgv(input_type='auto-from-args', 
                        output_file=self.output_file.name)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_specific_taxa(self):
        argv = baseArgv(input_type='auto-from-args', output_file=self.output_file.name,
                        specific='taxa', specificity_file=self.sp_file.name)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def tearDown(self):
        for file in self.files:
            if os.path.isfile(file):
                os.unlink(file)


def baseArgv(search_type='sliding-window', input_type='fasta', 
             objective='minimize-guides', model=False, specific=None, 
             input_file=None, output_file=None, specificity_file=None):

    argv = ['design.py', search_type, input_type]

    if input_type == 'fasta':
        argv.extend([input_file, '-o', output_file, '-gl', '2'])
    elif input_type == 'auto-from-args':
        argv.extend(['64320', 'None', 'NC_035889', output_file])
    elif input_type == 'auto-from-file':
        argv.extend([input_file, '.'])

    if input_type in ['auto-from-args', 'auto-from-file']:
        argv.extend(['--sample-seqs', '1', '--mafft-path', 'fake_path'])

    if search_type == 'sliding-window':
        argv.extend(['-w', '3', '--quiet-analysis'])
    if search_type == 'complete-targets':
        argv.extend(['--best-n-targets', '2', '-pp', '.75', '-pl', '1', 
                     '--max-primers-at-site', '2'])

    if objective == 'minimize-guides':
        argv.extend(['-gm', '0', '-gp', '.75'])
    elif objective =='maximize-activity':
        argv.extend(['--maximization-algorithm', 'greedy'])

    if specific == 'fasta':
        argv.extend(['--specific-against-fastas', specificity_file, '--id-m', '0'])
    elif specific == 'taxa':
        argv.extend(['--specific-against-taxa', specificity_file, '--id-m', '0'])

    if model:
        argv.extend(['--predict-activity-model-path', 'models/classify/model-51373185', 
                     'models/regress/model-f8b6fd5d'])
    elif objective =='maximize-activity':
        argv.extend(['--use-simple-binary-activity-prediction', '-gm', '0'])

    argv.extend(['--obj', objective, '--seed', '294'])

    return argv
