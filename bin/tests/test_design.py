"""Tests for design_naively.py
"""

import random
import os
import copy
import tempfile
import unittest
import logging

from collections import OrderedDict
from argparse import Namespace
from adapt import alignment
from adapt.prepare import align, ncbi_neighbors, prepare_alignment
from adapt.utils import seq_io
from bin import design

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'

SEQS = OrderedDict()
# Default args: window size 3, guide size 2, allow GU pairing
# GU pairing allows AA to match GG in 1st window
SEQS["genome_1"] = "AACTA"
SEQS["genome_2"] = "AAACT"
SEQS["genome_3"] = "GGCTA"
SEQS["genome_4"] = "GGCTT"

# Specificity seq stops AA from being the best guide in the 1st window
SP_SEQS = OrderedDict()
SP_SEQS["genome_5"] = "AA---"


class TestDesign(object):
    class TestDesignCase(unittest.TestCase):
        def setUp(self):
            # Disable logging
            logging.disable(logging.INFO)
            
            # Create a temporary output file 
            self.output_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            self.output_file.close()

        def check_results(self, file, expected, header='target-sequences'):
            col_loc = None
            with open(file) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        headers = line.split('\t')
                        # Will raise an error if header is not in output
                        col_loc = headers.index(header)
                        continue
                    self.assertLess(i, len(expected) + 1)
                    guide_line = line.split('\t')[-2]
                    guides = guide_line.split(' ')
                    for guide in guides: 
                        self.assertIn(guide, expected[i-1])
                    self.assertEqual(len(guides), len(expected[i-1]))
                self.assertEqual(i, len(expected))

        @staticmethod
        def baseArgv(search_type='sliding-window', input_type='fasta', 
                     objective='minimize-guides', model=False, specific=None, 
                     input_file=None, output_loc=None, specificity_file=None):

            argv = ['design.py', search_type, input_type]

            if input_type == 'fasta':
                argv.extend([input_file, '-o', output_loc])
            elif input_type == 'auto-from-args':
                argv.extend(['64320', 'None', 'NC_035889', output_loc])
            elif input_type == 'auto-from-file':
                argv.extend([input_file, output_loc])

            if input_type in ['auto-from-args', 'auto-from-file']:
                argv.extend(['--sample-seqs', '1', '--mafft-path', 'fake_path'])

            if search_type == 'sliding-window':
                argv.extend(['-w', '3'])
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

            argv.extend(['--obj', objective, '--seed', '0', '-gl', '2'])

            return argv

        def tearDown(self):
            for file in self.files:
                if os.path.isfile(file):
                    os.unlink(file)
            # Re-enable logging
            logging.disable(logging.NOTSET)


class TestDesignFasta(TestDesign.TestDesignCase):
    """Test design.py given an input FASTA
    """

    def setUp(self):
        super().setUp()

        # Create a temporary fasta file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()

        seq_io.write_fasta(SEQS, self.fasta.name)

        # Create a temporary fasta file for specificity
        self.sp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.sp_fasta.close()

        seq_io.write_fasta(SP_SEQS, self.sp_fasta.name)

        self.files = [self.fasta.name, self.sp_fasta.name, self.output_file.name]

    def test_min_guides(self):
        # Base args set the percentage of sequences to match at 75%
        argv = super().baseArgv(input_file=self.fasta.name, output_loc=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AA"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)

    def test_max_activity(self):
        # Doesn't use model, just greedy binary prediction with 0 mismatches
        # (so same outputs as min-guides)
        argv = super().baseArgv(objective='maximize-activity', input_file=self.fasta.name, 
                        output_loc=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AA"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)

    def test_complete_targets(self):
        # Since sequences are short and need 1 base for primer on each side, 
        # only finds 1 target in middle
        argv = super().baseArgv(search_type='complete-targets', input_file=self.fasta.name, 
                        output_loc=self.output_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["CT"]]
        self.check_results(self.output_file.name, expected, 
                           header='guide-target-sequences')

    def test_specificity_fastas(self):
        # AA isn't allowed in 1st window by specificity fasta, 
        # so 1st window changes
        argv = super().baseArgv(input_file=self.fasta.name, output_loc=self.output_file.name,
                        specific='fasta', specificity_file=self.sp_fasta.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AC", "GG"], ["CT"], ["CT"]]
        self.check_results(self.output_file.name, expected)


class TestDesignAutos(TestDesign.TestDesignCase):
    """Test design.py given arguments to automatically download FASTAs

    Does not run the entire design.py; prematurely stops by giving a fake path
    to MAFFT. Expected to return a FileNotFoundError
    """
    def setUp(self):
        super().setUp()
        
        # Disable logging
        logging.disable(logging.INFO)

        self.files = []
        # Create a temporary input file
        self.input_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.input_file.write("Zika virus\t64320\tNone\tNC_035889\n")
        # Closes the file so that it can be reopened on Windows
        self.input_file.close()

        # Create a temporary output directory 
        self.output_dir = tempfile.TemporaryDirectory()

        self.files = [self.input_file.name, self.output_file.name]

    def test_auto_from_file(self):
        argv = super().baseArgv(input_type='auto-from-file', 
                        input_file=self.input_file.name, 
                        output_loc=self.output_dir.name)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_auto_from_args(self):
        argv = super().baseArgv(input_type='auto-from-args', 
                        output_loc='')
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_specificity_taxa(self):
        argv = super().baseArgv(input_type='auto-from-args', output_loc='',
                        specific='taxa', specificity_file='')
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def tearDown(self):
        super().tearDown()
        self.output_dir.cleanup()


class TestDesignFull(TestDesign.TestDesignCase):
    """Test design.py fully through
    """
    def setUp(self):
        super().setUp()

        self.files = []
        # Create a temporary input file
        self.input_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.input_file.write("Zika virus\t64320\tNone\tNC_035889\n")
        # Closes the file so that it can be reopened on Windows
        self.input_file.close()

        # Create a temporary input file
        self.sp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.sp_file.write("123\tNone\n")
        # Closes the file so that it can be reopened on Windows
        self.sp_file.close()

        # Create a temporary output directory 
        self.output_dir = tempfile.TemporaryDirectory()

        self.real_output_file = self.output_file.name + '.0'
        self.files = [self.input_file.name, self.sp_file.name, self.output_file.name, \
                      self.real_output_file]

        self.set_mafft_exec = align.set_mafft_exec
        align.set_mafft_exec = lambda mafft_path: None

        self.curate_against_ref = align.curate_against_ref

        def small_curate(seqs, ref_accs, asm=None, remove_ref_accs=[]):
            return {seq: seqs[seq] for seq in seqs if seq.split('.')[0] != 'NC_035889'}

        align.curate_against_ref = small_curate

        self.align = align.align
        align.align = lambda seqs, am=None: SEQS

        self.fetch_sequences_for_taxonomy = prepare_alignment.fetch_sequences_for_taxonomy

        def small_fetch(taxid, segment):
            if taxid == 123:
                return SP_SEQS
            else:
                return self.fetch_sequences_for_taxonomy(taxid, segment)

        prepare_alignment.fetch_sequences_for_taxonomy = small_fetch

    def test_specificity_taxa(self):
        argv = super().baseArgv(input_type='auto-from-args', output_loc=self.output_file.name,
                        specific='taxa', specificity_file=self.sp_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [["AC", "GG"], ["CT"], ["CT"]]
        self.check_results(self.real_output_file, expected)

    def tearDown(self):
        align.set_mafft_exec = self.set_mafft_exec
        align.curate_against_ref = self.curate_against_ref
        align.align = self.align
        prepare_alignment.fetch_sequences_for_taxonomy = self.fetch_sequences_for_taxonomy
        super().tearDown()
        self.output_dir.cleanup()
