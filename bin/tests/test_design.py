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
    """Test design.py
    """

    def setUp(self):
        # Create a temporary fasta file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()

        self.seqs = OrderedDict()
        # self.seqs["genome_1"] = "AACCA"
        # self.seqs["genome_2"] = "AAATA"
        # self.seqs["genome_3"] = "GGATA"
        # self.seqs["genome_4"] = "GGGAA"
        self.seqs["genome_1"] = "AACTA-"
        self.seqs["genome_2"] = "AAACT-"
        self.seqs["genome_3"] = "GGCTA-"
        self.seqs["genome_4"] = "GGCTT-"

        seq_io.write_fasta(self.seqs, self.fasta.name)

        # Create a temporary output file 
        self.output = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.output.close()

        self.files = [self.fasta.name, self.output.name]

    def check_results(self, file, expected):
        with open(file) as f:
            for i, line in enumerate(f):
                self.assertLess(i, len(expected))
                self.assertEqual(line, expected[i])
            self.assertEqual(i, len(expected)-1)

    def test_min_guides(self):
        argv = baseArgv(input_file=self.fasta.name, output_file=self.output.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = ["window-start\twindow-end\tcount\tscore\ttotal-frac-bound"
                    "\ttarget-sequences\ttarget-sequence-positions\n",
                    "0\t3\t1\t1.0\t1.0\tAA\t{0}\n",
                    "1\t4\t1\t1.0\t0.75\tCT\t{2}\n",
                    "2\t5\t1\t1.0\t0.75\tCT\t{2}\n"]
        self.check_results(self.output.name, expected)

    def test_max_activity(self):
        argv = baseArgv(objective='maximize-activity', input_file=self.fasta.name, 
                        output_file=self.output.name)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = ["window-start\twindow-end\tcount\tobjective-value\ttotal-frac-bound"
                    "\tguide-set-expected-activity\tguide-set-median-activity"
                    "\tguide-set-5th-pctile-activity\tguide-expected-activities"
                    "\ttarget-sequences\ttarget-sequence-positions\n",
                    "0\t3\t1\t1.0\t1.0\t1.0\t1.0\t1.0\t1.0\tAA\t{0}\n",
                    "1\t4\t1\t1.0\t1.0\t1.0\t1.0\t1.0\t1.0\tAC\t{2}\n",
                    "2\t5\t1\t1.0\t1.0\t1.0\t1.0\t1.0\t1.0\tCT\t{3}\n"]
        self.check_results(self.output.name, expected)

    def test_complete_targets(self):
        argv = baseArgv(search_type='complete-targets', input_file=self.fasta.name, 
                        output_file=self.output.name)
        args = design.argv_to_args(argv)
        print(argv)
        design.run(args)
        with open(self.output.name) as f:
            for line in f:
                print(line)
        #self.check_results(self.output.name, expected)

    def tearDown(self):
        for file in self.files:
            if os.path.isfile(file):
                os.unlink(file)

class TestDesignAutos(unittest.TestCase):
    """Test design.py
    """
    def setUp(self):
        self.files = []

    def test_auto_from_file(self):
        pass

    def test_auto_from_args(self):
        pass

    def tearDown(self):
        for file in self.files:
            if os.path.isfile(file):
                os.unlink(file)

def baseArgv(search_type='sliding-window', input_type='fasta', 
             objective='minimize-guides', model=False, specific=None, 
             input_file=None, output_file=None):

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
                     '--max-primers-at-site', '1'])

    if objective == 'minimize-guides':
        argv.extend(['-gm', '0', '-gp', '.75'])

    if specific == 'fastas':
        argv.extend(['--specific-against-fastas', 'examples/powassan.tsv'])
    elif specific == 'taxa':
        argv.extend(['--specific-against-taxa', '11083'])

    if model:
        argv.extend(['--predict-activity-model-path', 'models/classify/model-51373185', 
                     'models/regress/model-f8b6fd5d'])
    elif objective =='maximize-activity':
        argv.extend(['--use-simple-binary-activity-prediction', '-gm', '1'])

    argv.extend(['--obj', objective, '--seed', '294', '--debug'])

    return argv
