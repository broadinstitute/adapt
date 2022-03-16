"""Tests for analyze_coverage.py
"""

import os
import tempfile
import unittest
import logging
import math

from collections import OrderedDict
from argparse import Namespace
from adapt import alignment
from adapt.prepare import align, ncbi_neighbors, prepare_alignment, cluster
from adapt.utils import seq_io, predict_activity
from bin import analyze_coverage

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'

SEQS = OrderedDict()
SEQS["genome_1"] = "AACTAG"
SEQS["genome_2"] = "AACACG"
SEQS["genome_3"] = "GCCTAG"
SEQS["genome_4"] = "AATTCG"

DESIGN_HEADERS = ['target-start',
                  'target-end',
                  'guide-target-sequences',
                  'left-primer-target-sequences',
                  'right-primer-target-sequences']
DESIGN_OPTS = [
    ['0', '5', 'CT', 'AA', 'AG'],
    ['0', '5', 'CT CA', 'AA GC', 'AG CG'],
]

class TestAnalyzeCoverage(object):
    """General class for testing analyze_coverage.py

    Defines helper functions for test cases and basic setUp and
    tearDown functions.
    """
    class TestAnalyzeCoverageCase(unittest.TestCase):
        def setUp(self):
            # Disable logging
            logging.disable(logging.INFO)

            # Create a temporary input files
            self.input_designs = tempfile.NamedTemporaryFile(mode='w', delete=False)
            self.input_seqs = tempfile.NamedTemporaryFile(mode='w', delete=False)
            # Create temporary output files
            self.write_frac_bound = tempfile.NamedTemporaryFile(mode='w', delete=False)
            self.write_mean_activity = tempfile.NamedTemporaryFile(mode='w', delete=False)
            self.write_per_seq = tempfile.NamedTemporaryFile(mode='w', delete=False)
            # Closes the files so that they can be reopened on Windows
            self.input_designs.close()
            self.input_seqs.close()
            self.write_frac_bound.close()
            self.write_mean_activity.close()
            self.write_per_seq.close()

            # Write input seqs
            seq_io.write_fasta(SEQS, self.input_seqs.name)
            # Write input designs
            with open(self.input_designs.name, 'w') as f:
                f.write("\t".join(DESIGN_HEADERS)+"\n")
                for design_opt in DESIGN_OPTS:
                    f.write("\t".join(design_opt)+"\n")

            self.files_to_delete = [self.input_designs.name,
                                    self.input_seqs.name,
                                    self.write_frac_bound.name,
                                    self.write_mean_activity.name,
                                    self.write_per_seq.name]

            def small_fetch(accessions):
                return self.input_seqs
            self.fetch_fastas = ncbi_neighbors.fetch_fastas
            ncbi_neighbors.fetch_fastas = small_fetch

        def check_results(self, file, expected, header):
            """Check the results of the test output

            Given a TSV file of test output and expected output, fails the test
            if the test output guide target sequences do not equal the expected
            guide target sequences

            Args:
                file: string, path name of the file
                expected: list of lists of strings, all the expected values for
                    each line of the output
                header: the header of the CSV that contains the value to check
            """
            col_loc = None
            with open(file) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        headers = line[:-1].split('\t')
                        # Will raise an error if header is not in output
                        col_loc = headers.index(header)
                        continue
                    self.assertLess(i, len(expected) + 1)
                    val_line = line[:-1].split('\t')[col_loc]
                    vals = val_line.split(' ')
                    for val in vals:
                        if val != 'None':
                            self.assertIn(float(val), expected[i-1])
                        else:
                            self.assertIn('None', expected[i-1])
                    self.assertEqual(len(vals), len(expected[i-1]))
                self.assertEqual(i, len(expected))

        def baseArgv(self, model=False, use_accessions=False,
                     primer_terminal_mismatches=False):
            """Get arguments for tests

            Produces the correct arguments for a test case given details of
            what the test case is testing. See design.py help for details
            on input

            Args:
                search_type: 'sliding-window' or 'complete-targets'
                model: boolean, true to use Cas13a built in model, false
                    to use simple binary prediction
                use_accessions: boolean, true to use an accession file for
                    sequences, false otherwise

            Returns:
                List of strings that are the arguments of the test
            """
            argv = ['analyze_coverage.py',
                    self.input_designs.name]

            if use_accessions:
                self.input_accs = tempfile.NamedTemporaryFile(mode='w', delete=False)
                self.input_accs.close()
                self.files_to_delete.append(self.input_accs.name)
                with open(self.input_accs.name, 'w') as f:
                    f.write("NC_035889\n")
                argv.extend([self.input_accs.name, '--use-accessions'])
            else:
                argv.append(self.input_seqs.name)

            if primer_terminal_mismatches:
                argv.extend(['-ptm', '0', '--bases-from-terminal', '1'])

            if model:
                argv.extend(['--predict-cas13a-activity-model',
                             '--write-mean-activity-of-guides',
                             self.write_mean_activity.name])
                class PredictorTest:
                    def __init__(self, *args, **kwargs):
                        self.context_nt = 0

                    def determine_highly_active(self, start_pos, pairs):
                        y = []
                        for target, guide in pairs:
                            target_without_context = target[
                                self.context_nt:len(target)-self.context_nt]
                            if guide == target_without_context:
                                if guide[1] == 'A':
                                    y += [True]
                                else:
                                    y += [False]
                            else:
                                y += [False]
                        return y

                    def compute_activity(self, start_pos, pairs):
                        y = []
                        for target, guide in pairs:
                            target_without_context = target[
                                self.context_nt:len(target)-self.context_nt]
                            if guide == target_without_context:
                                if guide[1] == 'A':
                                    y += [2]
                                else:
                                    y += [1]
                            else:
                                y += [0]
                        return y

                    def cleanup_memoized(self, pos):
                        pass
                predict_activity.Predictor = PredictorTest
            else:
                argv.extend(['-gm', '0'])

            argv.extend(['--write-frac-bound', self.write_frac_bound.name,
                         '--write-per-seq', self.write_per_seq.name,
                         '--fully-sensitive'])
            return argv

        def tearDown(self):
            for file in self.files_to_delete:
                if os.path.isfile(file):
                    os.unlink(file)
            # Re-enable logging
            logging.disable(logging.NOTSET)
            ncbi_neighbors.fetch_fastas = self.fetch_fastas


class TestAnalyzeCoverageCases(TestAnalyzeCoverage.TestAnalyzeCoverageCase):
    """Test analyze_coverage.py
    """

    def test_ptm(self):
        argv = super().baseArgv(primer_terminal_mismatches=True)
        args = analyze_coverage.argv_to_args(argv)
        analyze_coverage.run(args)
        expected = [[0.25], [1]]
        self.check_results(self.write_frac_bound.name, expected, 'frac-bound')
        expected = [[0], ['None'], [0], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'guide-mismatches')
        expected = [[0], [0], ['None'], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'left-primer-mismatches')
        expected = [[0], [0], ['None'], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'left-primer-terminal-mismatches')
        expected = [[0], ['None'], [0], ['None'],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'right-primer-mismatches')
        expected = [[0], ['None'], [0], ['None'],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'right-primer-terminal-mismatches')

    def test_accs(self):
        argv = super().baseArgv(use_accessions=True)
        args = analyze_coverage.argv_to_args(argv)
        analyze_coverage.run(args)
        expected = [[0.25], [1]]
        self.check_results(self.write_frac_bound.name, expected, 'frac-bound')
        expected = [[0], ['None'], [0], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'guide-mismatches')
        expected = [[0], [0], ['None'], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'left-primer-mismatches')
        expected = [[0], ['None'], [0], ['None'],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'right-primer-mismatches')

    def test_model(self):
        argv = super().baseArgv(model=True)
        args = analyze_coverage.argv_to_args(argv)
        analyze_coverage.run(args)
        expected = [[0.25], [0.75]]
        self.check_results(self.write_frac_bound.name, expected, 'frac-bound')
        expected = [[0.5], [1]]
        self.check_results(self.write_mean_activity.name, expected, 'mean-activity')
        expected = [[1], ['None'], [1], ['None'],
                    [1], [2], [1], ['None']]
        self.check_results(self.write_per_seq.name, expected, 'guide-activity')
        expected = [[0], [0], ['None'], [0],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'left-primer-mismatches')
        expected = [[0], ['None'], [0], ['None'],
                    [0], [0], [0], [0]]
        self.check_results(self.write_per_seq.name, expected, 'right-primer-mismatches')
