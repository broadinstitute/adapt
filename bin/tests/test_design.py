"""Tests for design.py
"""

import random
import os
import copy
import tempfile
import unittest
import logging
import math

from collections import OrderedDict
from argparse import Namespace
from adapt import alignment
from adapt.prepare import align, prepare_alignment, cluster
from adapt.utils import seq_io, thermo
from bin import design

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'

# Default args: window size 3, guide size 2, allow GU pairing
# GU pairing allows AA to match GG in 1st window, and AC to
# match GC in the 2nd window
SEQS = OrderedDict()
# 1 Dengue accession
SEQS["OK605599.1"] = "AACTA"
# 3 Zika accessions
SEQS["OK571913.1"] = "AAACT"
SEQS["OK054351.1"] = "GGCTA"
SEQS["MZ008356.1"] = "GGCTT"

# Specificity seq stops AA from being the best guide in the 1st window
SP_SEQS = OrderedDict()
SP_SEQS["genome_X"] = "AA---"

FAKE_DNA_DNA_INSIDE ={
    'A': {
        'A': (1, 0.005),
        'T': (10, 0.05),
        'C': (1, 0.005),
        'G': (1, 0.005),
    },
    'T': {
        'A': (10, 0.05),
        'T': (1, 0.005),
        'C': (1, 0.005),
        'G': (1, 0.005),
    },
    'C': {
        'A': (1, 0.005),
        'T': (1, 0.005),
        'C': (1, 0.005),
        'G': (10, 0.05),
    },
    'G': {
        'A': (1, 0.005),
        'T': (1, 0.005),
        'C': (10, 0.05),
        'G': (1, 0.005),
    }
}

FAKE_DNA_DNA_INTERNAL = {
    'A': FAKE_DNA_DNA_INSIDE,
    'T': FAKE_DNA_DNA_INSIDE,
    'C': FAKE_DNA_DNA_INSIDE,
    'G': FAKE_DNA_DNA_INSIDE,
}

FAKE_DNA_DNA_TERMINAL = FAKE_DNA_DNA_INTERNAL

FAKE_DNA_DNA_TERM_GC = (0, 0)

FAKE_DNA_DNA_SYM = (0, 0)

FAKE_DNA_DNA_TERM_AT = (0, 0)

# With 2 bp matching, delta H is 20 and delta S is 0.1. With the thermodynamic
# conditions set to not interfere, the melting temperature is delta H/delta S,
# which is 200K (Note: this doesn't actually make sense in practice, as Tm
# can't go below 0°C; this is a toy example to make testing easier)
PERFECT_TM = 200 - thermo.CELSIUS_TO_KELVIN


class TestDesign(object):
    """General class for testing design.py

    Defines helper functions for test cases and basic setUp and
    tearDown functions.
    """
    class TestDesignCase(unittest.TestCase):
        def setUp(self):
            # Disable logging
            logging.disable(logging.INFO)

            # Temporarily set constants to fake values
            self.DNA_DNA_INTERNAL = thermo.DNA_DNA_INTERNAL
            self.DNA_DNA_TERMINAL = thermo.DNA_DNA_TERMINAL
            self.DNA_DNA_TERM_GC = thermo.DNA_DNA_TERM_GC
            self.DNA_DNA_SYM = thermo.DNA_DNA_SYM
            self.DNA_DNA_TERM_AT = thermo.DNA_DNA_TERM_AT

            thermo.DNA_DNA_INTERNAL = FAKE_DNA_DNA_INTERNAL
            thermo.DNA_DNA_TERMINAL = FAKE_DNA_DNA_TERMINAL
            thermo.DNA_DNA_TERM_GC = FAKE_DNA_DNA_TERM_GC
            thermo.DNA_DNA_SYM = FAKE_DNA_DNA_SYM
            thermo.DNA_DNA_TERM_AT = FAKE_DNA_DNA_TERM_AT

            # Create a temporary input file
            self.input_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            # Closes the file so that it can be reopened on Windows
            self.input_file.close()

            # Create a temporary output file
            self.output_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            self.output_file.close()

            self.files_to_delete = [self.input_file.name, self.output_file.name]

        def check_results(self, file, expected, header='target-sequences'):
            """Check the results of the test output

            Given a TSV file of test output and expected output, fails the test
            if the test output guide target sequences do not equal the expected
            guide target sequences

            Args:
                file: string, path name of the file
                expected: list of lists of strings, all the expected guide
                    target sequences in each line of the output; if an inner
                    list is instead a set of lists, any of the lists in the
                    set can be a correct output
                header: the header of the CSV that contains the guide target
                    sequences
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
                    ei = expected[i-1]
                    if not isinstance(ei, set):
                        ei = {tuple(ei)}
                    val_line = line[:-1].split('\t')[col_loc].strip('[]')
                    vals = val_line.split(' ')
                    an_option_is_ok = False
                    for eij in ei:
                        is_correct = False
                        comp_fn = lambda a,b: a==b if isinstance(eij[0], str) \
                            else lambda a,b: math.isclose(float(a.strip(',')),b)
                        for val in vals:
                            for eijk in eij:
                                if comp_fn(val, eijk):
                                    is_correct = True
                        if len(vals) != len(eij):
                            is_correct = False
                        if is_correct:
                            an_option_is_ok = True
                    self.assertTrue(an_option_is_ok,
                            msg=(f"The design with {header} {vals} does "
                                f"not match any expected solution "
                                f"({ei})"))
                self.assertEqual(i, len(expected))

        def baseArgv(self, search_type='sliding-window', input_type='fasta',
                     objective='minimize-guides', model=False, specific=None,
                     specificity_file=None, output_loc=None, unaligned=False,
                     gp=0.75, weighted=False, allow_gu_pairing=True,
                     primer_thermo=False):
            """Get arguments for tests

            Produces the correct arguments for a test case given details of
            what the test case is testing. See design.py help for details
            on input

            Args:
                search_type: 'sliding-window' or 'complete-targets'
                input_type: 'fasta', 'auto-from-args', or 'auto-from-file'
                objective: 'minimize-guides' or 'maximize-activity'
                model: boolean, true to use Cas13a built in model, false
                    to use simple binary prediction
                specific: None, 'fasta', or 'taxa'; what sort of input
                    to be specific against
                output_loc: path to the output file/directory; set to
                    self.output_file.name if None
                unaligned: boolean, if input type is FASTA, true to align seqs
                    before designing, False otherwise
                weighted: if input type is FASTA, true to use manual weights
                    for sequences (self.weight_file must be defined);
                    if input type is 'auto-from-args', true to weight by the
                    log of the subtaxa
                allow_gu_pairing: True to allow GU pairing; False otherwise

            Returns:
                List of strings that are the arguments of the test
            """
            input_file = self.input_file.name
            if output_loc is None:
                output_loc = self.output_file.name
            argv = ['design.py', search_type, input_type]

            if input_type == 'fasta':
                argv.extend([input_file, '-o', output_loc])
                if unaligned:
                    argv.extend(['--unaligned'])
                if weighted:
                    argv.extend(['--weight-sequences', self.weight_file.name])
            elif input_type == 'auto-from-args':
                if weighted:
                    argv.extend(['11051', 'None', output_loc,
                        '--weight-by-log-size-of-subtaxa', 'species'])
                else:
                    argv.extend(['64320', 'None', output_loc])
            elif input_type == 'auto-from-file':
                argv.extend([input_file, output_loc])

            if not allow_gu_pairing:
                argv.append('--do-not-allow-gu-pairing')

            if input_type in ['auto-from-args', 'auto-from-file']:
                argv.extend(['--sample-seqs', '1', '--mafft-path', 'fake_path'])

            if search_type == 'sliding-window':
                argv.extend(['-w', '3', '-gl', '2'])
            elif search_type == 'complete-targets':
                if primer_thermo:
                    argv.extend(['-na', '1', '-mg', '0', '--pcr-dntp-conc',
                                 '0', '--pcr-oligo-conc', '1', '-pl', '2',
                                 '--primer-thermo', '-gl', '1',
                                 '--ideal-primer-melting-temperature',
                                 str(PERFECT_TM)])
                else:
                    argv.extend(['--best-n-targets', '2', '-pp', '.75', '-pl',
                                 '1', '--max-primers-at-site', '2',
                                 '-gl', '2'])

            if objective == 'minimize-guides':
                argv.extend(['-gm', '0', '-gp', str(gp)])
            elif objective =='maximize-activity':
                argv.extend(['--maximization-algorithm', 'greedy'])

            # ID-M (mismatches to be considered identical) must be set to 0 since otherwise
            # having 1 base in common with a 2 base guide counts as a match
            if specific == 'fasta':
                argv.extend(['--specific-against-fastas', specificity_file, '--id-m', '0'])
            elif specific == 'taxa':
                argv.extend(['--specific-against-taxa', specificity_file, '--id-m', '0'])

            if model:
                argv.append('--predict-cas13a-activity-model')
            elif objective =='maximize-activity':
                argv.extend(['--use-simple-binary-activity-prediction', '-gm', '0'])

            argv.extend(['--obj', objective, '--seed', '0'])

            return argv

        def tearDown(self):
            for file in self.files_to_delete:
                if os.path.isfile(file):
                    os.unlink(file)
            # Re-enable logging
            logging.disable(logging.NOTSET)

            # Fix modified constants and functions
            thermo.DNA_DNA_INTERNAL = self.DNA_DNA_INTERNAL
            thermo.DNA_DNA_TERMINAL = self.DNA_DNA_TERMINAL
            thermo.DNA_DNA_TERM_GC = self.DNA_DNA_TERM_GC
            thermo.DNA_DNA_SYM = self.DNA_DNA_SYM
            thermo.DNA_DNA_TERM_AT = self.DNA_DNA_TERM_AT


class TestDesignFasta(TestDesign.TestDesignCase):
    """Test design.py given an input FASTA
    """

    def setUp(self):
        super().setUp()
        self.real_output_file = self.output_file.name + '.tsv'
        self.files_to_delete.append(self.real_output_file)

        # Write to temporary input fasta
        seq_io.write_fasta(SEQS, self.input_file.name)

    def test_min_guides(self):
        argv = super().baseArgv()
        args = design.argv_to_args(argv)
        design.run(args)
        # Base args set the percentage of sequences to match at 75%
        expected = [["AA"], {("CT",), ("AC",)}, ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_max_activity(self):
        argv = super().baseArgv(objective='maximize-activity')
        args = design.argv_to_args(argv)
        design.run(args)
        # Doesn't use model, just greedy binary prediction with 0 mismatches
        # (so same outputs as min-guides)
        expected = [["AA"], {("CT",), ("AC",)}, ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_complete_targets(self):
        argv = super().baseArgv(search_type='complete-targets')
        args = design.argv_to_args(argv)
        design.run(args)
        # Since sequences are short and need 1 base for primer on each side,
        # only finds 1 target in middle
        expected = [{("CT",), ("AC",)}]
        self.check_results(self.real_output_file, expected,
                           header='guide-target-sequences')

    def test_specificity_fastas(self):
        # Create a temporary fasta file for specificity
        self.sp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.sp_fasta.close()

        seq_io.write_fasta(SP_SEQS, self.sp_fasta.name)

        self.files_to_delete.append(self.sp_fasta.name)

        argv = super().baseArgv(specific='fasta',
            specificity_file=self.sp_fasta.name)
        args = design.argv_to_args(argv)
        design.run(args)
        # AA isn't allowed in 1st window by specificity fasta,
        # so 1st window changes
        expected = [{("AC", "GG"), ("AC",)}, {("CT",), ("AC",)}, ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_primer_thermo(self):
        argv = super().baseArgv(search_type='complete-targets',
            primer_thermo=True)
        args = design.argv_to_args(argv)
        design.run(args)
        expected = [{("C",)}]
        self.check_results(self.real_output_file, expected,
            header='guide-target-sequences')
        expected = [{("AA", "GG")}]
        self.check_results(self.real_output_file, expected,
            header='left-primer-target-sequences')
        expected = [{("CT", "TA")}]
        self.check_results(self.real_output_file, expected,
            header='right-primer-target-sequences')
        expected = [{(PERFECT_TM, PERFECT_TM)}]
        self.check_results(self.real_output_file, expected,
            header='left-primer-ideal-melting-temperature')
        expected = [{(PERFECT_TM, PERFECT_TM)}]
        self.check_results(self.real_output_file, expected,
            header='right-primer-ideal-melting-temperature')


class TestDesignFastaUnaligned(TestDesign.TestDesignCase):
    """Test design.py given an input FASTA
    """

    def setUp(self):
        super().setUp()
        self.real_output_file = self.output_file.name + '.0.tsv'
        self.files_to_delete.append(self.real_output_file)

        unaligned_seqs = {key: value for key, value in SEQS.items()}
        unaligned_seqs["genome_X"] = "GGCT"
        # unaligned_seqs["genome_5"] = "GGCT"

        # Write to temporary input fasta
        seq_io.write_fasta(unaligned_seqs, self.input_file.name)

        # We cannot access MAFFT, so override this function; store original so
        # it can be fixed for future tests
        self.set_mafft_exec = align.set_mafft_exec
        align.set_mafft_exec = lambda mafft_path: None

        # Aligning requires MAFFT, so override this function and output simple
        # test sequences; store original so it can be fixed for future tests
        self.align = align.align
        align.align = lambda seqs, am=None: SEQS

        # Clustering will not work on short sequences in FASTA, so override;
        # store original so it can be fixed for future tests
        self.cluster = cluster.cluster_with_minhash_signatures
        cluster.cluster_with_minhash_signatures = lambda seqs, threshold=0.1: [seqs]

    def test_min_guides(self):
        argv = super().baseArgv(unaligned=True)
        args = design.argv_to_args(argv)
        design.run(args)
        # Base args set the percentage of sequences to match at 75%
        expected = [["AA"], {("CT",), ("AC",)}, ["CT"]]
        self.check_results(self.real_output_file, expected)

    def tearDown(self):
        # Fix all overridden functions
        align.set_mafft_exec = self.set_mafft_exec
        cluster.cluster_with_minhash_signatures = self.cluster
        align.align = self.align
        super().tearDown()


class TestDesignFastaWeighted(TestDesign.TestDesignCase):
    """Test design.py given an input FASTA
    """

    def setUp(self):
        super().setUp()
        self.real_output_file = self.output_file.name + '.tsv'
        self.files_to_delete.append(self.real_output_file)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            self.weight_file = f
            f.write("OK605599.1\t70\nOK571913.1\t5\n"
                    "OK054351.1\t15\nMZ008356.1\t5\n")

        self.files_to_delete.extend([self.real_output_file,
                                     self.weight_file.name])

        # Write to temporary input fasta
        seq_io.write_fasta(SEQS, self.input_file.name)

    def test_min_guides_weighted(self):
        argv = super().baseArgv(weighted=True)
        args = design.argv_to_args(argv)
        design.run(args)
        # Base args set the percentage of sequences to match at 75%
        expected = [["AA"], ["AC"], ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_max_activity_weighted(self):
        argv = super().baseArgv(weighted=True, objective='maximize-activity')
        args = design.argv_to_args(argv)
        design.run(args)
        # Doesn't use model, just greedy binary prediction with 0 mismatches
        # (so same outputs as min-guides)
        expected = [["AA"], ["AC"], ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_complete_targets_weighted(self):
        argv = super().baseArgv(weighted=True, search_type='complete-targets')
        args = design.argv_to_args(argv)
        design.run(args)
        # Since sequences are short and need 1 base for primer on each side,
        # only finds 1 target in middle
        expected = [["AC"]]
        self.check_results(self.real_output_file, expected,
                           header='guide-target-sequences')


class TestDesignAutosPartial(TestDesign.TestDesignCase):
    """Test design.py given arguments to automatically download FASTAs

    Does not run the entire design.py; prematurely stops by giving a fake path
    to MAFFT. All are expected to return a FileNotFoundError
    """
    def setUp(self):
        super().setUp()

        # Write to temporary input file
        with open(self.input_file.name, 'w') as f:
            f.write("Zika virus\t64320\tNone\tNC_035889\n")

        # Create a temporary output directory
        self.output_dir = tempfile.TemporaryDirectory()

    def test_auto_from_file(self):
        argv = super().baseArgv(input_type='auto-from-file',
                        output_loc=self.output_dir.name)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_auto_from_args(self):
        argv = super().baseArgv(input_type='auto-from-args')
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_specificity_taxa(self):
        argv = super().baseArgv(input_type='auto-from-args',
                        specific='taxa', specificity_file='')
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def test_weighted(self):
        argv = super().baseArgv(input_type='auto-from-args', weighted=True)
        args = design.argv_to_args(argv)
        try:
            design.run(args)
        except FileNotFoundError:
            pass

    def tearDown(self):
        super().tearDown()
        self.output_dir.cleanup()


class TestDesignAutosFull(TestDesign.TestDesignCase):
    """Test design.py fully through
    """
    def setUp(self):
        super().setUp()

        # Write to temporary input file
        with open(self.input_file.name, 'w') as f:
            f.write("Zika virus\t64320\tNone\tNC_035889\n")

        # Create a temporary specificity file
        self.sp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.sp_file.write("64286\tNone\n")
        # Closes the file so that it can be reopened on Windows
        self.sp_file.close()

        # 'auto-from-args' gives different outputs for every cluster
        # Our test only produces 1 cluster, so store the name of that file
        self.real_output_file = self.output_file.name + '.0.tsv'
        self.files_to_delete.extend([self.sp_file.name, self.real_output_file])

        # We cannot access MAFFT, so override this function; store original so
        # it can be fixed for future tests
        self.set_mafft_exec = align.set_mafft_exec
        align.set_mafft_exec = lambda mafft_path: None

        # Curating requires MAFFT, so override this function; store original so
        # it can be fixed for future tests
        self.curate_against_ref = align.curate_against_ref

        def small_curate(seqs, ref_accs, asm=None):
            return {seq: seqs[seq] for seq in seqs}

        align.curate_against_ref = small_curate

        # Aligning requires MAFFT, so override this function and output simple
        # test sequences; store original so it can be fixed for future tests
        self.align = align.align
        align.align = lambda seqs, am=None: SEQS

        # We don't want to fetch sequences for the specificity file since we're
        # doing a simple test case, so override this function; store original
        # so it can be fixed for future tests
        self.fetch_sequences_for_taxonomy = prepare_alignment.fetch_sequences_for_taxonomy

        def small_fetch(taxid, segment):
            # Test fetching the real sequences, but don't return them as they
            # won't be used
            # Note, this function is only used for specificity
            self.fetch_sequences_for_taxonomy(taxid, segment)
            return SP_SEQS

        prepare_alignment.fetch_sequences_for_taxonomy = small_fetch

        # Disable warning logging to avoid annotation warning
        logging.disable(logging.WARNING)

    def test_specificity_taxa(self):
        argv = super().baseArgv(input_type='auto-from-args', specific='taxa',
            specificity_file=self.sp_file.name)
        args = design.argv_to_args(argv)
        design.run(args)
        # Same output as test_specificity_fasta, as sequences are the same
        expected = [{("AC", "GG"), ("AC",)}, {("CT",), ("AC",)}, ["CT"]]
        self.check_results(self.real_output_file, expected)

    def test_weighted(self):
        # GP of 0.54 means covering Dengue sequence (weight: 1/3) + 1 sequence
        # from Zika (weight: 2/9) is sufficient (total: 5/9 ~= .556)
        argv = super().baseArgv(input_type='auto-from-args', weighted=True,
            allow_gu_pairing=False, gp=0.54)
        args = design.argv_to_args(argv)
        design.run(args)
        # Since GU pairs aren't allowed, GG won't work, but AA covers Dengue
        # + 1 Zika, so AA is sufficient
        expected = [["AA"], ["CT"], ["CT"]]
        self.check_results(self.real_output_file, expected)

    def tearDown(self):
        # Fix all overridden functions
        align.set_mafft_exec = self.set_mafft_exec
        align.curate_against_ref = self.curate_against_ref
        align.align = self.align
        prepare_alignment.fetch_sequences_for_taxonomy = self.fetch_sequences_for_taxonomy
        super().tearDown()
