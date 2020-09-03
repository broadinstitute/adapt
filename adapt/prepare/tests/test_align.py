"""Tests for align module.
"""

from collections import OrderedDict
import tempfile
from os import unlink
import unittest
try:
    import boto3
    import botocore
    S3 = boto3.client()
    resource = S3.list_buckets()
except:
    cloud = False
else:
    cloud = True
    import random
    import warnings

from adapt.prepare import align

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignStats(unittest.TestCase):
    """Tests functions manipulating and computing stats on alignments.
    """

    def test_aln_identity(self):
        a = 'ATCGATGGAT--G'
        b = 'ATAGTAGGATAAG'
        self.assertEqual(align._aln_identity(a, b), 8.0/13)

    def test_collapse_consecutive_gaps_simple(self):
        a = 'ATC----GA-TAT'
        b = 'ATCATCGGAAT--'

        expected_a = 'ATC-GA-TA'
        expected_b = 'ATCAGAAT-'

        self.assertEqual(align._collapse_consecutive_gaps(a, b),
            (expected_a, expected_b))

    def test_collapse_consecutive_gaps_simple(self):
        a = 'ATC----GA'
        b = 'ATCATCGGA'

        expected_a = 'ATC-GA'
        expected_b = 'ATCAGA'

        self.assertEqual(align._collapse_consecutive_gaps(a, b),
            (expected_a, expected_b))

    def test_collapse_consecutive_gaps_complex(self):
        a = 'ATC----GA-TAT'
        b = 'ATCATCGGAAT--'

        expected_a = 'ATC-GA-TA'
        expected_b = 'ATCAGAAT-'

        self.assertEqual(align._collapse_consecutive_gaps(a, b),
            (expected_a, expected_b))


class TestIO(unittest.TestCase):
    """Tests i/o functions.
    """

    def setUp(self):
        # Create a temporary file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.fasta.write(">AB123.2 Name of the sequence\n")
        self.fasta.write("ATCG\n")
        self.fasta.write("\n")
        self.fasta.write(">KY456 Another sequence\n")
        self.fasta.write("AATTAA\n")
        self.fasta.write("\n")
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()


    def test_read_unaligned_seqs(self):
        # Check that it keys by accession.version (or just accession)
        seqs = align.read_unaligned_seqs(self.fasta)
        self.assertEqual(len(seqs), 2)
        self.assertIn('AB123.2', seqs)
        self.assertEqual(seqs['AB123.2'], "ATCG")
        self.assertIn('KY456', seqs)
        self.assertEqual(seqs['KY456'], "AATTAA") 

    def tearDown(self):
        # Delete temporary file
        unlink(self.fasta.name)


class TestMemoization(unittest.TestCase):
    """Tests memoizers.
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        if cloud:
            cloudtempfold = random.getrandbits(32) 
            s3 = boto3.resource("s3")
            self.bucket = s3.Bucket("%08x" % cloudtempfold)
            self.bucket.create()
            self.cloudtempdir = "s3://%08x" % cloudtempfold

    def test_alignment_memoizer(self):
        def test_on_dir(directory):
            # Create some fake alignments
            seqs1 = OrderedDict([('AB123.1', 'ATCG--A'),
                                 ('AB456.1', 'TTTTCCA')])
            seqs2 = OrderedDict([('KY123.1', 'AAAACCA'),
                                 ('AB456.1', 'TTTTCCA')])
            seqs3 = OrderedDict([('AB123.1', 'ATCG--A'),
                                 ('KY123.1', 'AAAACCA'),
                                 ('AB456.1', 'TTTTCCA')])
            seqs4 = OrderedDict([('AB123.1', 'ATCG--A'),
                                 ('YZ123.1', 'GGGGCCA'),
                                 ('DE456.1', 'CCCCCCA')])
            am = align.AlignmentMemoizer(directory)

            # Progressively save these and check the results
            am.save(seqs1)
            am.save(seqs2)
            self.assertEqual(am.get(seqs1.keys()), seqs1)
            self.assertEqual(am.get(seqs2.keys()), seqs2)
            self.assertIsNone(am.get(seqs3.keys()))
            self.assertIsNone(am.get(seqs4.keys()))

            am.save(seqs3)
            self.assertEqual(am.get(seqs1.keys()), seqs1)
            self.assertEqual(am.get(seqs2.keys()), seqs2)
            self.assertEqual(am.get(seqs3.keys()), seqs3)
            self.assertIsNone(am.get(seqs4.keys()))

            am.save(seqs4)
            self.assertEqual(am.get(seqs1.keys()), seqs1)
            self.assertEqual(am.get(seqs2.keys()), seqs2)
            self.assertEqual(am.get(seqs3.keys()), seqs3)
            self.assertEqual(am.get(seqs4.keys()), seqs4)

            seqs3 = OrderedDict([('AB123.1', 'ATCG--A'),
                                 ('KY123.1', 'GGGGGGG'),
                                 ('AB456.1', 'TTTTCCA')])
            am.save(seqs3)
            self.assertEqual(am.get(seqs1.keys()), seqs1)
            self.assertEqual(am.get(seqs2.keys()), seqs2)
            self.assertEqual(am.get(seqs3.keys()), seqs3)
            self.assertEqual(am.get(seqs4.keys()), seqs4)
            del am

            # Try reading from a new AlignmentMemoizer at the same path
            am_new = align.AlignmentMemoizer(directory)
            self.assertEqual(am_new.get(seqs1.keys()), seqs1)
            self.assertEqual(am_new.get(seqs2.keys()), seqs2)
            self.assertEqual(am_new.get(seqs3.keys()), seqs3)
            self.assertEqual(am_new.get(seqs4.keys()), seqs4)

            # Reorder keys and check that output is the same
            seqs4_keys = list(seqs4.keys())
            self.assertEqual(am_new.get(seqs4_keys), seqs4)
            self.assertEqual(am_new.get(seqs4_keys[::-1]), seqs4)
        
        test_on_dir(self.tempdir.name)
        
        # If Boto3 is installed, test AWS cloud services
        if cloud:
            test_on_dir(self.cloudtempdir)

    def test_alignment_stat_memoizer(self):
        def test_on_dir(directory):
            # Create some fake stats
            asm = align.AlignmentStatMemoizer(directory)
            asm.save(['AB123.1', 'KY456.2'], (0.8, 0.9))
            asm.save(['AB123.1', 'KY789.2'], (0.85, 0.95))
            self.assertEqual(asm.get(['AB123.1', 'KY456.2']), (0.8, 0.9))
            self.assertEqual(asm.get(['AB123.1', 'KY789.2']), (0.85, 0.95))
            self.assertEqual(asm.get(['KY789.2', 'AB123.1']), (0.85, 0.95))
            self.assertIsNone(asm.get(['KY456.2', 'KY789.2']))
            del asm

            # Check that a new AlignmentStatMemoizer can read these
            asm_new = align.AlignmentStatMemoizer(directory)
            self.assertEqual(asm_new.get(['AB123.1', 'KY456.2']), (0.8, 0.9))
            self.assertEqual(asm_new.get(['AB123.1', 'KY789.2']), (0.85, 0.95))
            self.assertEqual(asm_new.get(['KY789.2', 'AB123.1']), (0.85, 0.95))
            self.assertIsNone(asm_new.get(['KY456.2', 'KY789.2']))

            # Add a new stat to the memoizer
            asm_new.save(['AB123.1', 'KY000.1'], (0.5, 0.6))
            del asm_new

            # Check that we can read all the stats
            asm_new2 = align.AlignmentStatMemoizer(directory)
            self.assertEqual(asm_new2.get(['AB123.1', 'KY456.2']), (0.8, 0.9))
            self.assertEqual(asm_new2.get(['AB123.1', 'KY789.2']), (0.85, 0.95))
            self.assertEqual(asm_new2.get(['KY789.2', 'AB123.1']), (0.85, 0.95))
            self.assertIsNone(asm_new2.get(['KY456.2', 'KY789.2']))
            self.assertEqual(asm_new2.get(['AB123.1', 'KY000.1']), (0.5, 0.6))

        test_on_dir(self.tempdir.name)

        # If Boto3 is installed, test AWS cloud services
        if cloud:
            test_on_dir(self.cloudtempdir)

    def tearDown(self):
        self.tempdir.cleanup()
        if cloud:
            self.bucket.objects.delete()
            self.bucket.delete()


class TestCurateAgainstRef(unittest.TestCase):
    """Tests the curate_against_ref() function.
    """

    def setUp(self):
        self.seqs = {'AB123.1': 'ATCGAAATTTA',
                     'AB456.1': 'AGCGAAGTTA',
                     'AB789.1': 'GGGGG',
                     'KY123.1': 'GAAAA',
                     'KY456.1': 'TTCGAAATTTA',
                     'KY789.1': 'GAAATTT',
                     'KZ123.1': 'GGGGGGGGGGG'}

        aln1  = {'AB123.1': 'ATCGAAATTTA',
                 'AB456.1': 'AGCGAAGTT-A'}
        aln2  = {'AB123.1': 'ATCGAAATTTA',
                 'AB789.1': '---GGGGG---'}
        aln3  = {'AB123.1': 'ATCGAAATTTA',
                 'KY123.1': '---GAAAA---'}
        aln4  = {'AB123.1': 'ATCGAAATTTA',
                 'KY456.1': 'TTCGAAATTTA'}
        aln5  = {'AB123.1': 'ATCGAAATTTA',
                 'KY789.1': '---GAAATTT-'}
        aln6  = {'AB123.1': 'ATCGAAATTTA',
                 'KZ123.1': 'GGGGGGGGGGG'}
        aln7  = {'KZ123.1': 'GGGGGGGGGGG',
                 'AB456.1': 'AGCGAAGTTA-'}
        aln8  = {'KZ123.1': 'GGGGGGGGGGG',
                 'AB789.1': '---GGGGG---'}
        aln9  = {'KZ123.1': 'GGGGGGGGGGG',
                  'KY123.1': 'GAAAA------'}
        aln10 = {'KZ123.1': 'GGGGGGGGGGG',
                 'KY456.1': 'TTCGAAATTTA'}
        aln11 = {'KZ123.1': 'GGGGGGGGGGG',
                 'KY789.1': '--GAAATTT--'}
        alns = [aln1, aln2, aln3, aln4, aln5, aln6,
                aln7, aln8, aln9, aln10, aln11]

        expected_curated_seq_accs1 = ['AB123.1', 'AB456.1', 'KY456.1',
                'KY789.1']
        self.expected_curated_seqs1 = {acc: self.seqs[acc]
            for acc in expected_curated_seq_accs1}

        expected_curated_seq_accs2 = ['AB123.1', 'AB456.1', 'KY456.1',
                'KY789.1', 'KZ123.1']
        self.expected_curated_seqs2 = {acc: self.seqs[acc]
            for acc in expected_curated_seq_accs2}
        
        # Override align.align() to provide expected alignments, but
        # keep the real function
        self.align_real = align.align
        def a(s):
            for aln in alns:
                if s.keys() == aln.keys():
                    return aln
        align.align = a

    def test_ref_acc_with_ver(self):
        ref_acc = 'AB123.1'
        self.assertEqual(
            align.curate_against_ref(self.seqs, [ref_acc],
                remove_ref_accs=[]),
            self.expected_curated_seqs1)

    def test_ref_acc_without_ver(self):
        ref_acc = 'AB123'
        self.assertEqual(
            align.curate_against_ref(self.seqs, [ref_acc],
                remove_ref_accs=[]),
            self.expected_curated_seqs1)

    def test_multiple_ref_accs(self):
        ref_accs = ['KZ123', 'AB123']
        self.assertEqual(
            align.curate_against_ref(self.seqs, ref_accs,
                remove_ref_accs=[]),
            self.expected_curated_seqs2)

    def tearDown(self):
        # Reset align.align()
        align.align = self.align_real
