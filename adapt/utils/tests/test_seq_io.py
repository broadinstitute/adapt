"""Tests for seq_io module.
"""

from collections import OrderedDict
import logging
import tempfile
from os import unlink
import unittest

from adapt.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestFastaRead(unittest.TestCase):
    """Tests reading a fasta file.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

        # Write the temporary fasta file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.fasta.write(">genome_1\n")
        self.fasta.write("ATACG\n")
        self.fasta.write("TATGC\n")
        self.fasta.write(">genome_2\n")
        self.fasta.write("ATCG\n")
        self.fasta.write("TT\n")
        self.fasta.write("Gg\n")
        self.fasta.write("\n")
        self.fasta.write(">genome_3\n")
        self.fasta.write("AAA\n")
        self.fasta.write("CCC\n")
        self.fasta.write("\n")
        self.fasta.write("\n")
        self.fasta.write(">genome_4\n")
        self.fasta.write("AtA\n")
        self.fasta.write("CGC\n")
        self.fasta.write("\n")
        self.fasta.write("\n")
        self.fasta.write("\n")
        self.fasta.write(">genome_5\n")
        self.fasta.write("AGGA\n")
        self.fasta.write("CAAT\n")
        self.fasta.write("\n")
        self.fasta.write("\n")
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()

        self.expected = OrderedDict()
        self.expected["genome_1"] = "ATACGTATGC"
        self.expected["genome_2"] = "ATCGTTGG"
        self.expected["genome_3"] = "AAACCC"
        self.expected["genome_4"] = "ATACGC"
        self.expected["genome_5"] = "AGGACAAT"

    def test_read(self):
        seqs = seq_io.read_fasta(self.fasta.name, make_uppercase=True)
        self.assertEqual(seqs, self.expected)

    def tearDown(self):
        # Delete temporary file
        unlink(self.fasta.name)

        # Re-enable logging
        logging.disable(logging.NOTSET)


class TestFastaWrite(unittest.TestCase):
    """Tests writing and reading a fasta file.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

        # Create a temporary fasta file
        self.fasta = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # Closes the file so that it can be reopened on Windows
        self.fasta.close()

        self.seqs = OrderedDict()
        self.seqs["genome_1"] = "ATACGTATGC"
        self.seqs["genome_2"] = "ATCGTTGG"
        self.seqs["genome_3"] = "AAACCC"
        self.seqs["genome_4"] = "ATACGC"
        self.seqs["genome_5"] = "AGGACAAT"

    def test_write_and_read(self):
        seq_io.write_fasta(self.seqs, self.fasta.name)
        seqs_read = seq_io.read_fasta(self.fasta.name, make_uppercase=True)
        self.assertEqual(self.seqs, seqs_read)

    def tearDown(self):
        # Delete temporary file
        unlink(self.fasta.name)

        # Re-enable logging
        logging.disable(logging.NOTSET)
