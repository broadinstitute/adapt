"""Functions for working with sequence i/o.
"""

from collections import OrderedDict
import logging
import re
import textwrap

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def read_fasta(fn, replace_degenerate=False,
               skip_gaps=False, make_uppercase=True):
    """Read a FASTA file.

    Args:
        fn: path to FASTA file to read
        replace_degenerate: when True, replace the degenerate
            bases ('Y','R','W','S','M','K','B','D','H','V')
            with 'N'
        skip_gaps: when True, do not read dashes ('-'), which
            represent gaps
        make_uppercase: when True, change all bases to be
            uppercase

    Returns:
        dict mapping the name of each sequence to the sequence
        itself. The mapping is ordered by the order in which
        the sequence is encountered in the FASTA file; this
        helps in particular with replicating past results,
        where the input order could affect the output.
    """
    logger.info("Reading fasta file %s", fn)

    degenerate_pattern = re.compile('[YRWSMKBDHV]')

    m = OrderedDict()
    with open(fn, 'r') as f:
        curr_seq_name = ""
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # Reset the sequence being read on an empty line
                curr_seq_name = ""
                continue
            if curr_seq_name == "":
                # Must encounter a new sequence
                assert line.startswith('>')
            if line.startswith('>'):
                curr_seq_name = line[1:]
                m[curr_seq_name] = ''
            else:
                # Append the sequence
                if make_uppercase:
                    line = line.upper()
                if replace_degenerate:
                    line = degenerate_pattern.sub('N', line)
                if skip_gaps:
                    line = line.replace('-', '')
                m[curr_seq_name] += line
    return m


def write_fasta(seqs, out_fn, chars_per_line=70):
    """Write sequences to a FASTA file.

    Args:
        seqs: dict (or OrderedDict) mapping the name of each
            sequence to the sequence itself
        out_fn: path to FASTA file to write
        chars_per_line: the number of characters to put on
            each line when writing a sequence
    """
    with open(out_fn, 'w') as f:
        for seq_name, seq in seqs.items():
            f.write('>' + seq_name + '\n')
            seq_wrapped = textwrap.wrap(seq, chars_per_line)
            for seq_line in seq_wrapped:
                f.write(seq_line + '\n')
            f.write('\n')
