"""Functions for working with sequence (and related) i/o.
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


def read_required_guides(fn, expected_guide_length, num_alignments):
    """Read list of required guides.

    There must be 3 columns in the file:
        1 - an identifier for an alignment that the guide should be
            covering (0-based, with maximum value < num_alignments)
        2 - the guide sequence
        3 - the position (start) of the guide sequence in the
            corresponding alignment

    Args:
        fn: path to file, with the format given above
        expected_guide_length: the length of each guide sequence in
            the file in nt; only used as a check
        num_alignments: the number of alignments that the required
            guides cover

    Returns:
        list x of length num_alignments such that x[i] corresponds
        to the i'th alignment, as given in column 1. x[i] is a
        dict mapping guide sequences (column 2) to their position
        (column 2)
    """
    required_guides = [{} for _ in range(num_alignments)]
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            aln_id = int(ls[0])
            gd = ls[1]
            gd_pos = int(ls[2])

            # Check aln_id
            if aln_id < 0 or aln_id > num_alignments - 1:
                raise Exception(("Alignment id %d in column 1 of required "
                    "guides file is invalid; must be in [0, %d]") %
                    (aln_id, num_alignments - 1))

            # Check guide length
            if len(gd) != expected_guide_length:
                raise Exception(("Guide with sequence '%s' in required guides "
                    "file has length %d, but it should have length %d") %
                    (gd, len(gd), expected_guide_length))

            # Check that the guide sequence only shows once for this
            # alignment (i.e., has not already appeared)
            if gd in required_guides[aln_id]:
                raise Exception(("Guide with sequence '%s' shows >1 time "
                    "for alignment with id %d") % (gd, aln_id))

            required_guides[aln_id][gd] = gd_pos

    return required_guides


def read_blacklisted_ranges(fn, num_alignments):
    """Read list of blacklisted ranges.

    There must be 3 columns in the file:
        1 - an identifier for an alignment that the guide should be
            covering (0-based, with maxmimum value < num_alignments)
        2 - the start position (inclusive) of a range in the
            corresponding alignment
        3 - the end position (exclusive) of a range in the
            corresponding alignment

    Args:
        fn: path to file, with the format given above
        num_alignments: the number of alignments that the ranges might
            correspond to
   
    Returns:
        list x of length num_alignments such that x[i] corresponds
        to the i'th alignment, as given in column 1. x[i] is a set
        of tuples (start, end) corresponding to the values in columns 2 and 3
    """
    blacklisted_ranges = [set() for _ in range(num_alignments)]
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            aln_id = int(ls[0])
            start = int(ls[1])
            end = int(ls[2])

            # Check aln_id
            if aln_id < 0 or aln_id > num_alignments - 1:
                raise Exception(("Alignment id %d in column 1 of blacklisted "
                    "ranges file is invalid; must be in [0, %d]") %
                    (aln_id, num_alignments - 1))

            # Check that end > start
            if start < 0 or end <= start:
                raise Exception(("Blacklisted range [%d, %d) is invalid; "
                    "values must be >= 0 and end > start") % (start, end))

            blacklisted_ranges[aln_id].add((start, end))

    return blacklisted_ranges


def read_blacklisted_kmers(fn, min_len_warning=5, max_len_warning=28):
    """Read file of blacklisted k-mers.

    Args:
        fn: path to FASTA file, where each sequence is a k-mer (names
            of sequences are ignored)
        min_len_warning/max_len_warning: log a warning if the k-mer
            length is outside this range

    Returns:
        set of k-mers
    """
    seqs = read_fasta(fn)
    kmers = set()
    for name, kmer in seqs.items():
        if len(kmer) < min_len_warning:
            logger.warning(("Blacklisted k-mer '%s' might be shorter than "
                "desired and may lead to many guides being treated as "
                "unsuitable") % kmer)
        if len(kmer) > max_len_warning:
            logger.warning(("Blacklisted k-mer '%s' might be longer than "
                "desired") % kmer)
        kmers.add(kmer)
    return kmers
