"""Functions for working with sequence (and related) i/o.
"""

from collections import defaultdict
from collections import OrderedDict
import gzip
import logging
import re
import textwrap

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def process_fasta(f, replace_degenerate=False,
               skip_gaps=False, make_uppercase=True,
               replace_U=True):
    
    degenerate_pattern = re.compile('[YRWSMKBDHV]')
    m = OrderedDict()
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
            if replace_U:
                line = line.replace('u', 't').replace('U', 'T')
            m[curr_seq_name] += line

    return m


def read_fasta(fn, replace_degenerate=False,
               skip_gaps=False, make_uppercase=True,
               replace_U=True):
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
        replace_U: when True, change all 'U' bases to 'T'

    Returns:
        dict mapping the name of each sequence to the sequence
        itself. The mapping is ordered by the order in which
        the sequence is encountered in the FASTA file; this
        helps in particular with replicating past results,
        where the input order could affect the output.
    """
    logger.debug("Reading fasta file %s", fn)

    if fn.endswith('.gz'):
        with gzip.open(fn, 'rt') as f:
            m = process_fasta(f, replace_degenerate,
               skip_gaps, make_uppercase)
    else:
        with open(fn, 'r') as f:
            m = process_fasta(f, replace_degenerate,
               skip_gaps, make_uppercase)

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
            seq_wrapped = textwrap.wrap(seq, chars_per_line,
                break_on_hyphens=False)
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


def read_taxonomies(fn):
    """Read file of taxonomies from which to prepare alignments.

    The columns must be, in order:
        1) a label for the row (used for output files; must be unique)
        2) a taxonomic (e.g., species) ID from NCBI
        3) a segment label, or 'None' if unsegmented
        4) one or more accessions of reference sequences (comma-separated)
        5) (optional) metadata filters to only design for a subset of a taxa,
            format: 'metadata=value' or 'metadata!=value', 
                    commas to separate values, semicolons to separate filters

        6) (optional) metadata filters to exclude from a taxa and be specific against,
            column 5 must be included to include this column,
            format: 'metadata=value' or 'metadata!=value', 
                    commas to separate values, semicolons to separate filters

    Args:
        fn: path to TSV file, where each row corresponds to a taxonomy

    Returns:
        list of tuples (label, taxonomic_id, segment, reference_accessions, metadata_filters, metadata_filters_against)
    """
    labels = set()
    taxs = []
    with open(fn) as f:
        for line in f:
            none_strings = ['', 'none']
            ls = line.rstrip().split('\t')
            if len(ls) < 4 or len(ls) > 6:
                raise Exception(("Input taxonomy TSV must have between 4 and 6 columns"))

            label = ls[0]
            if label in labels:
                raise Exception(("Taxonomy label '%s' is not unique") % label)
            labels.add(label)

            try:
                tax_id = int(ls[1])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[1])

            segment = ls[2]
            if segment.lower() in none_strings:
                segment = None
            ref_accs = ls[3].split(',')

            meta_filt = None
            if len(ls) > 4 and ls[4].lower() not in none_strings:
                meta_filt = read_metadata_filters(ls[4].split(';'))

            meta_filt_against = None
            if len(ls) > 5 and ls[5].lower() not in none_strings:
                meta_filt_against = read_metadata_filters(ls[5].split(';'))

            taxs += [(label, tax_id, segment, ref_accs, meta_filt, meta_filt_against)]
    return taxs


def read_accessions_for_taxonomies(fn):
    """Read file of accessions for each taxonomy.

    The columns must be, in order:
        1) a taxonomic (e.g., species) ID from NCBI
        2) a segment label, or 'None' if unsegmented
        3) an accession to include in the design for the taxonomic ID

    A taxonomic ID can appear in multiple rows, if there should be multiple
    accessions used for it.

    Args:
        fn: path to TSV file, where each row corresponds to an accession
            to include in the design

    Returns:
        dict {(taxonomic-id, segment): [list of accessions]}
    """
    accs = defaultdict(list)
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            if len(ls) != 3:
                raise Exception(("Input accession TSV must have 3 columns"))

            try:
                tax_id = int(ls[0])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[0])

            segment = ls[1]
            if segment.lower() == 'none':
                segment = None
            accession = ls[2]
            accs[(tax_id, segment)].append(accession)
    return accs


def read_sequences_for_taxonomies(fn):
    """Read sequences for different taxonomies

    The columns must be, in order:
        1) a taxonomic (e.g., species) ID from NCBI
        2) a segment label, or 'None' if unsegmented
        3) a path to a FASTA file containing sequences

    Args:
        fn: path to TSV file, where each row corresponds to a taxonomy

    Returns:
        dict {(taxonomic-id, segment): dict of sequences}
    """
    seqs = defaultdict(list)
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            if len(ls) != 3:
                raise Exception(("Input fasta TSV must have 3 columns"))

            try:
                tax_id = int(ls[0])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[0])

            segment = ls[1]
            if segment.lower() == 'none':
                segment = None

            fasta_path = ls[2]
            seqs_for_tax = read_fasta(fasta_path)
            seqs[(tax_id, segment)] = seqs_for_tax
    return seqs


def read_taxonomies_to_design_for(fn):
    """Read different taxonomies to design for.

    The columns must be, in order:
        1) a taxonomic (e.g., species) ID from NCBI
        2) a segment label, or 'None' if unsegmented

    Args:
        fn: path to TSV file, where each row corresponds to a taxonomy

    Returns:
        collection of (taxonomic-id, segment)
    """
    taxs = []
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            if len(ls) != 2:
                raise Exception(("Input fasta TSV must have 2 columns"))

            try:
                tax_id = int(ls[0])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[0])

            segment = ls[1]
            if segment.lower() == 'none':
                segment = None

            taxs += [(tax_id, segment)]
    return taxs


def read_taxonomy_specificity_ignore(fn):
    """Read taxonomies to ignore when enforcing specificity.

    The columns must be, in order:
      1) a taxonomic ID for taxonomy A
      2) a taxonomic ID such that this taxonomy should be ignored when
         designing for A, in terms of specificity

    Args:
        fn: path to TSV file

    Returns:
        dict {A: {B}} where {B} consists of the taxonomies that should
        be ignored for A
    """
    tax_ignore = defaultdict(set)
    with open(fn) as f:
        for line in f:
            ls = line.rstrip().split('\t')

            if len(ls) != 2:
                raise Exception(("Input TSV must have 2 columns"))

            try:
                tax_id_a = int(ls[0])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[0])
            try:
                tax_id_b = int(ls[1])
            except ValueError:
                raise Exception(("Taxonomy ID '%s' must be an integer") %
                    ls[1])

            tax_ignore[tax_id_a].add(tax_id_b)
    return tax_ignore


def read_metadata_filters(meta_filts):
    """Create dictionaries of metadata filters from a list.
    
    Args:
        meta_filts: list of filters in the format 'key=values' or 
            'key!=values' with a comma separated list of values

    Returns:
        tuple of 2 dicts where each key is any of ['taxid', 'year', 'country']
            and each value is what to include for the first dict and exclude for
            the second dict
    """
    dict_filter_eq = {}
    dict_filter_neq = {}
    for filt in meta_filts:
        if ' ' in filt:
            raise ValueError("Incorrect format for filter '%s'; individual filters "
                "should not include spaces" %filt)
        dict_to_use = dict_filter_eq
        filt_split = filt.split('!=')
        if len(filt_split) == 2:
            dict_to_use = dict_filter_neq
        else:
            filt_split = filt.split('=')
        if len(filt_split) != 2:
            raise ValueError("Incorrect format for filter '%s'; should be "
                "'metadata=value or metadata!=value'" %filt)
        if filt_split[0] not in ['taxid', 'year', 'country']:
            raise ValueError("Incorrect filter key '%s'; should be one of "
                "['taxid', 'year', 'country']" %filt_split[0])
        if filt_split[0] in ['taxid', 'year']:
            dict_to_use[filt_split[0]] = [int(i) for i in filt_split[1].split(",")]
        else:
            dict_to_use[filt_split[0]] = filt_split[1].split(",")
    return (dict_filter_eq, dict_filter_neq)

