#!/usr/bin/env python3
"""Pick targets recommended for testing."""

import argparse
import logging

from adapt import alignment
from adapt.prepare import cluster
from adapt import target_search
from adapt.utils import formatting
from adapt.utils import guide
from adapt.utils import log
from adapt.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def find_test_targets(design_target, aln, args):
    """Find collection of test target sequences for a design option.

    Note that there is probably going to be some confusion with the word
    'target' here. It refers both to a design option (in the TSV written
    by design.py), as well as a sequence that should be used for testing.

    Args:
        design_target: target_search.DesignTarget object, representing
            a design option (amplicon)
        aln: alignment of genomes; positions correspond to positions in
            design_target
        args: parsed arguments

    Returns:
        collection of tuples (s, s_tagged) where s is a target sequence
        to use for testing and s_tagged is s tagged to show where
        primers and guides may bind s
    """
    logger.info(("Finding test targets for design option with endpoints "
        "[%d, %d)"), design_target.target_start, design_target.target_end)
    
    # Expand the extracted range to the minimum
    # Note that this extracts to get a target length of args.min_target_len
    # *in the alignment*; if there are gaps in the alignment within the
    # region, the actual length of the sequences could be shorter
    target_len = design_target.target_end - design_target.target_start
    nt_to_add = int(max(0, args.min_target_len - target_len) / 2)
    target_start = max(0, design_target.target_start - nt_to_add)
    target_end = min(aln.seq_length, design_target.target_end + nt_to_add)
    if target_end - target_start == args.min_target_len - 1:
        # Fix off-by-1 edge case
        target_end = min(aln.seq_length, target_end + 1)

    # Extract the alignment where this design target (amplicon) is
    aln_extract = aln.extract_range(target_start, target_end)
    # Pull out the sequences, without gaps
    aln_extract_seqs = aln_extract.make_list_of_seqs(remove_gaps=True)

    # Add indices for each sequence so it can be used as a dict
    aln_extract_seqs_dict = {i: s for i, s in enumerate(aln_extract_seqs)}

    # The number of k-mers to use with MinHash for clustering cannot be
    # more than the number of k-mers in a sequence
    minhash_k = 12
    min_seq_len = min(len(s) for s in aln_extract_seqs)
    minhash_N = min(50, min_seq_len - minhash_k - 1)

    # Find representative sequences
    rep_seqs_idx = cluster.find_representative_sequences(aln_extract_seqs_dict,
            k=minhash_k, N=minhash_N, threshold=0.1)
    rep_seqs = [aln_extract_seqs[i] for i in rep_seqs_idx]

    # Find where primers and guides overlap each representative sequence
    guide_allow_gu_pairs = True
    if args.do_not_allow_gu_pairing:
        guide_allow_gu_pairs = False
    rep_seqs_tagged = []
    for rep_seq in rep_seqs:
        primer_seqs = (design_target.left_primer_seqs +
                design_target.right_primer_seqs)
        primer_overlap = guide.guide_overlap_in_seq(primer_seqs,
                rep_seq, args.pm, False)
        guide_seqs = design_target.guide_seqs
        guide_overlap = guide.guide_overlap_in_seq(guide_seqs,
                rep_seq, args.gm, guide_allow_gu_pairs)
        overlap_labels = {'primer': primer_overlap, 'guide': guide_overlap}

        rep_seq_tagged = formatting.tag_seq_overlap(overlap_labels, rep_seq)
        rep_seqs_tagged += [rep_seq_tagged]

    return list(zip(rep_seqs, rep_seqs_tagged))


def main(args):
    # Read the design targets
    targets = target_search.DesignTarget.read_design_targets(args.design_tsv)

    # Read the alignment
    seqs = seq_io.read_fasta(args.alignment_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Find and write test targets
    with open(args.out_tsv, 'w') as f:
        def write_row(row):
            line = '\t'.join(str(x) for x in row)
            f.write(line + '\n')
        header = ['target-design-start', 'target-design-end',
                'test-target-seq', 'test-target-seq-tagged']
        write_row(header)
        for design_target in targets:
            targets_to_test = find_test_targets(design_target, aln, args)
            for tt in targets_to_test:
                row = [design_target.target_start, design_target.target_end,
                        tt[0], tt[1]]
                write_row(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('design_tsv',
        help=("Path to TSV with complete targets output by design.py"))
    parser.add_argument('alignment_fasta',
        help=("Path to alignment (FASTA) used for design of DESIGN_TSV (e.g., "
              "as output by design.py --write-input-aln"))
    parser.add_argument('out_tsv',
        help=("Path to output TSV with recommended targets for testing"))

    parser.add_argument('-pm',
        type=int, default=0,
        help=("Number of mismatches to tolerate when determining whether "
              "primer binds to a region of target sequence"))
    parser.add_argument('-gm',
        type=int, default=0,
        help=("Number of mismatches to tolerate when determining whether "
              "guide binds to a region of target sequence"))
    parser.add_argument('--do-not-allow-gu-pairing', action='store_true',
        help=("When determining whether a guide binds to a region of target "
              "sequence, do not count G-U (wobble) base pairs as matching."))

    parser.add_argument('--min-target-len',
        type=int, default=0,
        help=("Minimum length of a target region; if the region "
              "in a design bound by primers is less than this, sequence "
              "will be added on both sides of the primer to reach this "
              "length. Note that this is in the alignment; the actual "
              "sequence could be shorter if there are gaps in the "
              "alignment"))

    # Log levels
    parser.add_argument("--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.WARNING,
        help=("Debug output"))
    parser.add_argument("--verbose",
        dest="log_level",
        action="store_const",
        const=logging.INFO,
        help=("Verbose output"))

    args = parser.parse_args()
    log.configure_logging(args.log_level)

    main(args)