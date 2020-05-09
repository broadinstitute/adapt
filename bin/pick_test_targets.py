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
        collection of tuples (s, s_tagged, f) where s is a target sequence
        to use for testing; s_tagged is s tagged to show where
        primers and guides may bind s; and f is the fraction of all sequences
        that s represents
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

    # Remove target sequences that are too short
    # This can happen due to gaps in the alignment; a sequence can have
    # length 0, for example, if is it all '-' in the amplicon (extracted
    # range)
    # If they are extremely short (shorter than minhash_k, below), then
    # this will cause an error downstream
    aln_extract_seqs = [s for s in aln_extract_seqs
            if len(s) >= args.min_seq_len_to_consider]

    # Add indices for each sequence so it can be used as a dict
    aln_extract_seqs_dict = {i: s for i, s in enumerate(aln_extract_seqs)}

    # The number of k-mers to use with MinHash for clustering cannot be
    # more than the number of k-mers in a sequence
    minhash_k = 12
    min_seq_len = min(len(s) for s in aln_extract_seqs)
    minhash_N = min(50, min_seq_len - minhash_k - 1)

    # Find representative sequences
    if args.num_representative_targets:
        # Use a maximum number of clusters, ignoring the inter-cluster
        # distance threshold
        threshold = None
        num_clusters = args.num_representative_targets

        if args.min_frac_to_cover_with_rep_seqs < 1.0:
            logger.warning(("Fewer than %d targets may be reported "
                "because --min-frac-to-cover-with-rep-seqs is <1.0; set "
                "it to 1.0 to obtain %d representative targets") %
                (num_clusters, num_clusters))
    else:
        # Use an inter-cluster distance threhsold
        threshold = args.max_cluster_distance
        num_clusters = None
    rep_seqs_idx, rep_seqs_frac = cluster.find_representative_sequences(
            aln_extract_seqs_dict,
            k=minhash_k, N=minhash_N, threshold=threshold,
            num_clusters=num_clusters,
            frac_to_cover=args.min_frac_to_cover_with_rep_seqs)
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

        if len(primer_overlap) == 0 and len(guide_overlap) == 0:
            logger.warning(("Neither primers nor guides could be found "
                "in a representative target sequence; it is possible that "
                "the representative is from the wrong region, or it is "
                "too divergent from the primer/guide sequences (for the "
                "latter, try raising -pm and -gm). The representative target "
                "sequence is for a design from the window [%d, %d) of the input "
                "alignment. This only matters for tagging the target, and it "
                "is ok to ignore this message.") % (design_target.target_start,
                    design_target.target_end))
        elif len(primer_overlap) == 0:
            logger.warning(("Primers could not be found in a representative "
                "target sequence; it is possible that the representative "
                "is from the wrong region, or it is too divergent from the "
                "primer sequences (for the latter, try raising -pm). The "
                "representative target sequence is for a design from the "
                "window [%d, %d) of the input alignment. This only matters "
                "for tagging the target, and it is ok to ignore this message.") %
                (design_target.target_start, design_target.target_end))
        elif len(guide_overlap) == 0:
            logger.warning(("Guides could not be found in a representative "
                "target sequence; it is possible that the representative "
                "is from the wrong region, or it is too divergent from the "
                "guide sequences (for the latter, try raising -gm). The "
                "representative target sequence is for a design from the "
                "window [%d, %d) of the input alignment. This only matters "
                "for tagging the target, and it is ok to ignore this message.") %
                (design_target.target_start, design_target.target_end))

        rep_seq_tagged = formatting.tag_seq_overlap(overlap_labels, rep_seq)
        rep_seqs_tagged += [rep_seq_tagged]

    return list(zip(rep_seqs, rep_seqs_tagged, rep_seqs_frac))


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
        header = ['design-target-start', 'design-target-end',
                'test-target-frac-represented',
                'test-target-seq', 'test-target-seq-tagged']
        write_row(header)
        for design_target in targets:
            targets_to_test = find_test_targets(design_target, aln, args)
            for tt in targets_to_test:
                row = [design_target.target_start, design_target.target_end,
                        tt[2], tt[0], tt[1]]
                write_row(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('design_tsv',
            help=("Path to TSV with design options, as output by design.py. "
                  "Alternatively, it can be a custom-made TSV, with one "
                  "row per design option, containing the following columns: "
                  "'target-start' (5' end of a targeted genomic region, e.g., "
                  "an amplicon); 'target-end' (3' end); 'guide-target-"
                  "sequences' (space-separated list of guide sequences); "
                  "'left-primer-target-sequences' (space-separated list of "
                  "forward primer sequences); 'right-primer-target-"
                  "sequences' (space-separated list of reverse primer "
                  "sequences)"))
    parser.add_argument('alignment_fasta',
        help=("Path to alignment (FASTA) from which to extract targets. "
              "Target positions in DESIGN_TSV must correspond to positions "
              "in this alignment. If DESIGN_TSV is the output of design.py, "
              "then this can be the output of `design.py --write-input-aln`."))
    parser.add_argument('out_tsv',
        help=("Path to output TSV with recommended targets for testing"))

    parser.add_argument('-pm',
        type=int, default=5,
        help=("Number of mismatches to tolerate when determining whether "
              "primer binds to a region of target sequence"))
    parser.add_argument('-gm',
        type=int, default=3,
        help=("Number of mismatches to tolerate when determining whether "
              "guide binds to a region of target sequence"))
    parser.add_argument('--do-not-allow-gu-pairing', action='store_true',
        help=("When determining whether a guide binds to a region of target "
              "sequence, do not count G-U (wobble) base pairs as matching."))

    parser.add_argument('--min-seq-len-to-consider',
        type=int, default=80,
        help=("Do not consider, when identifying representative sequences, "
              "target sequences that are shorter than this length. These "
              "can occur due to gaps in the alignment (e.g., a target "
              "sequence can have length 0 if it is all '-' in the amplicon."))
    parser.add_argument('--min-target-len',
        type=int, default=500,
        help=("Minimum length of a target region; if the region "
              "in a design bound by primers is less than this, sequence "
              "will be added on both sides of the primer to reach this "
              "length. Note that this is in the alignment; the actual "
              "sequence could be shorter if there are gaps in the "
              "alignment"))
    parser.add_argument('--min-frac-to-cover-with-rep-seqs',
        type=float, default=0.95,
        help=("For representative sequences, use medoids of clusters such "
              "that the clusters account for at least this fraction of all "
              "sequences. This allows ignoring outlier clusters (whose "
              "sequence(s) may have not been covered by the design."))

    parser.add_argument('--max-cluster-distance',
        type=float, default=0.1,
        help=("Maximum inter-cluster distance to merge clusters (measured "
              "as 1-ANI). Higher values result in fewer representative "
              "targets."))
    parser.add_argument('--num-representative-targets',
        type=int,
        help=("Maximum number of clusters (equivalent to maximum number "
              "of representative targets). If set, then "
              "--max-cluster-distance is ignored. Note that fewer may "
              "be reported if --min-frac-to-cover-with-rep-seqs is "
              "<1.0; set it to 1.0 to report all."))

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
