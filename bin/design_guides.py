#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
import logging

from dxguidedesign import alignment
from dxguidedesign import guide_search
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'


def design_independently(args):
    # Treat each alignment independently
    for in_fasta, out_tsv in zip(args.in_fasta, args.out_tsv):
        # Read the sequences and make an Alignment object
        seqs = seq_io.read_fasta(in_fasta)
        aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file
        gs = guide_search.GuideSearcher(aln, args.guide_length, args.mismatches,
                                        args.window_size, args.cover_frac,
                                        args.missing_thres)
        gs.find_guides_that_cover(out_tsv, sort=args.sort_out)


def design_for_id(args):
    pass 


def main(args):
    logger = logging.getLogger(__name__)

    if len(args.in_fasta) != len(args.out_tsv):
        raise Exception("Number output TSVs must match number of input FASTAs")

    if args.diff_id:
        design_for_id(args)
    else:
        design_independently(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_fasta', nargs='+',
                        help=("Path to input FASTA. More than one can be "
                              "given; without --id specified, this just "
                              "outputs guides independently for each alignment"))
    parser.add_argument('-o', '--out_tsv', nargs='+', required=True,
                        help=("Path to output TSV. If more than one input "
                              "FASTA is given, the same number of output TSVs "
                              "must be given; each output TSV corresponds to "
                              "an input FASTA."))
    parser.add_argument('--id', dest="diff_id", action='store_true',
                        help=("Design guides to perform differential "
                              "identification, where each input FASTA is a "
                              "group/taxon to identify with specificity"))
    parser.add_argument('-l', '--guide_length', type=int, default=28,
                        help="Length of guide to construct")
    parser.add_argument('-m', '--mismatches', type=int, default=0,
                        help=("Allow for this number of mismatches when "
                              "determining whether a guide covers a sequence"))
    parser.add_argument('-w', '--window_size', type=int, default=200,
                        help=("Ensure that selected guides are all within a "
                              "window of this size"))
    def check_cover_frac(val):
        fval = float(val)
        if fval > 0 and fval <= 1:
            # a float in (0, 1]
            return fval
        else:
            raise argparse.ArgumentTypeError("%s is an invalid -p value" % val)
    parser.add_argument('-p', '--cover_frac', type=check_cover_frac, default=1.0,
                        help=("The fraction of sequences that must be covered "
                              "by the selected guides"))
    parser.add_argument('--sort', dest='sort_out', action='store_true',
                        help=("If set, sort output TSV by number of guides "
                              "(ascending) then by score (descending); "
                              "default is to sort by window position"))
    parser.add_argument('--missing-thres', nargs=3, type=float, default=[0.5, 0.05, 1.5],
        help=("Parameters governing the threshold on which sites to ignore "
              "due to too much missing data. Two values (a, b, c) specifying "
              "not to attempt to design guides overlapping sites where the "
              "fraction of sequences with missing data is > min(a, max(b, c*m) "
              "where m is the median fraction of sequences with missing data "
              "over the alignment. Set a=1 and b=1 to not ignore sites due "
              "to missing data."))
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
