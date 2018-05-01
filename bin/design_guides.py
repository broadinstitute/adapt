#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
import logging

from dxguidedesign import alignment
from dxguidedesign import guide_search
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

def main(args):
    logger = logging.getLogger(__name__)

    # Read the sequences and make an Alignment object
    seqs = seq_io.read_fasta(args.in_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Find all strong primers in the alignment
    ps = guide_search.PrimerSearcher(aln, args.guide_length, args.mismatches,
                                        args.window_size, args.cover_frac,
                                        args.primer_length, args.primer_mismatches,
                                        args.primer_window_size)

    # From primers, construct list of windows, return those as the `primers` object
    primers = ps.find_primers_that_cover(args.out_tsv, sort=args.sort_out)

    # Set up the guide search
    gs = guide_search.GuideSearcher(aln, args.guide_length, args.mismatches,
                                    args.window_size, args.cover_frac,
                                    args.primer_length, args.primer_mismatches,
                                    args.primer_window_size)
    if len(primers) < 2:
        print("Not enough primers identified. No guides could be constructed")
        exit(1)

    # Take the candidate primer windows and use those genomic coordinates to invoke GuideSearcher
    for primer_slice in primers:
        if primer_slice == primers[0]:
            gs.find_guides_that_cover(args.out_tsv, primer_slice, sort=args.sort_out, first_slice=True)
        else:
            gs.find_guides_that_cover(args.out_tsv, primer_slice, sort=args.sort_out)

    # Filter out any duplicated entries or suboptimal amplicons
    gs.clean_up_output(out_fn=args.out_tsv)

    # Visualize the target amplicons
    gs.plot(args.out_tsv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_fasta', help="Path to input fasta")
    parser.add_argument('out_tsv', help="Path to output TSV")
    parser.add_argument('-l', '--guide_length', type=int, default=28,
                        help="Length of guide to construct")
    parser.add_argument('-m', '--mismatches', type=int, default=0,
                        help=("Allow for this number of mismatches when "
                              "determining whether a guide covers a sequence"))
    parser.add_argument('-w', '--window_size', type=int, default=200,
                        help=("Maximum amplicon size. Needs to be long enough"
                              "to include forward-primer, guide, and reverse-primer"))
    # Add in the primer arguments
    parser.add_argument('-pl', '--primer_length', type=int, default=28,
                        help="Length of each primer to construct")
    parser.add_argument('-pm', '--primer_mismatches', type=int, default=0,
                        help='Number of mistmatches to tolerate in each primer')
    parser.add_argument('-pw', '--primer_window_size', type=int, default=28,
                        help=('Set the window size in which one primer will be found '
                              'Ex: If -pl=28, -pw=50, the strongest 28-nt primer will ' 
                              'be found in the 50-nt window '))
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
