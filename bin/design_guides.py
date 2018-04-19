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

    # Testing: remove old guides file
    import os
    os.system('rm short.guides.tsv && rm primer_positions.txt && rm primer_short.guides.tsv')

    # Read the sequences and make an Alignment object
    seqs = seq_io.read_fasta(args.in_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Find all strong primers in the alignment
    ps = guide_search.PrimerSearcher(aln, args.guide_length, args.mismatches,
                                        args.window_size, args.cover_frac,
                                        args.primer_length, args.primer_mismatches,
                                        args.primer_window_size)

    # From primers, construct list of windows, output those to primer_obj
    primer_obj = ps.find_primers_that_cover(args.out_tsv, sort=args.sort_out)
    print(primer_obj)

    # Find an optimal set of guides for each window in the genome,
    # and write them to a file
    gs = guide_search.GuideSearcher(aln, args.guide_length, args.mismatches,
                                    args.window_size, args.cover_frac,
                                    args.primer_length, args.primer_mismatches,
                                    args.primer_window_size)
    counter=0
    for slice in primer_obj:
        if counter >= 0:
            gs.find_guides_that_cover(args.out_tsv, slice, sort=args.sort_out)
            counter+=1
    gs.clean_up_output(out_fn=args.out_tsv)

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
                        help=("Ensure that selected guides are all within a "
                              "window of this size"))
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

"""Improvements:
    
    Add a PrimerFinder step that feeds into the GuideFinder step.
    
    When I ran this on Norovirus, it output the same optimal guide like 30 times for the same region.
    This happens because it iterates by 1 nt windows (which is great).
    But instead of displaying the same guide 30 times, we should just display it once and give it some
    kind of really high confidence score.
    
    Ways to code this: If the start position of window2 is within window1 AND guide1 and guide2 are identical,
    score this well but don't display both.
"""