#!/usr/bin/env python3
"""Design probes for diagnostics."""

import argparse
import logging

from probedesign import alignment
from probedesign import probe_search
from probedesign.utils import log
from probedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'


def main(args):
    logger = logging.getLogger(__name__)

    # Read the sequences and make an Alignment object
    seqs = seq_io.read_fasta(args.in_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Find the optimal set of probes
    ps = probe_search.ProbeSearcher(aln, args.probe_length, args.mismatches,
                                    args.window_size, args.cover_frac)
    probes = ps.find_probes_that_cover()

    if len(probes) == 1:
        print("The 1 probe is:")
    else:
        print("The %d probes are:" % len(probes))
    for prb_seq in probes:
        print("  %s" % prb_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_fasta', help="Path to input fasta")
    parser.add_argument('-l', '--probe_length', type=int, default=28,
                        help="Length of probe to construct")
    parser.add_argument('-m', '--mismatches', type=int, default=0,
                        help=("Allow for this number of mismatches when "
                              "determining whether a probe covers a sequence"))
    parser.add_argument('-w', '--window_size', type=int, default=200,
                        help=("Ensure that selected probes are all within a "
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
                              "by the selected probes"))
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
