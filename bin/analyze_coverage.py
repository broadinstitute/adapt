#!/usr/bin/env python3
"""Analyze coverage obtained by designs."""

import argparse
import logging

from dxguidedesign import coverage_analysis
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def read_designs(fn):
    """Read a collection of targets from a file.

    Each target represents a design.

    Args:
        fn: path to a TSV file giving targets

    Returns:
        dict {design_id: design} where design_id is the row number
        of the target in fn (starting at 1 for the first target) and
        design is a coverage_analysis.Design object
    """
    rows = []
    with open(fn) as f:
        col_names = {}
        for i, line in enumerate(f):
            line = line.rstrip()
            ls = line.split('\t')
            if i == 0:
                # Parse header
                for j in range(len(ls)):
                    col_names[j] = ls[j]
            else:
                # Read each column as a variable
                cols = {}
                for j in range(len(ls)):
                    cols[col_names[j]] = ls[j]
                rows += [cols]

    designs = {}
    for i, row in enumerate(rows):
        cols = row
        if 'target-sequences' in cols:
            # This design only contains guides
            design = coverage_analysis.Design(
                    cols['target-sequences'].split(' '))
        else:
            # This design contains primers and guides
            design = coverage_analysis.Design(
                    cols['guide-target-sequences'].split(' '),
                    (cols['left-primer-target-sequences'].split(' '),
                     cols['right-primer-target-sequences'].split(' ')))
        designs[i + 1] = design
    return designs


def write_frac_bound(designs, frac_bound, out_fn):
    """Write table giving the fraction of sequences bound by a target.

    Args:
        designs: dict {design_id: design} where design_id is an
            identifier for design and design is a coverage_analysis.
            Design object
        frac_bound: dict {design_id: frac_bound} where design_id is
            an identifier for a design and frac_bound is the fraction
            of target sequences that are bound by the design
        out_fn: path to TSV file at which to write table
    """
    header = ['design_id',
              'guide-target-sequences',
              'left-primer-target-sequences',
              'right-primer-target-sequences',
              'frac-bound']
    with open(out_fn, 'w') as fw:
        fw.write('\t'.join(header) + '\n')
        for design_id in sorted(list(designs.keys())):
            guides = designs[design_id].guides
            guides = ' '.join(sorted(guides))
            if designs[design_id].primers is not None:
                left_primers, right_primers = designs[design_id].primers
                left_primers = ' '.join(sorted(left_primers))
                right_primers = ' '.join(sorted(right_primers))
            else:
                left_primers, right_primers = 'n/a', 'n/a'
            row = [design_id, guides, left_primers, right_primers,
                   frac_bound[design_id]]
            fw.write('\t'.join([str(x) for x in row]) + '\n')


def main(args):
    # Allow G-U base pairing, unless it is explicitly disallowed
    allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Read the designs
    designs = read_designs(args.designs_fn)

    # Read the input sequences to compute coverage against; use
    # skip_gaps=True so that, if an alignment is input, this
    # is read as unaligned sequences
    seqs = seq_io.read_fasta(args.seqs_fn, skip_gaps=True)

    analyzer = coverage_analysis.CoverageAnalyzer(
            seqs, designs, args.guide_mismatches, args.primer_mismatches,
            allow_gu_pairs)

    performed_analysis = False
    if args.write_frac_bound:
        frac_bound = analyzer.frac_of_seqs_bound()
        write_frac_bound(designs, frac_bound, args.write_frac_bound)
        performed_analysis = True

    if not performed_analysis:
        logger.warning(("No analysis was requested"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required inputs
    parser.add_argument('designs_fn',
        help=("Path to output of running design.py; a TSV file where "
              "each row contains a design (target)"))
    parser.add_argument('seqs_fn',
        help=("Path to FASTA file giving sequences against which to "
              "compute coverage"))

    # Analyses to output
    parser.add_argument('--write-frac-bound',
        help=("If set, write a table in which each row represents an "
              "input design and gives the fraction of all sequences that "
              "are covered by the design. The 'design_id' column gives "
              "the row number of the design in the designs input (1 for "
              "the first design). The provided argument is a path to "
              "a TSV file at which to the write the table."))

    # Parameters determining whether a sequence binds to target
    parser.add_argument('-gm', '--guide-mismatches',
        type=int, default=0,
        help=("allow for this number of mismatches when "
              "determining whether a guide covers a sequence"))
    parser.add_argument('-pm', '--primer-mismatches',
        type=int, default=0,
        help=("Allow for this number of mismatches when determining "
              "whether a primer covers a sequence (ignore this if "
              "the targets only consist of guides)"))
    parser.add_argument('--do-not-allow-gu-pairing',
        action='store_true',
        help=("When determining whether a guide binds to a region of "
              "target sequence, do not count G-U (wobble) base pairs as "
              "matching. Default is to tolerate G-U pairing: namely, "
              "A in an output guide sequence matches G in the "
              "target and C in an output guide sequence matches T "
              "in the target (since the synthesized guide is the reverse "
              "complement of the output guide sequence)"))

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
