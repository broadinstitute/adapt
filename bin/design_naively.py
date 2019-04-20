#!/usr/bin/env python3
"""Design guides naively for diagnostics.

This can be used as a baseline for the output of design.py.
"""

import argparse
import logging

from dxguidedesign import alignment
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def construct_guide_naively_at_each_pos(aln, args):
    """Naively construct a guide sequence at each position of an alignment.

    This constructs a guide sequence at each position. It does so in two
    ways: 'consensus' (the consensus sequence at the position) and
    'mode' (the most common sequence at the position).

    Args:
        aln: alignment.Alignment object
        args: arguments to program

    Returns:
        list x where x[i] is a dict, giving a guide sequence at position
        i of the alignment. x[i][method] is a tuple (guide, frac),
        where method is 'consensus' or 'mode', and guide gives a guide
        sequence (a string) constructed using the method and frac is
        the fraction of sequences in the alignment to which the guide
        binds
    """
    start_positions = range(aln.seq_length - args.guide_length + 1)
    guides = [None for _ in start_positions]
    for i in start_positions:
        # Extract the portion of the alignment that starts at i
        pos_start, pos_end = i, i + args.guide_length
        aln_for_guide = aln.extract_range(pos_start, pos_end)

        # When constructing guides, ignore any sequences in the alignment
        # that have a gap in this region
        seqs_with_gap = set(aln_for_guide.seqs_with_gap())
        seqs_to_consider = set(range(aln.num_sequences)) - seqs_with_gap

        # Construct guides
        consensus_guide = aln_for_guide.determine_consensus_sequence(
                seqs_to_consider=seqs_to_consider)
        mode_guide = aln_for_guide.determine_most_common_sequence(
                seqs_to_consider=seqs_to_consider, skip_ambiguity=True)

        # Determine the fraction of the sequences that each guide binds to
        consensus_guide_bound = aln_for_guide.sequences_bound_by_guide(
                consensus_guide, 0, args.guide_mismatches,
                args.allow_gu_pairs)
        consensus_guide_frac = float(len(consensus_guide_bound)) / aln.num_sequences
        mode_guide_bound = aln_for_guide.sequences_bound_by_guide(
                mode_guide, 0, args.guide_mismatches,
                args.allow_gu_pairs)
        mode_guide_frac = float(len(mode_guide_bound)) / aln.num_sequences

        d = {'consensus': (consensus_guide, consensus_guide_frac),
             'mode': (mode_guide, mode_guide_frac)}
        guides[i] = d
    return guides


def find_guide_in_each_window(guides, aln_length, args):
    """Determine a guide for each window of an alignment.

    For each window, this selects the guide within it (given one
    guide per position) that covers the highest fraction of sequences.

    To break ties, this selects the first guide (in terms of
    position) among ones with a tied fraction covered.

    Args:
        guides: list such that guides[i] is a tuple (guide, frac)
            giving a guide sequence (guide) at position i of
            an alignment and the fraction of sequences in the
            alignment (frac) covered by the guide
        aln_length: length of alignment
        args: arguments to program

    Returns:
        list x where x[i] gives a tuple (guide, frac) representing a
        guide in the window that starts at position i
    """
    window_start_positions = range(aln_length - args.window_size + 1)
    guide_in_window = [None for _ in window_start_positions]
    best_guide_seq, best_guide_frac, best_guide_pos = None, -1, -1
    for i in window_start_positions:
        window_start, window_end = i, i + args.window_size
        last_guide_pos = window_end - args.guide_length
        
        if best_guide_pos < window_start:
            # The best guide is no longer in the window; find
            # a new one
            best_guide_seq, best_guide_frac, best_guide_pos = None, -1, -1
            for j in range(window_start, last_guide_pos + 1):
                guide, frac = guides[j]
                if frac > best_guide_frac:
                    best_guide_seq = guide
                    best_guide_frac = frac
                    best_guide_pos = j
        else:
            # The last best guide is still within the window, but now
            # check if the new guide at the very end of the window
            # does better
            guide, frac = guides[last_guide_pos]
            if frac > best_guide_frac:
                best_guide_seq = guide
                best_guide_frac = frac
                best_guide_pos = last_guide_pos

        # Save the best guide for the current window
        guide_in_window[i] = (best_guide_seq, best_guide_frac)
    return guide_in_window


def main(args):
    # Allow G-U base pairing, unless it is explicitly disallowed
    args.allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Read the input alignment
    seqs = seq_io.read_fasta(args.in_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Construct a guide at each position of the alignment
    guides = construct_guide_naively_at_each_pos(aln, args)

    # Find the best guide in each window (for both the
    # consensus and mode approach)
    consensus_guides_in_window = find_guide_in_each_window(
            [guides[i]['consensus'] for i in range(len(guides))],
            aln.seq_length, args)
    mode_guides_in_window = find_guide_in_each_window(
            [guides[i]['mode'] for i in range(len(guides))],
            aln.seq_length, args)

    # Write the guides to a TSV file
    with open(args.out_tsv, 'w') as outf:
        header = ['window-start', 'window-end',
                'target-sequence-by-consensus', 'frac-bound-by-consensus',
                'target-sequence-by-mode', 'frac-bound-by-mode']
        outf.write('\t'.join(header) + '\n')
        for i in range(len(consensus_guides_in_window)):
            window_start, window_end = i, i + args.window_size
            consensus_guide_seq, consensus_guide_frac = \
                    consensus_guides_in_window[i]
            mode_guide_seq, mode_guide_frac = mode_guides_in_window[i]
            line = [window_start, window_end,
                    consensus_guide_seq, consensus_guide_frac,
                    mode_guide_seq, mode_guide_frac]
            outf.write('\t'.join([str(x) for x in line]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input alignment and output file
    parser.add_argument('in_fasta',
            help=("Path to input FASTA (aligned)"))
    parser.add_argument('out_tsv',
            help=("Path to TSV file to which to write the output"))

    # Window size
    parser.add_argument('-w', '--window-size', type=int, default=200,
            help=("Output a guide within each window (sliding along "
                  "the alignment) of this length"))

    # Parameters on guide length and mismatches
    parser.add_argument('-gl', '--guide-length', type=int, default=28,
            help="Length of guide to construct")
    parser.add_argument('-gm', '--guide-mismatches', type=int, default=0,
            help=("Allow for this number of mismatches when "
                  "determining whether a guide covers a sequence"))

    # G-U pairing options
    parser.add_argument('--do-not-allow-gu-pairing', action='store_true',
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
