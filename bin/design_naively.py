#!/usr/bin/env python3
"""Design guides naively for diagnostics.

This can be used as a baseline for the output of design.py.
"""

import argparse
import logging
import heapq

from adapt import alignment
from adapt.utils import log
from adapt.utils import seq_io

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


def construct_guide_naively_at_each_pos(aln, args, ref_seq=None):
    """Naively construct a guide sequence at each position of an alignment.

    This constructs a guide sequence at each position. It does so in two
    ways: 'consensus' (the consensus sequence at the position) and
    'mode' (the most common sequence at the position).

    Args:
        aln: alignment.Alignment object
        args: arguments to program
        ref_seq: reference sequence to base diversity guides on

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
    ref_seq_aln = alignment.Alignment.from_list_of_seqs([ref_seq])
    for i in start_positions:
        # Extract the portion of the alignment that starts at i
        pos_start, pos_end = i, i + args.guide_length
        aln_for_guide = aln.extract_range(pos_start, pos_end)

        # When constructing guides, ignore any sequences in the alignment
        # that have a gap in this region. 
        seqs_with_gap = set(aln_for_guide.seqs_with_gap())
        seqs_to_consider = set(range(aln.num_sequences)) - seqs_with_gap
        # Only look at sequences with valid flanking regions
        seqs_to_consider = aln.seqs_with_required_flanking(i, args.guide_length,
                args.required_flanking_seqs, seqs_to_consider=seqs_to_consider)
        ref_seqs_to_consider = ref_seq_aln.seqs_with_required_flanking(
                i, args.guide_length, args.required_flanking_seqs)

        consensus_guide = None
        mode_guide = None
        diversity_guide = None

        frac_with_gap = float(len(seqs_with_gap)) / aln.num_sequences
        # Do not bother designing a guide here; there are
        # too many sequences with a gap
        if frac_with_gap < args.skip_gaps:
            # Construct guides
            # Only design guides if there exist sequences to consider
            if len(seqs_to_consider) > 0:
                if args.consensus:
                    consensus_guide = aln_for_guide.determine_consensus_sequence(
                            seqs_to_consider=seqs_to_consider)
                if args.mode:
                    mode_guide = aln_for_guide.determine_most_common_sequence(
                            seqs_to_consider=seqs_to_consider, skip_ambiguity=True) \

            if len(ref_seqs_to_consider) > 0:
                if args.diversity:
                    diversity_guide = ref_seq[pos_start:pos_end]

        # Determine the fraction of the sequences that each guide binds to
        if consensus_guide is not None:
            consensus_guide_bound = aln.sequences_bound_by_guide(
                    consensus_guide, i, args.guide_mismatches,
                    args.allow_gu_pairs, required_flanking_seqs=args.required_flanking_seqs)
            consensus_guide_frac = float(len(consensus_guide_bound)) / aln.num_sequences
        else:
            consensus_guide = 'None'
            consensus_guide_frac = 0

        if mode_guide is not None:
            mode_guide_bound = aln.sequences_bound_by_guide(
                    mode_guide, i, args.guide_mismatches,
                    args.allow_gu_pairs, required_flanking_seqs=args.required_flanking_seqs)
            mode_guide_frac = float(len(mode_guide_bound)) / aln.num_sequences
        else:
            mode_guide = 'None'
            mode_guide_frac = 0

        if diversity_guide is not None:
            diversity_guide_bound = aln.sequences_bound_by_guide(
                diversity_guide, i, args.guide_mismatches,
                args.allow_gu_pairs, required_flanking_seqs=args.required_flanking_seqs)
            diversity_guide_frac = float(len(diversity_guide_bound)) / aln.num_sequences
            if args.diversity == 'entropy':
                all_entropy = aln_for_guide.position_entropy()
                diversity_metric = sum(all_entropy)/args.guide_length
            else:
                raise ValueError("Invalid diversity method '%s'; use one of ['entropy']" %args.diversity)
        else:
            diversity_guide = 'None'
            diversity_guide_frac = 0
            diversity_metric = float('inf')

        d = {}
        if args.consensus:
            d['consensus'] = (consensus_guide, consensus_guide_frac)
        if args.mode:
            d['mode'] = (mode_guide, mode_guide_frac)
        if args.diversity:
            d[args.diversity] = ((diversity_guide, diversity_guide_frac), diversity_metric)
        guides[i] = d
    return guides


def find_guide_in_each_window(guides, aln_length, args, obj_type='max'):
    """Determine a guide for each window of an alignment.

    For each window, this selects the args.best_n guides within it 
    (given one guide per position) that has the best metric score.

    To break ties, this selects the first guide (in terms of
    position) among ones with a tied metric.

    Args:
        guides: list such that guides[i] is a tuple (guide, metric)
            giving a guide sequence (guide) at position i of
            an alignment and a metric (metric) that can be used to 
            compare guide quality
        aln_length: length of alignment
        args: arguments to program
        obj_type: if 'max', consider the largest value the best; else
            if 'min', consider the smallest value the best. Must be
            either 'min' or 'max'

    Returns:
        list x where x[i] gives a list of the args.best_n tuples (guide, metric)
        representing guides in the window that starts at position i
    """
    window_start_positions = range(aln_length - args.window_size + 1)
    guide_in_window = [[] for _ in window_start_positions]
    for i in window_start_positions:
        window_start, window_end = i, i + args.window_size
        last_guide_pos = window_end - args.guide_length
        logger.info("Searching for a guide within window [%d, %d)" %
                (window_start, window_end))

        # Check if any of the guides are no longer within the window
        # Keep track of which positions still have guides in the heap
        positions = set()
        if i > 0:
            for guide in guide_in_window[i-1]:
                if guide[1] >= window_start:
                    guide_in_window[i].append(guide)
                    positions.add(guide[1])
        heapq.heapify(guide_in_window[i])

        if len(guide_in_window[i]) < args.best_n: 
            # One of the previous args.best_n best guides is no longer in the 
            # window; find a new one
            for j in range(window_start, last_guide_pos + 1):
                # Skip if guide is already in the heap
                if -j in positions:
                    continue

                guide, metric = guides[j]
                # Reverse order for minimizing
                if obj_type == 'min':
                    metric = -metric

                # If there aren't args.best_n guides on the heap yet, add the 
                # guide to the heap
                if len(guide_in_window[i]) < args.best_n:
                    # Use negative position to break ties
                    heapq.heappush(guide_in_window[i], (metric, -j, guide))
                # If the new guide has a better metric than the worst guide on the
                # heap, remove the worst guide and add the new one
                elif metric > guide_in_window[i][0][0]:
                    # Use negative position to break ties
                    heapq.heappushpop(guide_in_window[i], (metric, -j, guide))
        else:
            # All args.best_n guides are still within the window, but now
            # check if the new guide at the very end of the window does better
            guide, metric = guides[last_guide_pos]
            # Reverse order for minimizing
            if obj_type == 'min':
                metric = -metric
            if metric > guide_in_window[i][0][0]:
                heapq.heappushpop(guide_in_window[i], (metric, -last_guide_pos, guide))

    # Undo reverse order for minimizing and sort
    fix = 1 if obj_type == 'max' else -1
    guide_in_window = [[(guide, fix*metric) for metric, _, guide \
            in sorted(guide_in_window_i, reverse=True)] \
            for guide_in_window_i in guide_in_window]
    return guide_in_window


def main(args):
    # Allow G-U base pairing, unless it is explicitly disallowed
    args.allow_gu_pairs = not args.do_not_allow_gu_pairing
    # Run the consensus method, unless it is explicitly disallowed
    args.consensus = not args.no_consensus
    # Run the mode method, unless it is explicitly disallowed
    args.mode = not args.no_mode
    # Check if there is a reference sequence if there is a diversity method
    if args.diversity:
        if not args.ref_seq:
            raise Exception('Must include a reference sequence label to run any diversity method')

    args.required_flanking_seqs = (args.require_flanking5, args.require_flanking3)

    # Read the input alignment
    seqs = seq_io.read_fasta(args.in_fasta)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
    try:
        ref_seq = seqs[args.ref_seq] if args.ref_seq else None
    except KeyError:
        raise Exception('Reference sequence %s does not match any label of the sequences in the given FASTA' \
                %args.ref_seq)

    # Construct a guide at each position of the alignment
    logger.info("Constructing guides naively at each position of alignment")
    guides = construct_guide_naively_at_each_pos(aln, args, ref_seq=ref_seq)

    # Find the best guide in each window (for the
    # consensus, mode, and diversity approaches)
    if args.consensus:
        logger.info("Searching for consensus guides")
        consensus_guides_in_window = find_guide_in_each_window(
                [guides[i]['consensus'] for i in range(len(guides))],
                aln.seq_length, args)
    if args.mode:
        logger.info("Searching for mode guides")
        mode_guides_in_window = find_guide_in_each_window(
                [guides[i]['mode'] for i in range(len(guides))],
                aln.seq_length, args)
    if args.diversity:
        logger.info("Searching for %s guides" %args.diversity)
        diversity_guides_in_window = find_guide_in_each_window(
                [guides[i][args.diversity] for i in range(len(guides))],
                aln.seq_length, args, obj_type='min')

    # Write the guides to a TSV file
    with open(args.out_tsv, 'w') as outf:
        header = ['window-start', 'window-end', 'rank']
        if args.consensus:
            header.extend(['target-sequence-by-consensus', 'frac-bound-by-consensus'])
        if args.mode:
            header.extend(['target-sequence-by-mode', 'frac-bound-by-mode'])
        if args.diversity:
            header.extend(['target-sequence-by-%s' %args.diversity, args.diversity, 'frac-bound-by-%s' %args.diversity])

        outf.write('\t'.join(header) + '\n')
        for i in range(aln.seq_length - args.window_size + 1):
            for j in range(args.best_n):
                line = [i, i + args.window_size, j+1]
                if args.consensus:
                    line.extend(consensus_guides_in_window[i][j])
                if args.mode:
                    line.extend(mode_guides_in_window[i][j])
                if args.diversity:
                    diversity_line = (diversity_guides_in_window[i][j][0][0], 
                                      diversity_guides_in_window[i][j][1],
                                      diversity_guides_in_window[i][j][0][1])
                    line.extend(diversity_line)
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
            help=("Output guide(s) within each window (sliding along "
                  "the alignment) of this length"))

    # Parameters on guide length and mismatches
    parser.add_argument('-gl', '--guide-length', type=int, default=28,
            help="Length of guide to construct")
    parser.add_argument('-gm', '--guide-mismatches', type=int, default=0,
            help=("Allow for this number of mismatches when "
                  "determining whether a guide covers a sequence"))

    # Best n guides per window
    parser.add_argument('--best-n', type=int, default=1,
            help=("Find the best BEST_N guides in each window"))

    # G-U pairing options
    parser.add_argument('--do-not-allow-gu-pairing', action='store_true',
            help=("When determining whether a guide binds to a region of "
                  "target sequence, do not count G-U (wobble) base pairs as "
                  "matching. Default is to tolerate G-U pairing: namely, "
                  "A in an output guide sequence matches G in the "
                  "target and C in an output guide sequence matches T "
                  "in the target (since the synthesized guide is the reverse "
                  "complement of the output guide sequence)"))

    # Options to skip
    parser.add_argument('--skip-gaps', type=float, default=0.5,
            help=("If this fraction or more of sequences at a position contain "
                  "a gap character, do not design a guide there"))

    # Reference sequence
    parser.add_argument('--ref-seq', type=str, default=None,
            help=("The label used in the FASTA file of the reference sequence "
                  "to design guides based on sequence diversity; required "
                  "for diversity method"))

    # Guide sequence methods
    parser.add_argument('--no-consensus', action='store_true',
            help=("If set, do not use the consensus method to determine guides; "
                  "otherwise, will use the consensus method"))
    parser.add_argument('--no-mode', action='store_true',
            help=("If set, do not use the mode method to determine guides; "
                  "otherwise, will use the mode method"))
    parser.add_argument('--diversity', type=str, default=None, choices=["entropy"],
            help=("A string of which diversity method to use to determine guides "
                  "('entropy'); None (default) to not use a diversity method. "
                  "'entropy' will calculate the average per position entropy of "
                  "each potential guide, then return the guides at the positions "
                  "with the lowest entropy; nucleotides are determined by the "
                  "reference sequence"))

    # Requiring flanking sequence (PFS)
    parser.add_argument('--require-flanking5',
        help=("Require the given sequence on the 5' protospacer flanking "
              "site (PFS) of each designed guide; this tolerates ambiguity "
              "in the sequence (e.g., 'H' requires 'A', 'C', or 'T', or, "
              "equivalently, avoids guides flanked by 'G'). Note that "
              "this is the 5' end in the target sequence (not the spacer "
              "sequence)."))
    parser.add_argument('--require-flanking3',
        help=("Require the given sequence on the 3' protospacer flanking "
              "site (PFS) of each designed guide; this tolerates ambiguity "
              "in the sequence (e.g., 'H' requires 'A', 'C', or 'T', or, "
              "equivalently, avoids guides flanked by 'G'). Note that "
              "this is the 3' end in the target sequence (not the spacer "
              "sequence)."))

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
