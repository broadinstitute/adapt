#!/usr/bin/env python3
"""Analyze coverage obtained by designs."""

import argparse
import logging
import os
import sys
import math

from adapt import coverage_analysis
from adapt.prepare import ncbi_neighbors
from adapt.utils import log
from adapt.utils import predict_activity
from adapt.utils import seq_io
from adapt.utils.version import get_project_path, get_latest_model_version

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
                    cols['guide-target-sequences'].split(' ')
                    if len(cols['guide-target-sequences']) > 0 else [],
                    (cols['left-primer-target-sequences'].split(' '),
                     cols['right-primer-target-sequences'].split(' ')))
        designs[i + 1] = design
    return designs


def read_accessions(fn):
    """Read a list of accessions from a file.

    Args:
        fn: path to file where each line gives an accession

    Returns:
        collection of accessions
    """
    accs = []
    with open(fn) as f:
        for line in f:
            line = line.rstrip()
            accs += [line]
    return accs


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


def write_mean_activity_of_guides(designs, mean_activities, out_fn):
    """Write table giving the mean activity of guide sets.

    Args:
        designs: dict {design_id: design} where design_id is an
            identifier for design and design is a coverage_analysis.
            Design object
        mean_activities: dict {design_id: activity} where design_id is
            an identifier for a design and activity is the mean activity,
            across the target sequences, of its guide set
        out_fn: path to TSV file at which to write table
    """
    header = ['design_id',
              'guide-target-sequences',
              'mean-activity']
    with open(out_fn, 'w') as fw:
        fw.write('\t'.join(header) + '\n')
        for design_id in sorted(list(designs.keys())):
            guides = designs[design_id].guides
            guides = ' '.join(sorted(guides))
            row = [design_id, guides, mean_activities[design_id]]
            fw.write('\t'.join([str(x) for x in row]) + '\n')

def write_per_seq(designs, seqs, out_fn, per_seq_guides=None,
                  per_seq_primers=None, guide_activity_model=True,
                  primer_terminal_mismatches=False):
    """Write table giving the fraction of sequences bound by a target.

    Args:
        designs: dict {design_id: design} where design_id is an
            identifier for design and design is a coverage_analysis.
            Design object
        seqs: dict {seq_name: target sequence} where seq_name is the name of
            a sequence in the input FASTA file
        out_fn: path to TSV file at which to write table
        per_seq_guides: output of CoverageAnalyzer.per_seq_guide_mismatches;
            dict mapping design identifier (self.designs.keys()) to a tuple of
            the best guide scores per target sequence of its guide set
            and the best guide per target sequence of its guide set
        per_seq_primers: output of CoverageAnalyzer.per_seq_primer_mismatches;
            tuple of two dicts (first for the left, second for the right)
            mapping design identifier (self.designs.keys()) to a tuple of
            the best primer mismatches per target sequence of its primer set
            and the best primer per target sequence of its primer set
        guide_activity_model: if True (default), per_seq_guides contains
            activity score; if False, per_seq_guides contains number of
            mismatches against the target
        primer_terminal_mismatches: if True, per_seq_primers contains a tuple
            of mismatches and terminal mismatches in the score; if False
            (default), per_seq_primers contains the number of mismatches in
            the score
    """
    header = ['design_id', 'seq_name']

    if per_seq_primers:
        header.extend(['target-start', 'target-end', 'target-length'])
        if primer_terminal_mismatches:
            header.extend(['left-primer-mismatches',
                           'left-primer-terminal-mismatches',
                           'left-primer-ideal-target-sequence',
                           'left-primer-start',
                           'right-primer-mismatches',
                           'right-primer-terminal-mismatches',
                           'right-primer-ideal-target-sequence',
                           'right-primer-start'])
        else:
            header.extend(['left-primer-mismatches',
                           'left-primer-ideal-target-sequence',
                           'left-primer-start',
                           'right-primer-mismatches',
                           'right-primer-ideal-target-sequence',
                           'right-primer-start'])
    if per_seq_guides:
        if guide_activity_model:
            header.append('guide-activity')
            guide_none = 0
        else:
            header.append('guide-mismatches')
            guide_none = math.inf
        header.extend(['guide-ideal-target-sequence', 'guide-start'])
    left_primers = per_seq_primers[0]
    right_primers = per_seq_primers[1]
    with open(out_fn, 'w') as fw:
        fw.write('\t'.join(header) + '\n')
        for design_id in sorted(list(designs.keys())):
            for seq_name in seqs:
                row = [design_id, seq_name]
                if per_seq_primers:
                    left_scores, left_target, left_start = \
                        left_primers[design_id][seq_name]
                    right_scores, right_target, right_start = \
                        right_primers[design_id][seq_name]
                    # Get left primer start for target start
                    target_start = left_start
                    target_length = None
                    # Get right primer end + right primer len for target end
                    if right_start is not None:
                        target_end = right_start + len(right_target)
                        if target_start is not None:
                            target_length = target_end-target_start
                    else:
                        target_end = None
                    row.extend([target_start, target_end, target_length])
                    if left_scores[0] == math.inf:
                        if primer_terminal_mismatches:
                            left_scores = (None, None)
                        else:
                            left_scores = (None, )
                    if right_scores[0] == math.inf:
                        if primer_terminal_mismatches:
                            right_scores = (None, None)
                        else:
                            right_scores = (None, )
                    row.extend([*left_scores, left_target, left_start,
                                *right_scores, right_target, right_start])
                if per_seq_guides:
                    guide_scores, guide_target, guide_start = \
                        per_seq_guides[design_id][seq_name]
                    if guide_scores[0] == guide_none:
                        guide_scores = (None, )
                    row.extend([*guide_scores, guide_target, guide_start])
                fw.write('\t'.join([str(x) for x in row]) + '\n')


def run(args):
    # Allow G-U base pairing, unless it is explicitly disallowed
    allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Read the designs
    designs = read_designs(args.designs_fn)

    # If accessions were given, fetch them as use them as input
    if args.use_accessions:
        accessions = read_accessions(args.seqs_fn)
        seqs_tempfile = ncbi_neighbors.fetch_fastas(accessions)
        seqs_fn = seqs_tempfile.name
    else:
        seqs_tempfile = None
        seqs_fn = args.seqs_fn

    # Read the input sequences to compute coverage against; use
    # skip_gaps=True so that, if an alignment is input, this
    # is read as unaligned sequences
    seqs = seq_io.read_fasta(seqs_fn, skip_gaps=True)

    if (args.guide_mismatches is not None) and args.predict_activity_model_path:
        raise Exception(("Cannot set both --guide-mismatches and "
            "--predict-activity-model-path. Choose --guide-mismatches "
            "for a model based on mismatches, and --predict-activity-model-"
            "path to make determinations based on whether predicted "
            "activity is high."))
    elif args.guide_mismatches is not None:
        analyzer = coverage_analysis.CoverageAnalyzerWithMismatchModel(
                seqs, designs, args.guide_mismatches,
                primer_mismatches=args.primer_mismatches,
                allow_gu_pairs=allow_gu_pairs,
                fully_sensitive=args.fully_sensitive,
                primer_terminal_mismatches=args.primer_terminal_mismatches,
                bases_from_terminal=args.bases_from_terminal,
                max_target_length=args.max_target_length)
    elif args.predict_activity_model_path or args.predict_cas13a_activity_model is not None:
        if args.predict_activity_model_path:
            cla_path, reg_path = args.predict_activity_model_path
        else:
            dir_path = get_project_path()
            cla_path_all = os.path.join(dir_path, 'models', 'classify',
                                    'cas13a')
            reg_path_all = os.path.join(dir_path, 'models', 'regress',
                                    'cas13a')
            if len(args.predict_cas13a_activity_model) not in (0,2):
                raise Exception(("If setting versions for "
                    "--predict-cas13a-activity-model, both a version for "
                    "the classifier and the regressor must be set."))
            if (len(args.predict_cas13a_activity_model) == 0 or
                    args.predict_cas13a_activity_model[0] == 'latest'):
                cla_version = get_latest_model_version(cla_path_all)
            else:
                cla_version = args.predict_cas13a_activity_model[0]
            if (len(args.predict_cas13a_activity_model) == 0 or
                    args.predict_cas13a_activity_model[1] == 'latest'):
                reg_version = get_latest_model_version(reg_path_all)
            else:
                reg_version = args.predict_cas13a_activity_model[1]
            cla_path = os.path.join(cla_path_all, cla_version)
            reg_path = os.path.join(reg_path_all, reg_version)
        if args.predict_activity_thres:
            # Use specified thresholds on classification and regression
            cla_thres, reg_thres = args.predict_activity_thres
        else:
            # Use default thresholds specified with the model
            cla_thres, reg_thres = None, None
        predictor = predict_activity.Predictor(cla_path, reg_path,
                classification_threshold=cla_thres,
                regression_threshold=reg_thres)
        highly_active = args.predict_activity_require_highly_active
        analyzer = coverage_analysis.CoverageAnalyzerWithPredictedActivity(
                seqs, designs, predictor,
                primer_mismatches=args.primer_mismatches,
                highly_active=highly_active,
                fully_sensitive=args.fully_sensitive,
                primer_terminal_mismatches=args.primer_terminal_mismatches,
                bases_from_terminal=args.bases_from_terminal,
                max_target_length=args.max_target_length)
    else:
        raise Exception(("One of --guide-mismatches, "
            "--predict-cas13a-activity-model, or"
            "--predict-activity-model-path must be set"))

    # Perform analyses
    performed_analysis = False
    if args.write_frac_bound:
        frac_bound = analyzer.frac_of_seqs_bound()
        write_frac_bound(designs, frac_bound, args.write_frac_bound)
        performed_analysis = True
    if args.write_mean_activity_of_guides:
        if (not (args.predict_activity_model_path or
                 args.predict_cas13a_activity_model is not None) or
                args.predict_activity_require_highly_active):
            raise Exception(("To use --write-mean-activity-of-guides, "
                    "a predictive model must be set and "
                    "--predict-activity-require-highly-active must *not* "
                    "be set"))
        mean_activity = analyzer.mean_activity_of_guides()
        write_mean_activity_of_guides(designs, mean_activity,
                args.write_mean_activity_of_guides)
        performed_analysis = True
    if args.write_per_seq:
        guide_activity_model = None
        if args.predict_activity_model_path or args.predict_cas13a_activity_model is not None:
            guide_activity_model = True
        elif args.guide_mismatches is not None:
            guide_activity_model = False
        else:
            raise Exception(("One of --guide-mismatches, "
            "--predict-cas13a-activity-model, or"
            "--predict-activity-model-path must be set"))
        per_seq_guides = analyzer.per_seq_guide()
        if args.predict_activity_require_highly_active:
            # TODO: Could in theory use this to create a binary "highly
            # active" column
            logger.warning(("When using --write-per-seq, "
                    "--predict-activity-require-highly-active is not "
                    "used"))
        per_seq_primers = analyzer.per_seq_primer_mismatches()
        write_per_seq(designs, seqs, args.write_per_seq,
                      per_seq_guides=per_seq_guides,
                      per_seq_primers=per_seq_primers,
                      guide_activity_model=guide_activity_model,
                      primer_terminal_mismatches=(args.primer_terminal_mismatches is not None))
        performed_analysis = True

    if not performed_analysis:
        logger.warning(("No analysis was requested"))

    # Close tempfiles
    if seqs_tempfile is not None:
        seqs_tempfile.close()


def argv_to_args(argv):
    parser = argparse.ArgumentParser()

    # Required inputs
    parser.add_argument('designs_fn',
        help=("Path to output of running design.py; a TSV file where "
              "each row contains a design (target)"))
    parser.add_argument('seqs_fn',
        help=("Path to FASTA file giving sequences against which to "
              "compute coverage. (See --use-accessions to pass accessions "
              "as input rather than a FASTA file.)"))

    # Analyses to output
    parser.add_argument('--write-frac-bound',
        help=("If set, write a table in which each row represents an "
              "input design and gives the fraction of all sequences that "
              "are covered by the design. The 'design_id' column gives "
              "the row number of the design in the designs input (1 for "
              "the first design). The provided argument is a path to "
              "a TSV file at which to the write the table."))
    parser.add_argument('--write-mean-activity-of-guides',
        help=("If set, write a table in which each row represents an "
              "input design and gives the mean activity across the target "
              "sequences of the guide set. The 'design_id' column gives "
              "the row number of the design in the designs input (1 for "
              "the first design). The provided argument is a path to "
              "a TSV file at which to write the table. If set, a predictive "
              "model must be set without "
              "--predict-activity-require-highly-active"))
    parser.add_argument('--write-per-seq',
        help=("If set, write a table in which each row represents an assay's "
              "performance on an input sequence with the activity and "
              "objective score. The 'seq_name' gives the sequence name in the "
              "FASTA alignment provided; the 'design_id' column gives the row "
              "number of the design in the designs input (1 for the first "
              "design). The provided argument is a path to a TSV file at "
              "which to write the table. If set, a predictive model must be "
              "set without --predict-activity-require-highly-active"))

    # Parameter determining whether a primer binds to target
    parser.add_argument('-pm', '--primer-mismatches',
        type=int, default=0,
        help=("Allow for this number of mismatches when determining "
              "whether a primer covers a sequence. (ignore this if "
              "the targets only consist of guides)"))
    parser.add_argument('-ptm', '--primer-terminal-mismatches',
        type=int,
        help=("Allow for this number of mismatches in the BASES_FROM_TERMINAL "
              "bases from the 3' end when determining whether a primer covers "
              "a sequence. Default is unset (and therefore unused). (ignore "
              "this if the targets only consist of guides)"))
    parser.add_argument('--bases-from-terminal',
        type=int, default=5,
        help=("Allow for PRIMER_TERMINAL_MISMATCHES in this many bases from "
              "the 3' end when determining whether a primer covers a sequence."
              "Default is 5 and is only used if PRIMER_TERMINAL_MISMATCHES set"
              "(ignore this if the targets only consist of guides)"))

    # Parameters determining whether a guide binds to target based on
    # mismatch model
    parser.add_argument('-gm', '--guide-mismatches',
        type=int,
        help=("Allow for this number of mismatches when "
              "determining whether a guide covers a sequence.  Required if "
              "neither --predict-activity-model-path nor "
              "predict-cas13a-activity-model is not set."))
    parser.add_argument('--do-not-allow-gu-pairing',
        action='store_true',
        help=("When determining whether a guide binds to a region of "
              "target sequence, do not count G-U (wobble) base pairs as "
              "matching. Default is to tolerate G-U pairing: namely, "
              "A in an output guide sequence matches G in the "
              "target and C in an output guide sequence matches T "
              "in the target (since the synthesized guide is the reverse "
              "complement of the output guide sequence)"))

    # Use models to predict activity
    parser.add_argument('--predict-cas13a-activity-model',
        nargs='*',
        help=("Use ADAPT's premade Cas13a model to predict guide-target "
              "activity. Optionally, two arguments can be included to indicate "
              "version number, in the format 'v1_0' or 'latest'. Versions "
              "will default to latest. Required if --guide-mismatches or "
              "predict-activity-model-path is not set."))
    parser.add_argument('--predict-activity-model-path',
        nargs=2,
        help=("Paths to directories containing serialized models in "
              "TensorFlow's SavedModel format for predicting guide-target "
              "activity. There are two arguments: (1) classification "
              "model to determine which guides are active; (2) regression "
              "model, which is used to determine which guides (among "
              "active ones) are highly active. The models/ directory "
              "contains example models. Required if --guide-mismatches or"
              "predict-cas13a-activity-model is not set."))
    parser.add_argument('--predict-activity-thres',
        type=float,
        nargs=2,
        help=("Thresholds to use for decisions on output of predictive "
            "models. There are two arguments: (1) classification threshold "
            "for deciding which guide-target pairs are active (in [0,1], "
            "where higher values have higher precision but less recall); "
            "(2) regression threshold for deciding which guide-target pairs "
            "are highly active (>= 0, where higher values limit the number "
            "determined to be highly active). If not set but --predict-"
            "activity-model-path is set, then this uses default thresholds "
            "stored with the models. To 'bind to' or 'cover' a target, "
            "the guide-target pair must be active or, if "
            "--predict-activity-require-highly-active is set, highly active."))
    parser.add_argument('--predict-activity-require-highly-active',
        action='store_true',
        help=("When determining whether a guide-target pair binds using an "
              "activity model, require that the pair be predicted to be "
              "highly active (not just active)"))

    # Parameters determining whether a target is amplified
    parser.add_argument('--max-target-length',
        type=int,
        help=("The maximum length a target can be for it to be amplified; "
            "defaults to no maximum length (i.e. targets of any length "
            "will be considered amplifiable). Does nothing if only a guide is "
            "being considered."))

    # Miscellaneous
    parser.add_argument('--use-accessions',
        action='store_true',
        help=("When set, the input file of sequences gives accessions rather "
              "than being a FASTA of sequences -- each line in the file gives "
              "an accession. This fetches the sequences of those accessions "
              "uses them as input."))
    parser.add_argument('--fully-sensitive',
        action='store_true',
        help=("When set, use a naive, slow sliding approach to find binding "
              "for primers and guides; otherwise, this uses an index to "
              "more quickly identify binding sites"))

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

    args = parser.parse_args(argv[1:])
    log.configure_logging(args.log_level)

    return args

if __name__ == "__main__":
    run(argv_to_args(sys.argv))
