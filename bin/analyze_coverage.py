#!/usr/bin/env python3
"""Analyze coverage obtained by designs."""

import argparse
import logging

from adapt import coverage_analysis
from adapt.prepare import ncbi_neighbors
from adapt.utils import log
from adapt.utils import predict_activity
from adapt.utils import seq_io

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


def main(args):
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
                seqs, designs, args.guide_mismatches, args.primer_mismatches,
                allow_gu_pairs, fully_sensitive=args.fully_sensitive)
    elif args.predict_activity_model_path:
        cla_path, reg_path = args.predict_activity_model_path
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
                seqs, designs, predictor, args.primer_mismatches,
                highly_active=highly_active,
                fully_sensitive=args.fully_sensitive)
    else:
        raise Exception(("One of --guide-mismatches or "
            "--predict-activity-model-path must be set"))

    # Perform analyses
    performed_analysis = False
    if args.write_frac_bound:
        frac_bound = analyzer.frac_of_seqs_bound()
        write_frac_bound(designs, frac_bound, args.write_frac_bound)
        performed_analysis = True
    if args.write_mean_activity_of_guides:
        if (not args.predict_activity_model_path or
                args.predict_activity_require_highly_active):
            raise Exception(("To use --write-mean-activity-of-guides, "
                    "a predictive model must be set and "
                    "--predict-activity-require-highly-active must *not* "
                    "be set"))
        mean_activity = analyzer.mean_activity_of_guides()
        write_mean_activity_of_guides(designs, mean_activity,
                args.write_mean_activity_of_guides)
        performed_analysis = True

    if not performed_analysis:
        logger.warning(("No analysis was requested"))

    # Close tempfiles
    if seqs_tempfile is not None:
        seqs_tempfile.close()


if __name__ == "__main__":
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

    # Parameter determining whether a primer binds to target
    parser.add_argument('-pm', '--primer-mismatches',
        type=int, default=0,
        help=("Allow for this number of mismatches when determining "
              "whether a primer covers a sequence (ignore this if "
              "the targets only consist of guides)"))

    # Parameters determining whether a guide binds to target based on
    # mismatch model
    parser.add_argument('-gm', '--guide-mismatches',
        type=int,
        help=("Allow for this number of mismatches when "
              "determining whether a guide covers a sequence; either this "
              "or --predict-activity-model-path should be set"))
    parser.add_argument('--do-not-allow-gu-pairing',
        action='store_true',
        help=("When determining whether a guide binds to a region of "
              "target sequence, do not count G-U (wobble) base pairs as "
              "matching. Default is to tolerate G-U pairing: namely, "
              "A in an output guide sequence matches G in the "
              "target and C in an output guide sequence matches T "
              "in the target (since the synthesized guide is the reverse "
              "complement of the output guide sequence)"))

    # Parameters determining whether a guide binds to target based on
    # trained model
    parser.add_argument('--predict-activity-model-path',
        nargs=2,
        help=("Paths to directories containing serialized models in "
              "TensorFlow's SavedModel format for predicting guide-target "
              "activity. There are two arguments: (1) classification "
              "model to determine which guides are active; (2) regression "
              "model, which is used to determine which guides (among "
              "active ones) are highly active. The models/ directory "
              "contains example models. Either this or --guide-mismatches "
              "should be set."))
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

    args = parser.parse_args()
    log.configure_logging(args.log_level)
    main(args)
