#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
import logging
import re

from dxguidedesign import alignment
from dxguidedesign import guide_search
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io
from dxguidedesign.utils import year_cover

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def seqs_grouped_by_year(seqs, args):
    years_fn, year_highest_cover, year_cover_decay = args.cover_by_year_decay

    # Map sequence names to index in alignment, and construct alignment
    seq_list = []
    seq_idx = {}
    for i, (name, seq) in enumerate(seqs.items()):
        seq_idx[name] = i
        seq_list += [seq]
    aln = alignment.Alignment.from_list_of_seqs(seq_list)

    # Read sequences for each year, and check that every sequence has
    # a year
    years = year_cover.read_years(years_fn)
    all_seqs_with_year = set.union(*years.values())
    for seq in seq_idx.keys():
        if seq not in all_seqs_with_year:
            raise Exception("Unknown year for sequence '%s'" % seq)

    # Convert years dict to map to indices rather than sequence names
    years_idx = {}
    for year in years.keys():
        # Skip names not in seq_idx because the years file may contain
        # sequences that are not in seqs
        years_idx[year] = set(seq_idx[name] for name in years[year]
            if name in seq_idx)

    # Construct desired partial cover for each year
    cover_frac = year_cover.construct_partial_covers(
        years.keys(), year_highest_cover, args.cover_frac, year_cover_decay)

    return aln, years_idx, cover_frac

def design_independently(args):
    # Read required guides, if provided
    num_aln = len(args.in_fasta)
    if args.required_guides:
        required_guides = seq_io.read_required_guides(
            args.required_guides, args.guide_length, num_aln)
    else:
        required_guides = [{} for _ in range(num_aln)]

    # Read blacklisted ranges, if provided
    if args.blacklisted_ranges:
        blacklisted_ranges = seq_io.read_blacklisted_ranges(
            args.blacklisted_ranges, num_aln)
    else:
        blacklisted_ranges = [{} for _ in range(num_aln)]

    # Treat each alignment independently
    for i, (in_fasta, out_tsv) in enumerate(zip(args.in_fasta, args.out_tsv)):
        # Read the sequences and make an Alignment object
        seqs = seq_io.read_fasta(in_fasta)
        if args.cover_by_year_decay:
            aln, seq_groups, cover_frac = seqs_grouped_by_year(seqs, args)
        else:
            aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
            seq_groups = None
            cover_frac = args.cover_frac

        required_guides_for_aln = required_guides[i]
        blacklisted_ranges_for_aln = blacklisted_ranges[i]

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file
        gs = guide_search.GuideSearcher(aln, args.guide_length, args.mismatches,
                                        args.window_size, cover_frac,
                                        args.missing_thres,
                                        seq_groups=seq_groups,
                                        required_guides=required_guides_for_aln,
                                        blacklisted_ranges=blacklisted_ranges_for_aln)
        gs.find_guides_that_cover(out_tsv, sort=args.sort_out)


def design_for_id(args):
    # Create an alignment object for each input
    alns = []
    seq_groups_per_input = []
    cover_frac_per_input = []
    for in_fasta in args.in_fasta:
        seqs = seq_io.read_fasta(in_fasta)
        if args.cover_by_year_decay:
            aln, seq_groups, cover_frac = seqs_grouped_by_year(seqs, args)
        else:
            aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
            seq_groups = None
            cover_frac = args.cover_frac
        alns += [aln]
        seq_groups_per_input += [seq_groups]
        cover_frac_per_input += [cover_frac]

    # Read required guides, if provided
    num_aln = len(alns)
    if args.required_guides:
        required_guides = seq_io.read_required_guides(
            args.required_guides, args.guide_length, num_aln)
    else:
        required_guides = [{} for _ in range(num_aln)]

    # Read blacklisted ranges, if provided
    if args.blacklisted_ranges:
        blacklisted_ranges = seq_io.read_blacklisted_ranges(
            args.blacklisted_ranges, num_aln)
    else:
        blacklisted_ranges = [{} for _ in range(num_aln)]

    logger.info(("Constructing data structure to allow differential "
        "identification"))
    aq = alignment.AlignmentQuerier(alns, args.guide_length,
        args.diff_id_mismatches)
    aq.setup()

    for i, aln in enumerate(alns):
        seq_groups = seq_groups_per_input[i]
        cover_frac = cover_frac_per_input[i]
        required_guides_for_aln = required_guides[i]
        blacklisted_ranges_for_aln = blacklisted_ranges[i]

        def guide_is_specific(guide):
            # Returns True iff guide does not hit too many sequences in
            # alignments other than aln
            return aq.guide_is_specific_to_aln(guide, i, args.diff_id_frac)

        # Mask alignment with index i (aln) from being reported in queries
        # because we will likely get many guide sequences that hit aln, but we
        # do not care about these for checking specificity
        logger.info("Masking alignment %d (of %d) from alignment queries",
            i + 1, len(alns))
        aq.mask_aln(i)

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file; ensure that the selected guides are
        # specific to this alignment
        logger.info("Finding guides for alignment %d (of %d)", i + 1, len(alns))
        gs = guide_search.GuideSearcher(aln, args.guide_length, args.mismatches,
                                        args.window_size, cover_frac,
                                        args.missing_thres,
                                        guide_is_suitable_fn=guide_is_specific,
                                        seq_groups=seq_groups,
                                        required_guides=required_guides_for_aln,
                                        blacklisted_ranges=blacklisted_ranges_for_aln)
        gs.find_guides_that_cover(args.out_tsv[i], sort=args.sort_out)

        # i should no longer be masked from queries
        logger.info("Unmasking alignment %d (of %d) from alignment queries",
            i + 1, len(alns))
        aq.unmask_all_aln()


def main(args):
    logger = logging.getLogger(__name__)

    if len(args.in_fasta) != len(args.out_tsv):
        raise Exception("Number output TSVs must match number of input FASTAs")

    if (args.diff_id_mismatches or args.diff_id_frac) and not args.diff_id:
        logger.warning(("--id-m or --id-frac is useless without also "
            "specifying --id"))
    if args.diff_id:
        # Specify default values for --id-m and --id-frac (to allow the above
        # check, do not do this with argparse directly)
        if not args.diff_id_mismatches:
            args.diff_id_mismatches = 2
        if not args.diff_id_frac:
            args.diff_id_frac = 0.05

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
    parser.add_argument('-o', '--out-tsv', nargs='+', required=True,
        help=("Path to output TSV. If more than one input "
              "FASTA is given, the same number of output TSVs "
              "must be given; each output TSV corresponds to "
              "an input FASTA."))

    parser.add_argument('-l', '--guide-length', type=int, default=28,
        help="Length of guide to construct")
    parser.add_argument('-m', '--mismatches', type=int, default=0,
        help=("Allow for this number of mismatches when "
              "determining whether a guide covers a sequence"))
    parser.add_argument('-w', '--window-size', type=int, default=200,
        help=("Ensure that selected guides are all within a "
              "window of this size"))

    def check_cover_frac(val):
        fval = float(val)
        if fval > 0 and fval <= 1:
            # a float in (0, 1]
            return fval
        else:
            raise argparse.ArgumentTypeError("%s is an invalid -p value" % val)
    parser.add_argument('-p', '--cover-frac', type=check_cover_frac, default=1.0,
        help=("The fraction of sequences that must be covered "
              "by the selected guides"))

    class ParseCoverDecay(argparse.Action):
        # This is needed because --cover-by-year-decay has multiple args
        # of different types
        def __call__(self, parser, namespace, values, option_string=None):
            a, b, c = values
            # Check that b is a valid year
            year_pattern = re.compile('^(\d{4})$')
            if year_pattern.match(b):
                bi = int(b)
            else:
                raise argparse.ArgumentTypeError(("%s is an invalid 4-digit "
                    "year") % b)
            # Check that c is a valid decay
            cf = float(c)
            if cf <= 0 or cf >= 1:
                raise argparse.ArgumentTypeError(("%s is an invalid decay; it "
                    "must be a float in (0,1)" % c))
            setattr(namespace, self.dest, (a, bi, cf))
    parser.add_argument('--cover-by-year-decay', nargs=3, action=ParseCoverDecay,
        help=("<A> <B> <C>; if set, group input sequences by year and set a "
              "desired partial cover for each year (fraction of sequences that "
              "must be covered by guides) as follows: A is a tsv giving "
              "a year for each input sequence (col 1 is sequence name "
              "matching that in the input FASTA, col 2 is year). All years "
              ">= B receive a desired cover fraction of COVER_FRAC "
              "(specified with -p or --cover-frac). Each preceding year "
              "receives a desired cover fraction that decays by C -- i.e., "
              "year n is given C*(desired cover fraction of year n+1)"))

    parser.add_argument('--sort', dest='sort_out', action='store_true',
        help=("If set, sort output TSV by number of guides "
              "(ascending) then by score (descending); "
              "default is to sort by window position"))

    parser.add_argument('--missing-thres', nargs=3, type=float, default=[0.5, 0.05, 1.5],
        help=("<A> <B> <C>; parameters governing the threshold on which sites "
              "to ignore due to too much missing data. The 3 values specify "
              "not to attempt to design guides overlapping sites where the "
              "fraction of sequences with missing data is > min(A, max(B, C*m) "
              "where m is the median fraction of sequences with missing data "
              "over the alignment. Set a=1 and b=1 to not ignore sites due "
              "to missing data."))

    parser.add_argument('--id', dest="diff_id", action='store_true',
        help=("Design guides to perform differential "
              "identification, where each input FASTA is a "
              "group/taxon to identify with specificity; see "
              "--id-m and --id-thres for more"))
    parser.add_argument('--id-m', dest="diff_id_mismatches", type=int,
        help=("Allow for this number of mismatches when determining whether "
              "a guide 'hits' a sequence in a group/taxon other than the "
              "for which it is being designed; higher values correspond to more "
              "specificity. Ignored when --id is not set."))
    parser.add_argument('--id-frac', dest="diff_id_frac", type=float,
        help=("Decide that a guide 'hits' a group/taxon if it 'hits' a "
              "fraction of sequences in that group/taxon that exceeds this "
              "value; lower values correspond to more specificity. Ignored "
              "when --id is not set."))

    parser.add_argument('--required-guides',
        help=("Path to a file that gives guide sequences that will be "
              "included in the guide cover and output for the windows "
              "in which they belong, e.g., if certain guide sequences are "
              "shown experimentally to perform well. The file must have "
              "3 columns: col 1 gives an identifier for the alignment "
              "that the guide covers, such that i represents the i'th "
              "FASTA given as input (0-based); col 2 gives a guide sequence; "
              "col 3 gives the start position of the guide (0-based) in "
              "the alignment"))

    parser.add_argument('--blacklisted-ranges',
        help=("Path to a file that gives ranges in alignments from which "
              "guides will not be constructed. The file must have 3 columns: "
              "col 1 gives an identifier for the alignment that the range "
              "corresponds to, such that i represents the i'th FASTA "
              "given as input (0-based); col 2 gives the start position of "
              "the range (inclusive); col 3 gives the end position of the "
              "range (exclusive)"))

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
