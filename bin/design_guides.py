#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
import logging
import os
import re
import tempfile

from dxguidedesign import alignment
from dxguidedesign import guide_search
from dxguidedesign.prepare import align
from dxguidedesign.prepare import prepare_alignment
from dxguidedesign import primer_search
from dxguidedesign import target_search
from dxguidedesign.utils import guide
from dxguidedesign.utils import log
from dxguidedesign.utils import seq_io
from dxguidedesign.utils import year_cover

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def seqs_grouped_by_year(seqs, args):
    """Group sequences according to their year and assigned partial covers.

    Args:
        seqs: dict mapping sequence name to sequence, as read from a FASTA
        args: namespace of arguments provided to this executable

    Returns:
        tuple (aln, years_idx, cover_frac) where aln is an
        alignment.Alignment object from seqs; years_idx is a dict
        mapping each year to the set of indices in aln representing
        sequences for that year; and cover_frac is a dict mapping each
        year to the desired partial cover of sequences from that year,
        as determined by args.cover_by_year_decay
    """
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
        years.keys(), year_highest_cover, args.guide_cover_frac, year_cover_decay)

    return aln, years_idx, cover_frac


def parse_required_guides_and_blacklist(args):
    """Parse files giving required guides and blacklisted sequence.

    Args:
        args: namespace of arguments provided to this executable

    Returns:
        tuple (required_guides, blacklisted_ranges) where required_guides
        is a representation of data in the args.required_guides file;
        blacklisted_ranges is a representation of data in the
        args.blacklisted_ranges file; and blacklisted_kmers is a
        representation of data in the args.blacklisted_kmers file
    """
    num_aln = len(args.in_fasta)

    # Read required guides, if provided
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
        blacklisted_ranges = [set() for _ in range(num_aln)]

    # Read blacklisted kmers, if provided
    if args.blacklisted_kmers:
        blacklisted_kmers = seq_io.read_blacklisted_kmers(
            args.blacklisted_kmers,
            min_len_warning=5,
            max_len_warning=args.guide_length)
    else:
        blacklisted_kmers = set()

    return required_guides, blacklisted_ranges, blacklisted_kmers


def prepare_alignments(args):
    """Download, curate, and align sequences for input.

    Args:
        args: namespace of arguments provided to this executable

    Returns:
        tuple (in_fasta, out_tsv) in which in_fasta is a list of
        tempfile objects that contain an input alignment and out_tsv
        is a list of paths to files at which to write output
        (out_tsv[i] is the output for in_tsv[i])
    """
    logger.info(("Setting up to prepare alignments"))

    # Set the path to mafft
    align.set_mafft_exec(args.mafft_path)

    # Setup alignment and alignment stat memoizers
    if args.prep_memoize_dir:
        if not os.path.isdir(args.prep_memoize_dir):
            raise Exception(("Path '%s' does not exist") %
                args.prep_memoize_dir)
        align_memoize_dir = os.path.join(args.prep_memoize_dir, 'aln')
        if not os.path.exists(align_memoize_dir):
            os.makedirs(align_memoize_dir)
        align_memoize_stat_file = os.path.join(args.prep_memoize_dir, 'stats')

        am = align.AlignmentMemoizer(align_memoize_dir)
        asm = align.AlignmentStatMemoizer(align_memoize_stat_file)
    else:
        am = None
        asm = None

    # Read list of taxonomies
    if args.input_type == 'auto-from-args':
        s = None if args.segment == 'None' else args.segment
        taxs = [(None, args.tax_id, s, args.ref_acc)]
    elif args.input_type == 'auto-from-file':
        taxs = seq_io.read_taxonomies(args.in_tsv)
    else:
        raise Exception(("Unknown input type '%s'") % args.input_type)

    # Construct alignments for each taxonomy
    in_fasta = []
    out_tsv = []
    for label, tax_id, segment, ref_acc in taxs:
        aln_file = tempfile.NamedTemporaryFile()
        prepare_alignment.prepare_for(tax_id, segment, ref_acc,
            aln_file.name, aln_memoizer=am, aln_stat_memoizer=asm,
            limit_seqs=args.limit_seqs, prep_influenza=args.prep_influenza)
        in_fasta += [aln_file]

        if label is None:
            out_tsv += [args.out_tsv]
        else:
            out_name = label + '.tsv'
            out_tsv += [os.path.join(args.out_tsv_dir, out_name)]

    return in_fasta, out_tsv


def design_for_id(args):
    """Design guides for differential identification across targets.

    Args:
        args: namespace of arguments provided to this executable
    """
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
            cover_frac = args.guide_cover_frac
        alns += [aln]
        seq_groups_per_input += [seq_groups]
        cover_frac_per_input += [cover_frac]

    # Also add the specific_against alignments into alns, but keep
    # track of how many use for design (the first N of them)
    num_aln_for_design = len(args.in_fasta)
    for specific_against_fasta in args.specific_against:
        seqs = seq_io.read_fasta(specific_against_fasta)
        aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
        alns += [aln]

    required_guides, blacklisted_ranges, blacklisted_kmers = \
        parse_required_guides_and_blacklist(args)

    # Allow G-U base pairing, unless it is explicitly disallowed
    allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Construct the data structure for guide queries to perform
    # differential identification
    if len(alns) > 1:
        logger.info(("Constructing data structure to permit guide queries for "
            "differential identification"))
        aq = alignment.AlignmentQuerier(alns, args.guide_length,
            args.diff_id_mismatches, allow_gu_pairs)
        aq.setup()
    else:
        logger.info(("Only one alignment was provided, so not constructing "
            "data structure to permit queries for differential "
            "identification"))
        aq = None

    for i in range(num_aln_for_design):
        logger.info("Finding guides for alignment %d (of %d)",
            i + 1, num_aln_for_design)

        aln = alns[i]
        seq_groups = seq_groups_per_input[i]
        cover_frac = cover_frac_per_input[i]
        required_guides_for_aln = required_guides[i]
        blacklisted_ranges_for_aln = blacklisted_ranges[i]

        def guide_is_suitable(guide):
            # Return True iff the guide does not contain a blacklisted
            # k-mer and is specific to aln

            # Return False if the guide contains a blacklisted k-mer
            for kmer in blacklisted_kmers:
                if kmer in guide:
                    return False

            # Return True if guide does not hit too many sequences in
            # alignments other than aln
            if aq is not None:
                return aq.guide_is_specific_to_aln(guide, i, args.diff_id_frac)
            else:
                return True

        # Mask alignment with index i (aln) from being reported in queries
        # because we will likely get many guide sequences that hit aln, but we
        # do not care about these for checking specificity
        if aq is not None:
            aq.mask_aln(i)

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file; ensure that the selected guides are
        # specific to this alignment
        gs = guide_search.GuideSearcher(aln, args.guide_length,
                                        args.guide_mismatches,
                                        cover_frac, args.missing_thres,
                                        guide_is_suitable_fn=guide_is_suitable,
                                        seq_groups=seq_groups,
                                        required_guides=required_guides_for_aln,
                                        blacklisted_ranges=blacklisted_ranges_for_aln,
                                        allow_gu_pairs=allow_gu_pairs)

        if args.search_cmd == 'sliding-window':
            # Find an optimal set of guides for each window in the genome,
            # and write them to a file
            gs.find_guides_that_cover(args.window_size,
                args.out_tsv[i], sort=args.sort_out)
        elif args.search_cmd == 'complete-targets':
            # Find optimal targets (primer and guide set combinations),
            # and write them to a file
            ps = primer_search.PrimerSearcher(aln, args.primer_length,
                                              args.primer_mismatches,
                                              args.primer_cover_frac,
                                              args.missing_thres)
            ts = target_search.TargetSearcher(ps, gs,
                max_primers_at_site=args.max_primers_at_site,
                max_target_length=args.max_target_length,
                cost_weights=args.cost_fn_weights)
            ts.find_and_write_targets(args.out_tsv[i],
                best_n=args.best_n_targets)
        else:
            raise Exception("Unknown search subcommand '%s'" % args.search_cmd)

        # i should no longer be masked from queries
        if aq is not None:
            aq.unmask_all_aln()


def main(args):
    logger = logging.getLogger(__name__)

    if args.input_type in ['auto-from-file', 'auto-from-args']:
        if args.input_type == 'auto-from-file':
            if not os.path.isdir(args.out_tsv_dir):
                raise Exception(("Output directory '%s' does not exist") %
                    args.out_tsv_dir)

        # Prepare input alignments, stored in temp fasta files
        in_fasta, out_tsv = prepare_alignments(args)
        args.in_fasta = [tf.name for tf in in_fasta]
        args.out_tsv = out_tsv
    elif args.input_type == 'fasta':
        if len(args.in_fasta) != len(args.out_tsv):
            raise Exception(("Number output TSVs must match number of input "
                "FASTAs"))
    else:
        raise Exception("Unknown input type subcommand '%s'" % args.input_type)

    design_for_id(args)

    # Close temporary files storing alignments
    if args.input_type in ['auto_from_file', 'auto-from-args']:
        for tf in in_fasta:
            tf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###########################################################################
    # OPTIONS AVAILABLE ACROSS ALL SUBCOMMANDS
    ###########################################################################
    base_subparser = argparse.ArgumentParser(add_help=False)

    # Parameters on guide length and mismatches
    base_subparser.add_argument('-gl', '--guide-length', type=int, default=28,
        help="Length of guide to construct")
    base_subparser.add_argument('-gm', '--guide-mismatches', type=int, default=0,
        help=("Allow for this number of mismatches when "
              "determining whether a guide covers a sequence"))

    # Desired coverage of target sequences
    def check_cover_frac(val):
        fval = float(val)
        if fval > 0 and fval <= 1:
            # a float in (0, 1]
            return fval
        else:
            raise argparse.ArgumentTypeError("%s is an invalid -p value" % val)
    base_subparser.add_argument('-gp', '--guide-cover-frac',
        type=check_cover_frac, default=1.0,
        help=("The fraction of sequences that must be covered "
              "by the selected guides. If complete-targets is used, then "
              "this is the fraction of sequences *that are bound by the "
              "primers* that must be covered (so, in total, >= "
              "(GUIDE_COVER_FRAC * PRIMER_COVER_FRAC) sequences will be "
              "covered)."))

    # Automatically setting desired coverage of target sequences based
    # on their year
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
    base_subparser.add_argument('--cover-by-year-decay', nargs=3,
        action=ParseCoverDecay,
        help=("<A> <B> <C>; if set, group input sequences by year and set a "
              "desired partial cover for each year (fraction of sequences that "
              "must be covered by guides) as follows: A is a tsv giving "
              "a year for each input sequence (col 1 is sequence name "
              "matching that in the input FASTA, col 2 is year). All years "
              ">= B receive a desired cover fraction of GUIDE_COVER_FRAC "
              "(specified with -p/--cover-frac). Each preceding year receives "
              "a desired cover fraction that decays by C -- i.e., year n is "
              "given C*(desired cover fraction of year n+1). This "
              "grouping and varying cover fraction is only applied to guide "
              "coverage (i.e., not primers if complete-targets is used)."))

    # Handling missing data
    base_subparser.add_argument('--missing-thres', nargs=3,
        type=float, default=[0.5, 0.05, 1.5],
        help=("<A> <B> <C>; parameters governing the threshold on which sites "
              "to ignore due to too much missing data. The 3 values specify "
              "not to attempt to design guides overlapping sites where the "
              "fraction of sequences with missing data is > min(A, max(B, C*m)) "
              "where m is the median fraction of sequences with missing data "
              "over the alignment. Set a=1 and b=1 to not ignore sites due "
              "to missing data."))

    # Differential identification
    base_subparser.add_argument('--id-m', dest="diff_id_mismatches",
        type=int, default=2,
        help=("Allow for this number of mismatches when determining whether "
              "a guide 'hits' a sequence in a group/taxon other than the "
              "for which it is being designed; higher values correspond to more "
              "specificity."))
    base_subparser.add_argument('--id-frac', dest="diff_id_frac",
        type=float, default=0.05,
        help=("Decide that a guide 'hits' a group/taxon if it 'hits' a "
              "fraction of sequences in that group/taxon that exceeds this "
              "value; lower values correspond to more specificity."))
    base_subparser.add_argument('--specific-against', nargs='+',
        default=[],
        help=("Path to one or more FASTA files giving alignments, such that "
              "guides are designed to be specific against (i.e., not hit) "
              "these alignments, according to --id-m and --id-frac. This "
              "is equivalent to specifying the FASTAs in the main input "
              "(as positional inputs), except that, when provided here, "
              "guides are not designed for these alignments."))

    # G-U pairing options
    base_subparser.add_argument('--do-not-allow-gu-pairing', action='store_true',
        help=("When determining whether a guide binds to a region of "
              "target sequence, do not count G-U (wobble) base pairs as "
              "matching. Default is to tolerate G-U pairing: namely, "
              "A in an output guide sequence matches G in the "
              "target and C in an output guide sequence matches T "
              "in the target (since the synthesized guide is the reverse "
              "complement of the output guide sequence)"))

    # Requiring guides in the cover, and blacklisting ranges and/or k-mers
    base_subparser.add_argument('--required-guides',
        help=("Path to a file that gives guide sequences that will be "
              "included in the guide cover and output for the windows "
              "in which they belong, e.g., if certain guide sequences are "
              "shown experimentally to perform well. The file must have "
              "3 columns: col 1 gives an identifier for the alignment "
              "that the guide covers, such that i represents the i'th "
              "FASTA given as input (0-based); col 2 gives a guide sequence; "
              "col 3 gives the start position of the guide (0-based) in "
              "the alignment"))
    base_subparser.add_argument('--blacklisted-ranges',
        help=("Path to a file that gives ranges in alignments from which "
              "guides will not be constructed. The file must have 3 columns: "
              "col 1 gives an identifier for the alignment that the range "
              "corresponds to, such that i represents the i'th FASTA "
              "given as input (0-based); col 2 gives the start position of "
              "the range (inclusive); col 3 gives the end position of the "
              "range (exclusive)"))
    base_subparser.add_argument('--blacklisted-kmers',
        help=("Path to a FASTA file that gives k-mers to blacklisted from "
              "guide sequences. No guide sequences will be constructed that "
              "contain these k-mers. The k-mers make up the sequences in "
              "the FASTA file; the sequence names are ignored. k-mers "
              "should be long enough so that not too many guide sequences "
              "are deemed to be unsuitable, and should be at most the "
              "length of the guide"))

    # Log levels
    base_subparser.add_argument("--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.WARNING,
        help=("Debug output"))
    base_subparser.add_argument("--verbose",
        dest="log_level",
        action="store_const",
        const=logging.INFO,
        help=("Verbose output"))
    ###########################################################################

    ###########################################################################
    # SUBCOMMANDS FOR SEARCH TYPE
    ###########################################################################
    search_subparsers = parser.add_subparsers(dest='search_cmd')

    # Subcommand: sliding-window
    parser_sw = search_subparsers.add_parser('sliding-window',
        help=("Search for guides within a sliding window of a fixed size, "
              "and output the optimal guide set for each window"))
    parser_sw_args = argparse.ArgumentParser(add_help=False)
    parser_sw_args.add_argument('-w', '--window-size', type=int, default=200,
        help=("Ensure that selected guides are all a "
              "window of this size"))
    parser_sw_args.add_argument('--sort', dest='sort_out', action='store_true',
        help=("If set, sort output TSV by number of guides "
              "(ascending) then by score (descending); "
              "default is to sort by window position"))

    # Subcommand: complete-targets
    parser_ct = search_subparsers.add_parser('complete-targets',
        help=("Search for primer pairs and guides between them. This "
              "outputs the best BEST_N_TARGETS according to a cost "
              "function, where each target contains primers that bound "
              "an amplicon and a guide set within that amplicon."))
    parser_ct_args = argparse.ArgumentParser(add_help=False)
    parser_ct_args.add_argument('-pl', '--primer-length', type=int, default=30,
        help=("Length of primer in nt"))
    parser_ct_args.add_argument('-pp', '--primer-cover-frac',
        type=check_cover_frac, default=1.0,
        help=("Same as --cover-frac, except for the design of primers -- "
              "i.e., the fract_argsion of sequences that must be covered "
              "by the primers, independently on each end"))
    parser_ct_args.add_argument('-pm', '--primer-mismatches',
        type=int, default=0,
        help=("Allow for this number of mismatches when determining "
              "whether a primer hybridizes to a sequence"))
    parser_ct_args.add_argument('--max-primers-at-site', type=int,
        help=("Only use primer sites that contain at most this number "
              "of primers; if not set, there is no limit"))
    parser_ct_args.add_argument('--max-target-length', type=int,
        help=("Only allow amplicons (incl. primers) to be at most this "
              "number of nucleotides long; if not set, there is no limit"))
    parser_ct_args.add_argument('--cost-fn-weights', type=float, nargs=3,
        help=("Specify custom weights in the cost funct_argsion; given as "
              "3 weights (A B C), where the cost funct_argsion is "
              "A*(total number of primers) + B*log2(amplicon length) + "
              "C*(number of guides)"))
    parser_ct_args.add_argument('--best-n-targets', type=int, default=10,
        help=("Only compute and output up to this number of targets. Note "
              "that runtime will generally be longer for higher values"))
    ###########################################################################

    ###########################################################################
    # SUBCOMMANDS FOR INPUT TYPE
    ###########################################################################
    search_cmd_parsers = [(parser_sw, parser_sw_args),
                          (parser_ct, parser_ct_args)]

    # FASTA input
    input_fasta_subparser = argparse.ArgumentParser(add_help=False)
    input_fasta_subparser.add_argument('in_fasta', nargs='+',
        help=("Path to input FASTA. More than one can be "
              "given for differential identification"))
    input_fasta_subparser.add_argument('-o', '--out-tsv',
        nargs='+', required=True,
        help=("Path to output TSV. If more than one input FASTA is given, the "
              "same number of output TSVs must be given; each output TSV "
              "corresponds to an input FASTA."))

    # Auto prepare, common arguments
    input_auto_common_subparser = argparse.ArgumentParser(add_help=False)
    input_auto_common_subparser.add_argument('--mafft-path',
        required=True,
        help=("Path to mafft executable, used for generating alignments"))
    input_auto_common_subparser.add_argument('--prep-memoize-dir',
        help=("Path to directory in which to memoize alignments and "
              "statistics on them; if not set, this does not memoize "
              "this information"))
    input_auto_common_subparser.add_argument('--limit-seqs', type=int,
        help=("After fetching accessions, randomly select LIMIT_SEQS of them "
              "without replacement from each taxonomy and only move forward "
              "in the design with these. This is useful for testing and "
              "measuring output growth as input size grows."))
    input_auto_common_subparser.add_argument('--prep-influenza',
        action='store_true',
        help=("If set, fetch sequences using the NCBI Influenza database; "
              "should only be used for Influenza A or B virus taxonomies"))

    # Auto prepare from file
    input_autofile_subparser = argparse.ArgumentParser(add_help=False)
    input_autofile_subparser.add_argument('in_tsv',
        help=("Path to input TSV. Each row gives the following columns, "
              "in order: (1) label for the row (used for naming output "
              "files; must be unique); (2) taxonomic (e.g., species) ID from "
              "NCBI; (3) label of segment (e.g., 'S') if there is one, or "
              "'None' if unsegmented; (4) accession of reference sequence to "
              "use for curation"))
    input_autofile_subparser.add_argument('out_tsv_dir',
        help=("Path to directory in which to place output TSVs; each "
              "output TSV corresponds to a row in the input"))

    # Auto prepare from arguments
    input_autoargs_subparser = argparse.ArgumentParser(add_help=False)
    input_autoargs_subparser.add_argument('tax_id', type=int,
        help=("Taxonomic (e.g., species) ID from NCBI"))
    input_autoargs_subparser.add_argument('segment',
        help=("Label of segment (e.g., 'S') if there is one, or 'None' if "
              "unsegmented"))
    input_autoargs_subparser.add_argument('ref_acc',
        help=("Accession of reference sequence to use for curation"))
    input_autoargs_subparser.add_argument('out_tsv',
        help=("Path to output TSV"))

    # Add parsers for subcommands
    for search_cmd_parser, search_cmd_parser_args in search_cmd_parsers:
        parents = [base_subparser, search_cmd_parser_args]

        search_cmd_subparser = search_cmd_parser.add_subparsers(
            dest='input_type')
        search_cmd_subparser.add_parser('fasta',
            parents=parents + [input_fasta_subparser],
            help=("Search from a given alignment input as a FASTA file"))
        search_cmd_subparser.add_parser('auto-from-file',
            parents=parents + [input_auto_common_subparser, input_autofile_subparser],
            help=("Automatically fetch sequences for one or more "
                  "taxonomies, then curate and align each; use these "
                  "alignments as input. The information is provided in "
                  "a TSV file. Differential identification is performed "
                  "across the taxonomies."))
        search_cmd_subparser.add_parser('auto-from-args',
            parents=parents + [input_auto_common_subparser, input_autoargs_subparser],
            help=("Automatically fetch sequences for one taxonomy, then curate "
                  "and align them; use this alignment as input. The "
                  "taxonomy is provided as command-line arguments."))
    ###########################################################################

    args = parser.parse_args()

    log.configure_logging(args.log_level)
    main(args)
