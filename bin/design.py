#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
from collections import defaultdict
import logging
import os
import re
import shutil
import tempfile

from adapt import alignment
from adapt import guide_search
from adapt.prepare import align
from adapt.prepare import prepare_alignment
from adapt import primer_search
from adapt import target_search
from adapt.utils import guide
from adapt.utils import log
from adapt.utils import seq_io
from adapt.utils import year_cover

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
        as determined by args.cover_by_year_decay (if args.search_cmd
        is 'complete-targets', then cover_frac is a tuple (g, p) where
        g is the previously described dict calculated from
        args.guide_cover_frac and p is the same calculated from
        args.primer_cover_frac)
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
    guide_cover_frac = year_cover.construct_partial_covers(
        years.keys(), year_highest_cover, args.guide_cover_frac, year_cover_decay)

    if args.search_cmd == 'complete-targets':
        primer_cover_frac = year_cover.construct_partial_covers(
            years.keys(), year_highest_cover, args.primer_cover_frac, year_cover_decay)
        cover_frac = (guide_cover_frac, primer_cover_frac)
    else:
        cover_frac = guide_cover_frac

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
        tuple (in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs out_tsv) in
        which in_fasta is a list of paths to fasta files each containing an
        alignment, taxid_for_fasta[i] gives a taxon id for in_fasta[i],
        years_tsv gives a tempfile storing a tsv file containing a year for
        each sequence across all the fasta files (only if
        args.cover_by_year_decay is set), aln_tmp_dirs is a list of temp
        directories that need to be cleaned up, and out_tsv[i] is a path to the
        file at which to write the output for in_fasta[i]
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
        align_stat_memoize_dir = os.path.join(args.prep_memoize_dir, 'stats')
        if not os.path.exists(align_stat_memoize_dir):
            os.makedirs(align_stat_memoize_dir)

        am = align.AlignmentMemoizer(align_memoize_dir)
        asm = align.AlignmentStatMemoizer(align_stat_memoize_dir)
    else:
        am = None
        asm = None

    # Read list of taxonomies
    if args.input_type == 'auto-from-args':
        s = None if args.segment == 'None' else args.segment
        ref_accs = args.ref_accs.split(',')
        taxs = [(None, args.tax_id, s, ref_accs)]
    elif args.input_type == 'auto-from-file':
        taxs = seq_io.read_taxonomies(args.in_tsv)
    else:
        raise Exception(("Unknown input type '%s'") % args.input_type)

    # Read specified accessions, if provided
    if args.use_accessions:
        accessions_to_use = seq_io.read_accessions_for_taxonomies(
                args.use_accessions)
    else:
        accessions_to_use = None

    # Construct alignments for each taxonomy
    in_fasta = []
    taxid_for_fasta = []
    years_tsv_per_aln = []
    aln_tmp_dirs = []
    out_tsv = []
    for label, tax_id, segment, ref_accs in taxs:
        aln_file_dir = tempfile.TemporaryDirectory()
        if args.cover_by_year_decay:
            years_tsv_tmp = tempfile.NamedTemporaryFile()
            years_tsv_tmp_name = years_tsv_tmp.name
        else:
            years_tsv_tmp = None
            years_tsv_tmp_name = None

        if accessions_to_use is not None:
            if (tax_id, segment) in accessions_to_use:
                accessions_to_use_for_tax = accessions_to_use[(tax_id, segment)]
            else:
                accessions_to_use_for_tax = None
        else:
            accessions_to_use_for_tax = None

        nc = prepare_alignment.prepare_for(
            tax_id, segment, ref_accs,
            aln_file_dir.name, aln_memoizer=am, aln_stat_memoizer=asm,
            sample_seqs=args.sample_seqs, prep_influenza=args.prep_influenza,
            years_tsv=years_tsv_tmp_name,
            cluster_threshold=args.cluster_threshold,
            accessions_to_use=accessions_to_use_for_tax)

        for i in range(nc):
            in_fasta += [os.path.join(aln_file_dir.name, str(i) + '.fasta')]
            taxid_for_fasta += [tax_id]
        years_tsv_per_aln += [years_tsv_tmp]
        aln_tmp_dirs += [aln_file_dir]

        if label is None:
            out_tsv += [args.out_tsv + '.' + str(i) for i in range(nc)]
        else:
            for i in range(nc):
                out_name = label + '.' + str(i) + '.tsv'
                out_tsv += [os.path.join(args.out_tsv_dir, out_name)]

        if args.write_input_seqs:
            # Write the sequences that are in the alignment being used
            # as input
            all_seq_names = []
            for i in range(nc):
                fn = os.path.join(aln_file_dir.name, str(i) + '.fasta')
                seqs = seq_io.read_fasta(fn)
                all_seq_names += list(seqs.keys())
            all_seq_names = sorted(all_seq_names)
            if label is None:
                # args.write_input_seqs gives the path to where to write
                # the list
                out_file = args.write_input_seqs
            else:
                # Determine where to write the sequence names based on
                # the label and args.out_tsv_dir
                out_name = label + '.input-sequences.txt'
                out_file = os.path.join(args.out_tsv_dir, out_name)
            with open(out_file, 'w') as fw:
                for name in all_seq_names:
                    fw.write(name + '\n')
        if args.write_input_aln:
            # Write the alignments being used as input
            for i in range(nc):
                fn = os.path.join(aln_file_dir.name, str(i) + '.fasta')
                if label is None:
                    # args.write_input_aln gives the prefix of the path to
                    # which to write the alignment
                    copy_path = args.write_input_aln + '.' + str(i)
                else:
                    # Determine where to write the alignment based on the
                    # label and args.out_tsv_dir
                    out_name = label + '.' + str(i) + '.fasta'
                    copy_path = os.path.join(args.out_tsv_dir, out_name)
                shutil.copyfile(fn, copy_path)

    # Combine all years tsv (there is one per fasta file)
    if any(f is not None for f in years_tsv_per_aln):
        years_tsv = tempfile.NamedTemporaryFile()
        with open(years_tsv.name, 'w') as fw:
            for tf in years_tsv_per_aln:
                if tf is not None:
                    with open(tf.name) as fin:
                        for line in fin:
                            fw.write(line)
                    tf.close()
    else:
        years_tsv = None

    return in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs, out_tsv


def design_for_id(args):
    """Design guides for differential identification across targets.

    Args:
        args: namespace of arguments provided to this executable
    """
    # Create an alignment object for each input
    alns = []
    seq_groups_per_input = []
    guide_cover_frac_per_input = []
    primer_cover_frac_per_input = []
    for in_fasta in args.in_fasta:
        seqs = seq_io.read_fasta(in_fasta)
        if args.cover_by_year_decay:
            aln, seq_groups, cover_frac = seqs_grouped_by_year(seqs, args)
            if args.search_cmd == 'complete-targets':
                guide_cover_frac, primer_cover_frac = cover_frac
            else:
                guide_cover_frac = cover_frac
                primer_cover_frac = None
        else:
            aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
            seq_groups = None
            guide_cover_frac = args.guide_cover_frac
            if args.search_cmd == 'complete-targets':
                primer_cover_frac = args.primer_cover_frac
            else:
                primer_cover_frac = None
        alns += [aln]
        seq_groups_per_input += [seq_groups]
        guide_cover_frac_per_input += [guide_cover_frac]
        primer_cover_frac_per_input += [primer_cover_frac]

    # Also add the specific_against alignments into alns, but keep
    # track of how many use for design (the first N of them)
    num_aln_for_design = len(args.in_fasta)
    for specific_against_fasta in args.specific_against:
        seqs = seq_io.read_fasta(specific_against_fasta)
        aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))
        alns += [aln]

    required_guides, blacklisted_ranges, blacklisted_kmers = \
        parse_required_guides_and_blacklist(args)
    required_flanking_seqs = (args.require_flanking5, args.require_flanking3)

    # Allow G-U base pairing, unless it is explicitly disallowed
    allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Assign an id in [0, 1, 2, ...] for each taxid
    # Find all alignments with each taxid
    aln_with_taxid = defaultdict(set)
    for i, taxid in enumerate(args.taxid_for_fasta):
        aln_with_taxid[taxid].add(i)
    num_taxa = len(aln_with_taxid)
    logger.info(("Designing for %d taxa"), num_taxa)

    # Construct the data structure for guide queries to perform
    # differential identification
    if num_taxa > 1:
        logger.info(("Constructing data structure to permit guide queries for "
            "differential identification"))
        aq = alignment.AlignmentQuerier(alns, args.guide_length,
            args.diff_id_mismatches, allow_gu_pairs)
        aq.setup()
    else:
        logger.info(("Only one taxon was provided, so not constructing "
            "data structure to permit queries for differential "
            "identification"))
        aq = None

    for i in range(num_aln_for_design):
        taxid = args.taxid_for_fasta[i]
        logger.info(("Finding guides for alignment %d (of %d), which is in "
            "taxon %d"), i + 1, num_aln_for_design, taxid)

        aln = alns[i]
        seq_groups = seq_groups_per_input[i]
        guide_cover_frac = guide_cover_frac_per_input[i]
        primer_cover_frac = primer_cover_frac_per_input[i]
        required_guides_for_aln = required_guides[i]
        blacklisted_ranges_for_aln = blacklisted_ranges[i]
        alns_in_same_taxon = aln_with_taxid[taxid]
        memoized_specificity_results = {}

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
                if guide in memoized_specificity_results:
                    return memoized_specificity_results[guide]
                else:
                    is_spec = aq.guide_is_specific_to_alns(
                        guide, alns_in_same_taxon, args.diff_id_frac)
                    memoized_specificity_results[guide] = is_spec
                    return is_spec
            else:
                return True

        # Mask alignments from this taxon from being reported in queries
        # because we will likely get many guide sequences that hit its
        # alignments, but we do not care about these for checking specificity
        if aq is not None:
            for j in alns_in_same_taxon:
                aq.mask_aln(j)

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file; ensure that the selected guides are
        # specific to this alignment
        gs = guide_search.GuideSearcher(aln, args.guide_length,
                                        args.guide_mismatches,
                                        guide_cover_frac, args.missing_thres,
                                        guide_is_suitable_fn=guide_is_suitable,
                                        seq_groups=seq_groups,
                                        required_guides=required_guides_for_aln,
                                        blacklisted_ranges=blacklisted_ranges_for_aln,
                                        allow_gu_pairs=allow_gu_pairs,
                                        required_flanking_seqs=required_flanking_seqs,
                                        predict_activity_model_path=args.predict_activity_model_path)

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
                                              primer_cover_frac,
                                              args.missing_thres,
                                              seq_groups=seq_groups)
            ts = target_search.TargetSearcher(ps, gs,
                max_primers_at_site=args.max_primers_at_site,
                max_target_length=args.max_target_length,
                cost_weights=args.cost_fn_weights,
                guides_should_cover_over_all_seqs=args.gp_over_all_seqs)
            ts.find_and_write_targets(args.out_tsv[i],
                best_n=args.best_n_targets)
        else:
            raise Exception("Unknown search subcommand '%s'" % args.search_cmd)

        # i should no longer be masked from queries
        if aq is not None:
            aq.unmask_all_aln()


def main(args):
    logger = logging.getLogger(__name__)

    logger.info("Running design.py with arguments: %s", args)

    if args.input_type in ['auto-from-file', 'auto-from-args']:
        if args.input_type == 'auto-from-file':
            if not os.path.isdir(args.out_tsv_dir):
                raise Exception(("Output directory '%s' does not exist") %
                    args.out_tsv_dir)

        # Prepare input alignments, stored in temp fasta files
        in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs, out_tsv = prepare_alignments(args)
        args.in_fasta = in_fasta
        args.taxid_for_fasta = taxid_for_fasta
        args.out_tsv = out_tsv

        if args.cover_by_year_decay:
            # args.cover_by_year_decay contains two parameters: the year
            # with the highest cover and the decay; add in (to the beginning)
            # the file listing the years
            year_highest_cover, year_cover_decay = args.cover_by_year_decay
            args.cover_by_year_decay = (years_tsv.name, year_highest_cover,
                    year_cover_decay)
    elif args.input_type == 'fasta':
        if len(args.in_fasta) != len(args.out_tsv):
            raise Exception(("Number output TSVs must match number of input "
                "FASTAs"))
        args.taxid_for_fasta = list(range(len(args.in_fasta)))
    else:
        raise Exception("Unknown input type subcommand '%s'" % args.input_type)

    design_for_id(args)

    # Close temporary files storing alignments
    if args.input_type in ['auto_from_file', 'auto-from-args']:
        for td in aln_tmp_dirs:
            td.cleanup()
        if years_tsv is not None:
            years_tsv.close()


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
              "(GUIDE_COVER_FRAC * (2 * PRIMER_COVER_FRAC - 1)) sequences will "
              "be covered)."))

    # Automatically setting desired coverage of target sequences based
    # on their year
    class ParseCoverDecayWithYearsFile(argparse.Action):
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
    class ParseCoverDecayByGeneratingYearsFile(argparse.Action):
        # This is needed because --cover-by-year-decay has multiple args
        # of different types
        def __call__(self, parser, namespace, values, option_string=None):
            a, b = values
            # Check that a is a valid year
            year_pattern = re.compile('^(\d{4})$')
            if year_pattern.match(a):
                ai = int(a)
            else:
                raise argparse.ArgumentTypeError(("%s is an invalid 4-digit "
                    "year") % a)
            # Check that b is a valid decay
            bf = float(b)
            if bf <= 0 or bf >= 1:
                raise argparse.ArgumentTypeError(("%s is an invalid decay; it "
                    "must be a float in (0,1)" % b))
            setattr(namespace, self.dest, (ai, bf))

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

    # Requiring flanking sequence (PFS)
    base_subparser.add_argument('--require-flanking5',
        help=("Require the given sequence on the 5' protospacer flanking "
              "site (PFS) of each designed guide; this tolerates ambiguity "
              "in the sequence (e.g., 'H' requires 'A', 'C', or 'T', or, "
              "equivalently, avoids guides flanked by 'G'). Note that "
              "this is the 5' end in the target sequence (not the spacer "
              "sequence)."))
    base_subparser.add_argument('--require-flanking3',
        help=("Require the given sequence on the 3' protospacer flanking "
              "site (PFS) of each designed guide; this tolerates ambiguity "
              "in the sequence (e.g., 'H' requires 'A', 'C', or 'T', or, "
              "equivalently, avoids guides flanked by 'G'). Note that "
              "this is the 3' end in the target sequence (not the spacer "
              "sequence)."))

    # Use a model to predict activity
    base_subparser.add_argument('--predict-activity-model-path',
        help=("Path to directory containing model hyperparameters and "
              "trained weights to use to to predict guide-target activity; "
              "only consider guides to be active is the activity exceeds "
              "a threshold (specified in the 'predict_activity' module)"))

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
              "i.e., the fraction of sequences that must be covered "
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
        help=("Specify custom weights in the cost function; given as "
              "3 weights (A B C), where the cost funct_argsion is "
              "A*(total number of primers) + B*log2(amplicon length) + "
              "C*(number of guides)"))
    parser_ct_args.add_argument('--best-n-targets', type=int, default=10,
        help=("Only compute and output up to this number of targets. Note "
              "that runtime will generally be longer for higher values"))
    parser_ct_args.add_argument('--gp-over-all-seqs',
        action='store_true',
        help=("If set, design the guides so as to cover GUIDE_COVER_FRAC "
              "of *all* sequences, rather than GUIDE_COVER_FRAC of just "
              "the sequences covered by the primers. This changes the "
              "behavior of -gp/--guide-cover-frac. It may lead to "
              "more than the optimal number of guides because it requires "
              "covering more sequences. However, it may improve runtime "
              "because the the sequences to consider for guide design will "
              "be more similar across amplicons and therefore designs can "
              "be more easily memoized."))
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
    input_fasta_subparser.add_argument('--cover-by-year-decay', nargs=3,
        action=ParseCoverDecayWithYearsFile,
        help=("<A> <B> <C>; if set, group input sequences by year and set a "
              "desired partial cover for each year (fraction of sequences that "
              "must be covered by guides) as follows: A is a tsv giving "
              "a year for each input sequence (col 1 is sequence name "
              "matching that in the input FASTA, col 2 is year). All years "
              ">= A receive a desired cover fraction of GUIDE_COVER_FRAC "
              "for guides (and PRIMER_COVER_FRAC for primers). Each preceding "
              "year receives a desired cover fraction that decays by B -- "
              "i.e., year n is given B*(desired cover fraction of year n+1)."))

    # Auto prepare, common arguments
    input_auto_common_subparser = argparse.ArgumentParser(add_help=False)
    input_auto_common_subparser.add_argument('--mafft-path',
        required=True,
        help=("Path to mafft executable, used for generating alignments"))
    input_auto_common_subparser.add_argument('--prep-memoize-dir',
        help=("Path to directory in which to memoize alignments and "
              "statistics on them; if not set, this does not memoize "
              "this information"))
    input_auto_common_subparser.add_argument('--sample-seqs', type=int,
        help=("After fetching accessions, randomly select SAMPLE_SEQS of them "
              "with replacement from each taxonomy any move forward "
              "in the design with these. This is useful for testing and "
              "measuring output growth as input size grows, as well as "
              "assessing the dispersion in output associated with some "
              "input sample."))
    input_auto_common_subparser.add_argument('--prep-influenza',
        action='store_true',
        help=("If set, fetch sequences using the NCBI Influenza database; "
              "should only be used for Influenza A or B virus taxonomies"))
    input_auto_common_subparser.add_argument('--cover-by-year-decay', nargs=2,
        action=ParseCoverDecayByGeneratingYearsFile,
        help=("<A> <B>; if set, group input sequences by year and set a "
              "desired partial cover for each year (fraction of sequences that "
              "must be covered by guides) as follows: All years "
              ">= A receive a desired cover fraction of GUIDE_COVER_FRAC "
              "for guides (and PRIMER_COVER_FRAC for primers). Each preceding "
              "year receives a desired cover fraction that decays by B -- "
              "i.e., year n is given B*(desired cover fraction of year n+1)."))
    input_auto_common_subparser.add_argument('--cluster-threshold',
        type=float,
        default=0.2,
        help=(("Maximum inter-cluster distance to use when clustering "
               "input sequences prior to alignment. Expressed as average "
               "nucleotide dissimilarity (1-ANI, where ANI is average "
               "nucleotide identity); higher values result in fewer "
               "clusters")))
    input_auto_common_subparser.add_argument('--use-accessions',
        help=("If set, use specified accessions instead of fetching neighbors "
              "for the given taxonomic ID(s). This provides a path to a TSV "
              "file with 3 columns: (1) a taxonomic ID; (2) segment label, "
              "or 'None' if unsegmented; (3) accession. Each row specifies "
              "an accession to use in the input, and values for columns 1 "
              "and 2 can appear in multiple rows."))

    # Auto prepare from file
    input_autofile_subparser = argparse.ArgumentParser(add_help=False)
    input_autofile_subparser.add_argument('in_tsv',
        help=("Path to input TSV. Each row gives the following columns, "
              "in order: (1) label for the row (used for naming output "
              "files; must be unique); (2) taxonomic (e.g., species) ID from "
              "NCBI; (3) label of segment (e.g., 'S') if there is one, or "
              "'None' if unsegmented; (4) accessions of reference sequences to "
              "use for curation (comma-separated)"))
    input_autofile_subparser.add_argument('out_tsv_dir',
        help=("Path to directory in which to place output TSVs; each "
              "output TSV corresponds to a cluster for the taxon in a row "
              "in the input"))
    input_autofile_subparser.add_argument('--write-input-seqs',
        action='store_true',
        help=("If set, write the sequences (accession.version) being used as "
              "input for design to a file in OUT_TSV_DIR; the filename is "
              "determined based on the label for each taxonomy"))
    input_autofile_subparser.add_argument('--write-input-aln',
        action='store_true',
        help=("If set, write the alignments being used as "
              "input for design to a file in OUT_TSV_DIR; the filename is "
              "determined based on the label for each taxonomy (they are "
              "'[label].[cluster-number].fasta'"))

    # Auto prepare from arguments
    input_autoargs_subparser = argparse.ArgumentParser(add_help=False)
    input_autoargs_subparser.add_argument('tax_id', type=int,
        help=("Taxonomic (e.g., species) ID from NCBI"))
    input_autoargs_subparser.add_argument('segment',
        help=("Label of segment (e.g., 'S') if there is one, or 'None' if "
              "unsegmented"))
    input_autoargs_subparser.add_argument('ref_accs',
        help=("Accessions of reference sequences to use for curation (comma-"
              "separated)"))
    input_autoargs_subparser.add_argument('out_tsv',
        help=("Path to output TSVs, with one per cluster; output TSVs are "
              "OUT_TSV.{cluster-number}"))
    input_autoargs_subparser.add_argument('--write-input-seqs',
        help=("Path to a file to which to write the sequences "
              "(accession.version) being used as input for design"))
    input_autoargs_subparser.add_argument('--write-input-aln',
        help=("Prefix of path to files to which to write the alignments "
              "being used as input for design; filenames are "
              "'WRITE_INPUT_ALN.[cluster-number]'"))

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
