#!/usr/bin/env python3
"""Design guides for diagnostics."""

import argparse
from collections import defaultdict
import logging
import os
import random
import re
import shutil
import sys
import tempfile

import numpy as np

from adapt import alignment
from adapt import guide_search
from adapt.prepare import align
from adapt.prepare import ncbi_neighbors
from adapt.prepare import prepare_alignment
from adapt import primer_search
from adapt.specificity import alignment_query
from adapt import target_search
from adapt.utils import guide
from adapt.utils import log
from adapt.utils import predict_activity
from adapt.utils import seq_io
from adapt.utils import year_cover

try:
    import boto3
    from botocore.exceptions import ClientError
except:
    cloud = False
else:
    cloud = True

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


# Define defaults for parameters associated with different objectives
# The reason to define them here, rather than in argparse, is that
# defining them outside of argparse lets us check whether the
# argument has been set (versus just set to its default)
OBJ_PARAM_DEFAULTS = {
        'minimize-guides': {
            'guide_mismatches': 0,
            'guide_cover_frac': 1.0
        },
        'maximize-activity': {
            'guide_mismatches': 0,
            'soft_guide_constraint': 1,
            'hard_guide_constraint': 5,
            'penalty_strength': 0.25,
            'maximization_algorithm': 'random-greedy'
        }
}


def check_obj_args(args):
    """Check argument depending on the objective function and set defaults.

    Raise critical messages if arguments are set for the wrong objective
    function than what is specified, and set defaults for these arguments if
    they are not specified

    Args:
        args: namespace of arguments provided to this executable
    """
    if args.obj == 'minimize-guides':
        if (args.soft_guide_constraint or args.hard_guide_constraint or
                args.penalty_strength or args.maximization_algorithm):
            logger.critical(("When --obj is 'minimize-guides', the "
                "following arguments are not used: --soft-guide-constraint, "
                "--hard-guide-constraint, --penalty-strength, "
                "--maximization-algorithm"))
        if args.use_simple_binary_activity_prediction:
            raise Exception(("Cannot use --use-simple-binary-activity-prediction "
                "when --obj is 'minimize-guides'"))
        for arg in OBJ_PARAM_DEFAULTS['minimize-guides'].keys():
            if vars(args)[arg] is None:
                vars(args)[arg] = OBJ_PARAM_DEFAULTS['minimize-guides'][arg]

    if args.obj == 'maximize-activity':
        if args.use_simple_binary_activity_prediction:
            if (args.guide_cover_frac or args.cover_by_year_decay):
                logger.critical(("When --obj is 'maximize-activity', the "
                    "following arguments are not used: "
                    "--guide-cover-frac, --cover-by-year-decay"))
            if args.predict_activity_model_path:
                logger.critical(("When --use-simple-binary-activity-prediction "
                    "is set, --predict-activity-model-path is not used"))
        else:
            if (args.guide_mismatches or args.guide_cover_frac or
                    args.cover_by_year_decay):
                logger.critical(("When --obj is 'maximize-activity', the "
                    "following arguments are not used: --guide-mismatches, "
                    "--guide-cover-frac, --cover-by-year-decay"))
        for arg in OBJ_PARAM_DEFAULTS['maximize-activity'].keys():
            if vars(args)[arg] is None:
                vars(args)[arg] = OBJ_PARAM_DEFAULTS['maximize-activity'][arg]


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
        tuple (in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs, out_tsv, design_for)
        in which in_fasta is a list of paths to fasta files each containing an
        alignment, taxid_for_fasta[i] gives a taxon id for in_fasta[i],
        years_tsv gives a tempfile storing a tsv file containing a year for
        each sequence across all the fasta files (only if
        args.cover_by_year_decay is set), aln_tmp_dirs is a list of temp
        directories that need to be cleaned up, out_tsv[i] is a path to the
        file at which to write the output for in_fasta[i], and design_for[i]
        indicates whether to actually design for in_fasta[i] (True/False)
    """
    logger.info(("Setting up to prepare alignments"))

    # Set the path to mafft
    align.set_mafft_exec(args.mafft_path)

    # Setup alignment and alignment stat memoizers
    if args.prep_memoize_dir:
        if args.prep_memoize_dir[:5] == "s3://":
            bucket = args.prep_memoize_dir.split("/")[2]
            try:
                if args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
                    S3 = boto3.client("s3", aws_access_key_id = args.aws_access_key_id,
                        aws_secret_access_key = args.aws_secret_access_key)
                else:
                    S3 = boto3.client("s3")
                S3.head_bucket(Bucket=bucket)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    S3.create_bucket(Bucket=bucket)
                elif e.response['Error']['Code'] == "403":
                    raise Exception(("Incorrect AWS Access Key ID or Secret Access Key")) from e
                else:
                    raise
            except ConnectionError as e:
                raise Exception(("Cannot connect to Amazon S3")) from e
            if args.prep_memoize_dir[-1] == "/":
                align_memoize_dir = '%saln' %args.prep_memoize_dir
                align_stat_memoize_dir = '%sstats' %args.prep_memoize_dir
            else:
                align_memoize_dir = '%s/aln' %args.prep_memoize_dir
                align_stat_memoize_dir = '%s/stats' %args.prep_memoize_dir
        else:
            if not os.path.isdir(args.prep_memoize_dir):
                raise Exception(("Path '%s' does not exist") %
                    args.prep_memoize_dir)
            align_memoize_dir = os.path.join(args.prep_memoize_dir, 'aln')
            if not os.path.exists(align_memoize_dir):
                os.makedirs(align_memoize_dir)
            align_stat_memoize_dir = os.path.join(args.prep_memoize_dir, 'stats')
            if not os.path.exists(align_stat_memoize_dir):
                os.makedirs(align_stat_memoize_dir)

        am = align.AlignmentMemoizer(align_memoize_dir, 
            aws_access_key_id = args.aws_access_key_id,
            aws_secret_access_key = args.aws_secret_access_key)
        asm = align.AlignmentStatMemoizer(align_stat_memoize_dir, 
            aws_access_key_id = args.aws_access_key_id,
            aws_secret_access_key = args.aws_secret_access_key)
    else:
        am = None
        asm = None

    # Read list of taxonomies
    if args.input_type == 'auto-from-args':
        s = None if args.segment == 'None' else args.segment
        ref_accs = ncbi_neighbors.construct_references(args.tax_id) \
            if not args.ref_accs else args.ref_accs
        meta_filt = None
        meta_filt_against = None
        if args.metadata_filter:
            meta_filt = seq_io.read_metadata_filters(args.metadata_filter)
        if args.specific_against_metadata_filter:
            meta_filt_against = seq_io.read_metadata_filters(args.specific_against_metadata_filter)
        taxs = [(None, args.tax_id, s, ref_accs, meta_filt, meta_filt_against)]
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

    # Read specified sequences, if provided
    if args.use_fasta:
        sequences_to_use = seq_io.read_sequences_for_taxonomies(
                args.use_fasta)
    else:
        sequences_to_use = None

    # Only design for certain taxonomies, if provided
    if args.only_design_for:
        taxs_to_design_for = seq_io.read_taxonomies_to_design_for(
                args.only_design_for)
    else:
        taxs_to_design_for = None

    # Construct alignments for each taxonomy
    in_fasta = []
    taxid_for_fasta = []
    years_tsv_per_aln = []
    aln_tmp_dirs = []
    out_tsv = []
    design_for = []
    specific_against_metadata_accs = []
    for label, tax_id, segment, ref_accs, meta_filt, meta_filt_against in taxs:
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

        if sequences_to_use is not None:
            if (tax_id, segment) in sequences_to_use:
                sequences_to_use_for_tax = sequences_to_use[(tax_id, segment)]
            else:
                sequences_to_use_for_tax = None
        else:
            sequences_to_use_for_tax = None

        if (accessions_to_use_for_tax is not None and
                sequences_to_use_for_tax is not None):
            raise Exception(("Cannot use both --use-accessions and "
                "--use-fasta for the same taxonomy"))

        nc, specific_against_metadata_acc = prepare_alignment.prepare_for(
            tax_id, segment, ref_accs,
            aln_file_dir.name, aln_memoizer=am, aln_stat_memoizer=asm,
            sample_seqs=args.sample_seqs, 
            prep_influenza=args.prep_influenza,
            years_tsv=years_tsv_tmp_name,
            cluster_threshold=args.cluster_threshold,
            accessions_to_use=accessions_to_use_for_tax,
            sequences_to_use=sequences_to_use_for_tax,
            meta_filt=meta_filt, 
            meta_filt_against=meta_filt_against)

        for i in range(nc):
            in_fasta += [os.path.join(aln_file_dir.name, str(i) + '.fasta')]
            taxid_for_fasta += [tax_id]
            specific_against_metadata_accs.append(specific_against_metadata_acc)
            if taxs_to_design_for is None:
                design_for += [True]
            else:
                design_for += [(tax_id, segment) in taxs_to_design_for]
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

    return in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs, out_tsv, design_for, specific_against_metadata_accs


def design_for_id(args):
    """Design guides for differential identification across targets.

    Args:
        args: namespace of arguments provided to this executable
    """
    # Create an alignment object for each input
    # If obj is not 'minimize-guides', guide_cover_frac and seq_groups
    #  will be set to None
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

    # Keep track of how many items in alns to use for design (the first N of them)
    num_aln_for_design = len(args.in_fasta)

    # Add sequence lists to alns to be specific against
    # Note that although these are being place in alns, they may not
    # be actual alignments (they can be unaligned sequences)
    # If specific-against-metadata-filter is specified, add metadata filtered accessions to alns
    # and store what index it is located at in alns. Also, store start and end indices of
    # specific_against_metadata sequence lists
    specific_against_metadata_indices = {}
    specific_against_metadata_start = len(alns)
    for i in range(num_aln_for_design):
        if len(args.specific_against_metadata_accs[i]) > 0:
            logger.info(("Fetching %d sequences within taxon %d to be specific against"), 
                len(args.specific_against_metadata_accs[i]), args.taxid_for_fasta[i])
            seqs = prepare_alignment.fetch_sequences_for_acc_list(list(
                args.specific_against_metadata_accs[i]))
            seqs_list = alignment.SequenceList(list(seqs.values()))
            alns += [seqs_list]
            specific_against_metadata_indices[i] = len(alns) - 1
    specific_against_metadata_end = len(alns)
    # Also add the specific_against_fastas sequences into alns
    for specific_against_fasta in args.specific_against_fastas:
        seqs = seq_io.read_fasta(specific_against_fasta)
        seqs_list = alignment.SequenceList(list(seqs.values()))
        alns += [seqs_list]
    # Finally, add specific_against_taxa sequences into alns after
    # downloading them
    if args.specific_against_taxa:
        taxs_to_be_specific_against = seq_io.read_taxonomies_to_design_for(
                args.specific_against_taxa)
        for taxid, segment in taxs_to_be_specific_against:
            logger.info(("Fetching sequences to be specific against tax "
                "%d (segment: %s)"), taxid, segment)
            seqs = prepare_alignment.fetch_sequences_for_taxonomy(
                    taxid, segment)
            seqs_list = alignment.SequenceList(list(seqs.values()))
            alns += [seqs_list]

    specific_against_exists = ((len(args.specific_against_fastas) > 0 or
            args.specific_against_taxa is not None) or 
            (specific_against_metadata_end - specific_against_metadata_start) > 0)

    required_guides, blacklisted_ranges, blacklisted_kmers = \
        parse_required_guides_and_blacklist(args)
    required_flanking_seqs = (args.require_flanking5, args.require_flanking3)

    # Allow G-U base pairing, unless it is explicitly disallowed
    allow_gu_pairs = not args.do_not_allow_gu_pairing

    # Assign an id in [0, 1, 2, ...] for each taxid to design for
    # Find all alignments with each taxid
    aln_with_taxid = defaultdict(set)
    for i, taxid in enumerate(args.taxid_for_fasta):
        aln_with_taxid[taxid].add(i)
    num_taxa = len(aln_with_taxid)
    logger.info(("Designing for %d taxa"), num_taxa)

    # Read taxonomies to ignore for specificity, if specified
    tax_ignore = {}
    if (args.input_type in ['auto-from-file', 'auto-from-args'] and
            args.taxa_to_ignore_for_specificity):
        tax_ignore = seq_io.read_taxonomy_specificity_ignore(
                args.taxa_to_ignore_for_specificity)
        if args.specific_against_fastas or args.specific_against_taxa:
            logger.warning(("Taxa to ignore for specificity cannot from "
                "--specific-against-*"))

    # Construct the data structure for guide queries to perform
    # differential identification
    if num_taxa > 1 or specific_against_exists:
        logger.info(("Constructing data structure to permit guide queries for "
            "differential identification"))
        if args.diff_id_method == "lshnn":
            aq = alignment_query.AlignmentQuerierWithLSHNearNeighbor(alns,
                    args.guide_length, args.diff_id_mismatches, allow_gu_pairs)
        elif args.diff_id_method == "shard":
            aq = alignment_query.AlignmentQuerierWithKmerSharding(alns,
                    args.guide_length, args.diff_id_mismatches, allow_gu_pairs)
        else:
            raise Exception(("Unknown method for querying specificity: '%s'" %
                args.diff_id_method))
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

        if args.design_for is not None and args.design_for[i] is False:
            logger.info("Skipping design for this alignment")
            continue

        aln = alns[i]
        seq_groups = seq_groups_per_input[i]
        guide_cover_frac = guide_cover_frac_per_input[i]
        primer_cover_frac = primer_cover_frac_per_input[i]
        required_guides_for_aln = required_guides[i]
        blacklisted_ranges_for_aln = blacklisted_ranges[i]
        alns_in_same_taxon = aln_with_taxid[taxid]
        # For metadata filtering, we only want to be specific against the 
        # accessions in alns[specific_against_metadata_index]
        specific_against_metadata_index = specific_against_metadata_indices[i] \
            if i in specific_against_metadata_indices else None

        if aq is not None:
            guide_is_specific = aq.guide_is_specific_to_alns_fn(
                    alns_in_same_taxon, args.diff_id_frac,
                    do_not_memoize=args.do_not_memoize_guide_computations)
        else:
            # No specificity to check
            guide_is_specific = lambda guide: True

        def guide_is_suitable(guide):
            # Return True iff the guide does not contain a blacklisted
            # k-mer and is specific to aln

            # Return False if the guide contains a blacklisted k-mer
            for kmer in blacklisted_kmers:
                if kmer in guide:
                    return False

            # Return True if guide does not hit too many sequences in
            # alignments other than aln
            return guide_is_specific(guide)

        # Mask alignments from this taxon from being reported in queries
        # because we will likely get many guide sequences that hit its
        # alignments, but we do not care about these for checking specificity
        if aq is not None:
            for j in range(num_aln_for_design):
                if j in alns_in_same_taxon:
                    aq.mask_aln(j)
                    logger.info(("Masking alignment %d as it is taxon %d"), 
                        j + 1, taxid)
            # Also mask any specific against metadata sequence lists that do not 
            # match the one for this alignment
            if specific_against_metadata_index is not None:
                logger.info(("Masking all sequence lists for specificity except the one "
                    "for alignment %d (taxon %d)"), i + 1, taxid)

            for j in range(specific_against_metadata_start, specific_against_metadata_end):
                if specific_against_metadata_index != j:
                    aq.mask_aln(specific_against_metadata_index)

            # Also mask taxonomies to ignore when determining specificity
            # of taxid
            if taxid in tax_ignore:
                for ignore_taxid in tax_ignore[taxid]:
                    for j in aln_with_taxid[ignore_taxid]:
                        logger.info(("Masking alignment %d (from taxon %d) "
                            "from specificity queries"), j + 1, ignore_taxid)
                        aq.mask_aln(j)

        # Construct activity predictor
        if args.use_simple_binary_activity_prediction:
            predictor = predict_activity.SimpleBinaryPredictor(
                    args.guide_mismatches,
                    allow_gu_pairs,
                    required_flanking_seqs=required_flanking_seqs)
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
        else:
            if args.predict_activity_thres:
                raise Exception(("Cannot set --predict-activity-thres without "
                    "setting --predict-activity-model-path"))
            if args.obj == 'maximize-activity':
                raise Exception(("--predict-activity-model-path must be "
                    "specified if --obj is 'maximize-activity' (unless "
                    "--use-simple-binary-activity-prediction is set)"))
            # Do not predict activity
            predictor = None

        # Find an optimal set of guides for each window in the genome,
        # and write them to a file; ensure that the selected guides are
        # specific to this alignment
        if args.obj == 'minimize-guides':
            gs = guide_search.GuideSearcherMinimizeGuides(
                    aln,
                    args.guide_length,
                    args.guide_mismatches,
                    guide_cover_frac,
                    args.missing_thres,
                    seq_groups=seq_groups,
                    guide_is_suitable_fn=guide_is_suitable,
                    required_guides=required_guides_for_aln,
                    blacklisted_ranges=blacklisted_ranges_for_aln,
                    allow_gu_pairs=allow_gu_pairs,
                    required_flanking_seqs=required_flanking_seqs,
                    predictor=predictor,
                    do_not_memoize_guides=args.do_not_memoize_guide_computations)
        elif args.obj == 'maximize-activity':
            gs = guide_search.GuideSearcherMaximizeActivity(
                    aln,
                    args.guide_length,
                    args.soft_guide_constraint,
                    args.hard_guide_constraint,
                    args.penalty_strength,
                    args.missing_thres,
                    algorithm=args.maximization_algorithm,
                    guide_is_suitable_fn=guide_is_suitable,
                    required_guides=required_guides_for_aln,
                    blacklisted_ranges=blacklisted_ranges_for_aln,
                    allow_gu_pairs=allow_gu_pairs,
                    required_flanking_seqs=required_flanking_seqs,
                    predictor=predictor,
                    do_not_memoize_guides=args.do_not_memoize_guide_computations)

        if args.search_cmd == 'sliding-window':
            # Find an optimal set of guides for each window in the genome,
            # and write them to a file
            gs.find_guides_with_sliding_window(args.window_size,
                args.out_tsv[i],
                window_step=args.window_step,
                sort=args.sort_out,
                print_analysis=(args.log_level==logging.INFO))
        elif args.search_cmd == 'complete-targets':
            # Find optimal targets (primer and guide set combinations),
            # and write them to a file
            if args.primer_gc_content_bounds is None:
                primer_gc_content_bounds = None
            else:
                primer_gc_content_bounds = tuple(args.primer_gc_content_bounds)
            ps = primer_search.PrimerSearcher(aln, args.primer_length,
                                              args.primer_mismatches,
                                              primer_cover_frac,
                                              args.missing_thres,
                                              seq_groups=seq_groups,
                                              primer_gc_content_bounds=primer_gc_content_bounds)

            if args.obj == 'minimize-guides':
                obj_type = 'min'
            elif args.obj == 'maximize-activity':
                obj_type = 'max'
            ts = target_search.TargetSearcher(ps, gs,
                obj_type=obj_type,
                max_primers_at_site=args.max_primers_at_site,
                max_target_length=args.max_target_length,
                obj_weights=args.obj_fn_weights,
                only_account_for_amplified_seqs=args.only_account_for_amplified_seqs,
                halt_early=args.halt_search_early)
            ts.find_and_write_targets(args.out_tsv[i],
                best_n=args.best_n_targets)
        else:
            raise Exception("Unknown search subcommand '%s'" % args.search_cmd)

        # i should no longer be masked from queries
        if aq is not None:
            aq.unmask_all_aln()


def run(args):
    logger = logging.getLogger(__name__)

    # Set random seed for entire program
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_obj_args(args)

    logger.info("Running design.py with arguments: %s", args)

    # Set NCBI API key
    if args.input_type in ['auto-from-file', 'auto-from-args']:
        if args.ncbi_api_key:
            ncbi_neighbors.ncbi_api_key = args.ncbi_api_key

    if args.input_type in ['auto-from-file', 'auto-from-args']:
        if args.input_type == 'auto-from-file':
            if not os.path.isdir(args.out_tsv_dir):
                raise Exception(("Output directory '%s' does not exist") %
                    args.out_tsv_dir)

        # Prepare input alignments, stored in temp fasta files
        in_fasta, taxid_for_fasta, years_tsv, aln_tmp_dirs, out_tsv, design_for, specific_against_metadata_accs = prepare_alignments(args)
        args.in_fasta = in_fasta
        args.taxid_for_fasta = taxid_for_fasta
        args.out_tsv = out_tsv
        args.design_for = design_for
        args.specific_against_metadata_accs = specific_against_metadata_accs

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
        args.design_for = None
        args.taxid_for_fasta = list(range(len(args.in_fasta)))
        args.specific_against_metadata_accs = [[] for _ in range(len(args.in_fasta))]
    else:
        raise Exception("Unknown input type subcommand '%s'" % args.input_type)

    design_for_id(args)

    # Close temporary files storing alignments
    if args.input_type in ['auto-from-file', 'auto-from-args']:
        for td in aln_tmp_dirs:
            td.cleanup()
        if years_tsv is not None:
            years_tsv.close()


def argv_to_args(argv):
    parser = argparse.ArgumentParser()

    ###########################################################################
    # OPTIONS AVAILABLE ACROSS ALL SUBCOMMANDS
    ###########################################################################
    base_subparser = argparse.ArgumentParser(add_help=False)

    # Guide length
    base_subparser.add_argument('-gl', '--guide-length', type=int, default=28,
        help="Length of guide to construct")

    # Objective function
    base_subparser.add_argument('--obj',
        choices=['maximize-activity', 'minimize-guides'],
        default='minimize-guides',
        help=(("Objective function to solve. 'maximize-activity' maximizes "
               "the expected activity of the guide set of the target genomes "
               "subject to soft and hard constraints on the size of the guide "
               "set. 'minimize-guides' minimizes the number of guides in the "
               "guide set subject to coverage constraints across the target "
               "genomes.")))

    ##########
    # Parameters for minimization objective

    # Number of guide mismatches
    base_subparser.add_argument('-gm', '--guide-mismatches', type=int,
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
        type=check_cover_frac,
        help=("The fraction of all sequences that must be covered "
              "by the guides."))
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
    ##########

    ##########
    # Parameters for maximization objective

    # Soft guide constraint
    base_subparser.add_argument('-sgc', '--soft-guide-constraint', type=int,
        help=("Soft constraint on the number of guides. There is no "
              "penalty for a number of guides <= SOFT_GUIDE_CONSTRAINT, "
              "and having a number of guides beyond this is penalized. "
              "See --penalty-strength. This value must be <= "
              "HARD_GUIDE_CONSTRAINT."))
    # Hard guide constraint
    base_subparser.add_argument('-hgc', '--hard-guide-constraint', type=int,
        help=("Hard constraint on the number of guides. The number of "
              "guides designed for a target will be <= "
              "HARD_GUIDE_CONSTRAINT."))
    # Penalty strength
    base_subparser.add_argument('--penalty-strength', type=float,
        help=("Importance of the penalty when the number of guides "
              "exceeds the soft guide constraint. Namely, for a guide "
              "set G, if the penalty strength is L and the soft "
              "guide constraint is h, then the penalty in the objective "
              "function is L*max(0, |G|-h). Must be >= 0. The value "
              "depends on the output of activity model and reflects a "
              "tolerance for more guides; for the default activity model "
              "reasonable values are in the range [0.1, 0.5]."))
    # Algorithm for solving
    base_subparser.add_argument('--maximization-algorithm',
        choices=['greedy', 'random-greedy'],
        help=("Algorithm to use for solving submodular maximization "
              "problem. 'greedy' is the canonical deterministic greedy "
              "algorithm (Nemhauser 1978) for constrained monotone submodular "
              "maximization, which may perform well in practice but has "
              "poor theoretical guarantees here because the function is "
              "not monotone (unless --penalty-strength is 0). 'random-"
              "greedy' is the randomized greedy algorithm (Buchbinder "
              "2014) for constrained non-monotone submodular maximization "
              "that has good worst-case theoretical guarantees."))
    ##########

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
        type=int, default=4,
        help=("Allow for this number of mismatches when determining whether "
              "a guide 'hits' a sequence in a group/taxon other than the "
              "for which it is being designed; higher values correspond to more "
              "specificity."))
    base_subparser.add_argument('--id-frac', dest="diff_id_frac",
        type=float, default=0.01,
        help=("Decide that a guide 'hits' a group/taxon if it 'hits' a "
              "fraction of sequences in that group/taxon that exceeds this "
              "value; lower values correspond to more specificity."))
    base_subparser.add_argument('--id-method', dest="diff_id_method",
        choices=["lshnn", "shard"], default="shard",
        help=("Choice of method to query for specificity. 'lshnn' for "
              "LSH near-neighbor approach. 'shard' for approach that "
              "shards k-mers across small tries."))
    base_subparser.add_argument('--specific-against-fastas', nargs='+',
        default=[],
        help=("Path to one or more FASTA files giving sequences, such that "
              "guides are designed to be specific against (i.e., not hit) "
              "these sequences, according to --id-m and --id-frac. This "
              "is equivalent to specifying the FASTAs in the main input "
              "(as positional inputs), except that, when provided here, "
              "guides are not designed for them and they do not "
              "need to be aligned."))
    base_subparser.add_argument('--specific-against-taxa',
        help=("Path to TSV file giving giving taxonomies from which to "
              "download all genomes and ensure guides are specific against "
              "(i.e., not hit) these. The TSV file has 2 columns: (1) a "
              "taxonomic ID; (2) segment label, or 'None' if unsegmented"))

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
    base_subparser.add_argument('--seed', type=int,
        help=("SEED will set the random seed, guaranteeing the same output "
              "given the same inputs. If SEED is not set to the same value, "
              "output may vary across different runs."))

    # Use a model to predict activity
    base_subparser.add_argument('--predict-activity-model-path',
        nargs=2,
        help=("Paths to directories containing serialized models in "
              "TensorFlow's SavedModel format for predicting guide-target "
              "activity. There are two arguments: (1) classification "
              "model to determine which guides are active; (2) regression "
              "model, which is used to determine which guides (among "
              "active ones) are highly active. The models/ directory "
              "contains example models. If not set, ADAPT does not predict "
              "activities to use during design."))
    base_subparser.add_argument('--predict-activity-thres',
        type=float,
        nargs=2,
        help=("Thresholds to use for decisions on output of predictive "
            "models. There are two arguments: (1) classification threshold "
            "for deciding which guide-target pairs are active (in [0,1], "
            "where higher values have higher precision but less recall); "
            "(2) regression threshold for deciding which guide-target pairs "
            "are highly active (>= 0, where higher values limit the number "
            "determined to be highly active). If not set but --predict-"
            "activity-model-path is set, then ADAPT uses default thresholds "
            "stored with the models."))
    base_subparser.add_argument('--use-simple-binary-activity-prediction',
        action='store_true',
        help=("If set, predict activity using a simple binary prediction "
              "between guide and target according to their distance, with "
              "the threshold determined based on --guide-mismatches. This "
              "is only applicable when OBJ is 'maxmimize-activity'. This "
              "does not use a serialized model for predicting activity, so "
              "--predict-activity-model-path should not be set when this "
              "is set."))

    # Technical options
    base_subparser.add_argument('--do-not-memoize-guide-computations',
        action='store_true',
        help=("If set, do not memoize computations during the search, "
              "including of guides identified at each site and of "
              "specificity queries. This can be helpful for benchmarking "
              "the improvement of memoization, or if there is reason "
              "to believe memoization will slow the search (e.g., "
              "if possible amplicons rarely overlap). Note that activity "
              "predictions are still memoized."))

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
    parser_sw_args.add_argument('--window-step', type=int, default=1,
        help=("Amount by which to increase the window start position for "
              "every iteration"))
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
    parser_ct_args.add_argument('--primer-gc-content-bounds',
            nargs=2, type=float,
        help=("Only use primer sites where all primers are within the "
              "given GC content bounds. This consists of two values L and H, "
              "each fractions in [0,1], such that primer GC content must be "
              "in [L, H]. If not set, there are no bounds."))
    parser_ct_args.add_argument('--max-target-length', type=int,
        help=("Only allow amplicons (incl. primers) to be at most this "
              "number of nucleotides long; if not set, there is no limit"))
    parser_ct_args.add_argument('--obj-fn-weights', type=float, nargs=2,
        help=("Specify custom weights to use in the objective function "
              "for a target. These specify weights for penalties on primers "
              "and amplicons relative to the guide objective. There are "
              "2 weights (A B), where the target objective function "
              "is [(guide objective value) +/- (A*(total number of "
              "primers) + B*log2(amplicon length)]. It is + when "
              "--obj is minimize-guides and - when --obj is "
              "maximize-activity."))
    parser_ct_args.add_argument('--best-n-targets', type=int, default=10,
        help=("Only compute and output up to this number of targets. Note "
              "that runtime will generally be longer for higher values"))
    parser_ct_args.add_argument('--halt-search-early',
        action='store_true',
        help=('If set, stop the target search as soon as BEST_N_TARGETS '
              'have been identified. The targets will meet the given '
              'constraints but may not be optimal over the whole genome. '
              'They will likely be from the beginning of the genome.'))
    parser_ct_args.add_argument('--only-account-for-amplified-seqs',
        action='store_true',
        help=("If set, design guides to cover GUIDE_COVER_FRAC of just "
              "the sequences covered by the primers. This changes the "
              "behavior of -gp/--guide-cover-frac. This is only "
              "applicable when --obj is 'minimize-guides' as it is "
              "not implemented for 'maximize-activity'. In total, "
              ">= (GUIDE_COVER_FRAC * (2 * PRIMER_COVER_FRAC - 1)) "
              "sequences will be covered. Using this may worsen runtime "
              "because the sequences to consider for guide design will "
              "change more often across amplicons and therefore designs "
              "can be less easily memoized."))
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
              "statistics on them. If set to \"s3://BUCKET/PATH\", it "
              "will save to the S3 bucket if boto3 and botocore are "
              "installed and access key information exists via "
              "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or via AWS CLI. "
              "If not set, this does not memoize this information."))
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
    input_auto_common_subparser.add_argument('--use-fasta',
        help=("If set, use sequences in fasta instead of fetching neighbors "
              "for the given taxonomic ID(s). This provides a path to a TSV "
              "file with 3 columns: (1) a taxonomic ID; (2) segment label, "
              "or 'None' if unsegmented; (3) path to FASTA."))
    input_auto_common_subparser.add_argument('--only-design-for',
        help=("If set, only design for given taxonomies. This provides a "
              "path to a TSV file with 2 columns: (1) a taxonomic ID; (2) "
              "segment label, or 'None' if unsegmented"))
    input_auto_common_subparser.add_argument('--taxa-to-ignore-for-specificity',
        help=("If set, specify which taxa should be ignored when "
              "enforcing specificity while designing for other taxa. "
              "This provides a path to a TSV file with 2 columns: "
              "(1) a taxonomic ID A; (2) a taxonomic ID B such that "
              "B should be ignored when determining specificity for A. "
              "When designing for A, this masks taxonomy B from all "
              "specificity queries. This is useful, e.g., if B is a "
              "subset of A."))
    input_auto_common_subparser.add_argument('--ncbi-api-key',
        help=("API key to use for NCBI e-utils. Using this increases the "
              "limit on requests/second and may prevent an IP address "
              "from being blocked due to too many requests"))
    input_auto_common_subparser.add_argument('--aws-access-key-id',
        help=("User Account Access Key for AWS. This is only necessary "
            "if using S3 for memoization via PREP_MEMOIZE_DIR and AWS CLI "
            "is not installed and configured."))
    input_auto_common_subparser.add_argument('--aws-secret-access-key',
        help=("User Account Secret Access Key for AWS. This is only "
            "necessary if using S3 for memoization via PREP_MEMOIZE_DIR "
            "and AWS CLI is not installed and configured."))

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
    input_autoargs_subparser.add_argument('out_tsv',
        help=("Path to output TSVs, with one per cluster; output TSVs are "
              "OUT_TSV.{cluster-number}"))
    input_autoargs_subparser.add_argument('--ref-accs', nargs='+',
        help=("Accession(s) of reference sequence(s) to use for curation (comma-"
              "separated). If not set, ADAPT will automatically get accessions "
              "for reference sequences from NCBI based on the taxonomic ID"))
    input_autoargs_subparser.add_argument('--metadata-filter', nargs='+',
        help=("Only include accessions of specified taxonomic ID that match this metadata "
            "in the design. Metadata options are year, taxid, and country. Format as "
            "'metadata=value' or 'metadata!=value'. Separate multiple values with commas "
            "and different metadata filters with spaces (e.g. '--metadata-filter "
            "year!=2020,2019 taxid=11060')"))
    input_autoargs_subparser.add_argument('--specific-against-metadata-filter', nargs='+',
        help=("Only include accessions of the specified taxonomic ID that do not match this "
            "metadata in the design, and be specific against any accession that does match "
            "this metadata. Metadata options are year, taxid, and country. Format as "
            "'metadata=value' or 'metadata!=value'. Separate multiple values with commas "
            "and different metadata filters with spaces (e.g. "
            "'--specific-against-metadata-filter year!=2020,2019 taxid=11060')"))
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

    args = parser.parse_args(argv[1:])

    # Handle missing positional arguments by printing a help message
    if len(argv) == 1:
        # No arguments
        parser.print_help()
        sys.exit(1)
    if len(argv) == 2:
        # Only one position argument (missing input type)
        if args.search_cmd == 'sliding-window':
            parser_sw.print_help()
        if args.search_cmd == 'complete-targets':
            parser_ct.print_help()
        sys.exit(1)

    # Setup the logger
    log.configure_logging(args.log_level)

    return args


if __name__ == "__main__":
    run(argv_to_args(sys.argv))
