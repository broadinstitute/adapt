"""Functions for preparing an alignment for input.

This includes downloading, curating, and aligning sequences.
"""

from collections import Counter
from collections import OrderedDict
import logging
import os
import random

from adapt.prepare import align
from adapt.prepare import cluster
from adapt.prepare import ncbi_neighbors
from adapt.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def prepare_for(taxid, segment, ref_accs, out,
        aln_memoizer=None, aln_stat_memoizer=None,
        sample_seqs=None, filter_warn=0.25, min_seq_len=200,
        min_cluster_size=2, prep_influenza=False, years_tsv=None,
        cluster_threshold=0.1, accessions_to_use=None):
    """Prepare an alignment for a taxonomy.

    This does the following:
        1) Downloads a list of neighbors.
        2) Downloads sequences of those neighbors.
        3) Curates the sequences.
        4) Clusters the unaligned sequences.
        5) Aligns the sequences, with one alignment per cluster.

    Args:
        taxid: taxonomic ID from NCBI, for which to download
            sequences
        segment: only use sequences of this segment (ignored if set
            to '' or None, e.g., if taxid is unsegmented)
        ref_accs: list of accessions of reference sequences to use for curation
        out: path to direcotry in which write FASTA files of aligned
            sequences (one per cluster)
        aln_memoizer: AlignmentMemoizer to use for memoizing alignments;
            or None to not memoize
        aln_stat_memoizer: AlignmentStatMemoizer to use for memoization
            of alignment statistic values; or None to not memoize
        sample_seqs: fetch all accessions, and then randomly select
            SAMPLE_SEQS of them with replacement and design from
            these; if None (default), do not sample the input
        filter_warn: raise a warning if the fraction of sequences that
            are filtered out during curation is greater than or equal to
            this float
        min_seq_len: toss any sequences whose length is less than
            this. The value should be at least as largest as the
            number of hash values used to produce a sequence signature
            for clustering (in cluster.cluster_with_minhash_signatures());
            otherwise, that function will be unable to form a signature
        min_cluster_size: toss all sequences from any clusters with
            less than this number of sequences; in other words, ignore
            clusters less than this size
        prep_influenza: if True, assume taxid represents an Influenza
            A or B virus taxonomy, and fetch sequences using NCBI's
            Influenza database
        years_tsv: if set, a path to a TSV file to which this will write
            a year (column 2) for each sequence written to out (column 1);
            this is only done for influenza sequences (if prep_influenza
            is True)
        cluster_threshold: maximum inter-cluster distance to merge clusters, in
            average nucleotide dissimilarity (1-ANI, where ANI is
            average nucleotide identity); higher results in fewer
            clusters
        accessions_to_use: if set, a collection of accessions to use instead
            of fetching neighbors for taxid

    Returns:
        number of clusters
    """
    logger.info(("Preparing an alignment for tax %d (segment: %s) with "
        "references %s") % (taxid, segment, ref_accs))

    if accessions_to_use is not None:
        neighbors = [ncbi_neighbors.Neighbor(acc, None, None, None, None,
            segment) for acc in accessions_to_use]
    else:
        # Download neighbors for taxid
        if prep_influenza:
            neighbors = ncbi_neighbors.construct_influenza_genome_neighbors(taxid)
        else:
            neighbors = ncbi_neighbors.construct_neighbors(taxid)

        # Filter neighbors by segment
        if segment != None and segment != '':
            neighbors = [n for n in neighbors if n.segment == segment]

    num_unique_acc = len(set(n.acc for n in neighbors))
    logger.info(("There are %d neighbors (%d with unique accessions)"),
            len(neighbors), num_unique_acc)

    if years_tsv is not None:
        # Fetch metadata (including year), add it to neighbors, and
        # filter out ones without a known year
        neighbors = ncbi_neighbors.add_metadata_to_neighbors_and_filter(neighbors)

    if len(neighbors) == 0:
        if segment != None and segment != '':
            raise Exception(("No sequences were found for taxid %d and "
                "segment '%s'") % (taxid, segment))
        else:
            raise Exception(("No sequences were found for taxid %d") % taxid)

    if sample_seqs is not None:
        # Randomly sample sample_seqs from neighbors
        if sample_seqs > len(neighbors):
            logger.warning(("sample_seqs for tax %d (segment: %s) is %d and "
                "is greater than the number of sequences available (%d); "
                "still subsampling with replacement, but this might be "
                "unintended") %
                (taxid, segment, sample_seqs, len(neighbors)))
        # Sample accessions, rather than neighbors, because an accession can
        # show up multiple times in neighbors due to having multiple RefSeq
        # entries
        acc_to_sample = list(set([n.acc for n in neighbors]))
        acc_to_fetch = random.choices(acc_to_sample, k=sample_seqs)
        neighbors = [n for n in neighbors if n.acc in acc_to_fetch]
        # Because this is sampling with replacement, an accession may
        # show up multiple times in the sampling; to keep this multiplicity
        # going forward, track its count
        acc_count = Counter(acc_to_fetch)
        logger.info(("After sampling with replacement to %d sequences, there "
            "are %d unique accessions; after curation, this will repeat ones "
            "in the input that were sampled more than once (the final number, "
            "even after adding multiplicity, may be less than %d if accessions "
            "are filtered out during curation)"),
            sample_seqs, len(set(acc_to_fetch)), sample_seqs)
    else:
        # Only fetch each accession once
        acc_to_fetch = list(set([n.acc for n in neighbors]))
        acc_count = Counter(acc_to_fetch)   # 1 for each accession

    # Fetch FASTAs for the neighbors; also do so for ref_accs if ones
    # are not included
    # Keep track of whether they were added, so that they can be removed
    # later on
    added_ref_accs_to_fetch = []
    for ref_acc in ref_accs:
        if ref_acc not in acc_to_fetch:
            acc_to_fetch += [ref_acc]
            added_ref_accs_to_fetch += [ref_acc]
    seqs_unaligned_fp = ncbi_neighbors.fetch_fastas(acc_to_fetch)
    seqs_unaligned = align.read_unaligned_seqs(seqs_unaligned_fp)
    seqs_unaligned_fp.close()

    # Toss sequences that are too short
    seqs_too_short = set()
    for name, seq in seqs_unaligned.items():
        if len(seq) < min_seq_len:
            seqs_too_short.add(name)
    if len(seqs_too_short) > 0:
        logger.warning(("Ignoring the following sequences, whose lengths "
            "are <%d nt long: %s"), min_seq_len,
            str(sorted(list(seqs_too_short))))
    for name in seqs_too_short:
        del seqs_unaligned[name]

    # Curate against ref_accs
    seqs_unaligned_curated = align.curate_against_ref(
        seqs_unaligned, ref_accs, asm=aln_stat_memoizer,
        remove_ref_accs=added_ref_accs_to_fetch)

    # An accession can show up multiple times if there was sampling with
    # replacement, but the dicts above (seqs_unaligned and
    # seqs_unaligned_curate) are key'd on accession; replicate the
    # multiplicity here but renaming an accession that should appear multiple
    # times from [accession.version] to [accession.version]-{1,2,...}
    seqs_unaligned_curated_with_multiplicity = OrderedDict()
    for accver, seq in seqs_unaligned_curated.items():
        acc = accver.split('.')[0]
        assert acc_count[acc] > 0
        if acc_count[acc] == 1:
            seqs_unaligned_curated_with_multiplicity[accver] = seq
        else:
            logger.debug(("Adding multiplicity %d to accession %s") %
                    (acc_count[acc], acc))
            for i in range(1, acc_count[acc] + 1):
                new_name = accver + '-' + str(i)
                seqs_unaligned_curated_with_multiplicity[new_name] = seq
    seqs_unaligned_curated = seqs_unaligned_curated_with_multiplicity

    if sample_seqs is not None:
        logger.info(("After adding multiplicity, there are %d sequences"),
            len(seqs_unaligned_curated))

    # When determining how many were filtered out, we care about acc_to_fetch
    # (which has multiplicity in case of sampling with replacement)
    frac_filtered = 1.0 - float(len(seqs_unaligned_curated)) / len(acc_to_fetch)
    if frac_filtered >= filter_warn:
        logger.warning(("A fraction %f of sequences were filtered out "
            "during curation for tax %d (segment: %s) using references %s") %
            (frac_filtered, taxid, segment, ref_accs))

    # Check if there are no sequences left; if that's the case, don't
    # proceed
    if len(seqs_unaligned_curated) == 0:
        raise Exception("No sequences remain after curation")

    # Produce clusters of unaligned sequences
    logger.info(("Clustering %d sequences"), len(seqs_unaligned_curated))
    clusters = cluster.cluster_with_minhash_signatures(
        seqs_unaligned_curated, threshold=cluster_threshold)

    # Throw away clusters (and the sequences in them) that are too small;
    # but only throw away clusters if at least 1 will remain
    # Note that clusters is sorted in descending order of cluster size
    seqs_from_small_clusters = set()
    cluster_to_remove = None
    for cluster_idx, seqs_in_cluster in enumerate(clusters):
        if len(seqs_in_cluster) < min_cluster_size:
            seqs_from_small_clusters.update(seqs_in_cluster)
            if cluster_to_remove is None:
                cluster_to_remove = cluster_idx
    if cluster_to_remove is not None and cluster_to_remove > 0:
        # There is a cluster to remove, and it is not the first (i.e.,
        # one will remain)
        logger.warning(("Removing clusters that are too small (<%d), and "
            "ignoring all the sequences they contain: %s"), min_cluster_size,
            str(sorted(list(seqs_from_small_clusters))))
        clusters = clusters[:cluster_to_remove]

    # Align the curated sequences, with one alignment per cluster
    for cluster_idx, seqs_in_cluster in enumerate(clusters):
        logger.info(("Aligning sequences in cluster %d (of %d), with %d "
            "sequences"), cluster_idx + 1, len(clusters), len(seqs_in_cluster))

        # Produce a dict of the sequences in cluster_idx
        seqs_unaligned_curated_in_cluster = OrderedDict()
        for name, seq in seqs_unaligned_curated.items():
            if name in seqs_in_cluster:
                seqs_unaligned_curated_in_cluster[name] = seq

        # Align the sequences in this cluster
        seqs_aligned = align.align(seqs_unaligned_curated_in_cluster,
            am=aln_memoizer)

        # Write a fasta file of aligned sequences
        fasta_file = os.path.join(out, str(cluster_idx) + '.fasta')
        seq_io.write_fasta(seqs_aligned, fasta_file)

    # Write the years for each sequence, if requested
    if years_tsv:
        year_for_acc = {neighbor.acc: neighbor.metadata['year'] for
                neighbor in neighbors}
        all_seq_names = set().union(*clusters)
        with open(years_tsv, 'w') as fw:
            for name in all_seq_names:
                # name is [accession].[version] (or [accession].[version]-[n]
                # in case of multiplicity); extract the accession
                acc = name.split('.')[0]
                fw.write('\t'.join([name, str(year_for_acc[acc])]) + '\n')

    return len(clusters)

