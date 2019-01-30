"""Functions for preparing an alignment for input.

This includes downloading, curating, and aligning sequences.
"""

from collections import OrderedDict
import logging
import os
import random

from dxguidedesign.prepare import align
from dxguidedesign.prepare import cluster
from dxguidedesign.prepare import ncbi_neighbors
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def prepare_for(taxid, segment, ref_acc, out,
        aln_memoizer=None, aln_stat_memoizer=None,
        limit_seqs=None, filter_warn=0.25, min_seq_len=200,
        min_cluster_size=2, prep_influenza=False):
    """Prepare an alignment for a taxonomy.

    This does the following:
        1) Downloads a list of neighbors.
        2) Downloads sequences of those neighbors.
        3) Curates the sequences.
        4) Aligns the sequences.

    Args:
        taxid: taxonomic ID from NCBI, for which to download
            sequences
        segment: only use sequences of this segment (ignored if set
            to '' or None, e.g., if taxid is unsegmented)
        ref_acc: accession of reference sequence to use for curation
        out: path to direcotry in which write FASTA files of aligned
            sequences (one per cluster)
        aln_memoizer: AlignmentMemoizer to use for memoizing alignments;
            or None to not memoize
        aln_stat_memoizer: AlignmentStatMemoizer to use for memoization
            of alignment statistic values; or None to not memoize
        limit_seqs: fetch all accessions, and then randomly select
            LIMIT_SEQS of them without replacement and only design from
            these; if None (default), do not limit the input
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

    Returns:
        number of clusters
    """
    logger.info(("Preparing an alignment for tax %d (segment: %s) with "
        "reference %s") % (taxid, segment, ref_acc))

    # Download neighbors for taxid
    if prep_influenza:
        neighbors = ncbi_neighbors.construct_influenza_genome_neighbors(taxid)
    else:
        neighbors = ncbi_neighbors.construct_neighbors(taxid)

    # Filter neighbors by segment
    if segment != None and segment != '':
        neighbors = [n for n in neighbors if n.segment == segment]

    if len(neighbors) == 0:
        if segment != None and segment != '':
            raise Exception(("No sequences were found for taxid %d and "
                "segment '%s'") % (taxid, segment))
        else:
            raise Exception(("No sequences were found for taxid %d") % taxid)

    if limit_seqs is not None:
        # Randomly sample limit_seqs from neighbors
        if limit_seqs > len(neighbors):
            logger.warning(("limit_seqs for tax %d (segment: %s) is %d and "
                "is greater than the number of sequences available (%d); not "
                "subsampling, and instead using all available sequences") %
                (taxid, segment, limit_seqs, len(neighbors)))
        else:
            neighbors = random.sample(neighbors, limit_seqs)

    # Fetch FASTAs for the neighbors; also do so for ref_acc if it
    # is not included
    # Keep track of whether it was added, so that it can be removed
    # later on
    acc_to_fetch = [n.acc for n in neighbors]
    if ref_acc not in acc_to_fetch:
        acc_to_fetch += [ref_acc]
        added_ref_acc_to_fetch = True
    else:
        added_ref_acc_to_fetch = False
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

    seqs_unaligned_curated = align.curate_against_ref(
        seqs_unaligned, ref_acc, asm=aln_stat_memoizer,
        remove_ref_acc=added_ref_acc_to_fetch)

    # An accession can show up multiple times (as separate neighbors)
    # for different RefSeq entries, but seqs_unaligned only stores
    # unique accessions; when determining how many were filtered out,
    # compared against seqs_unaligned rather than neighbors
    frac_filtered = 1.0 - float(len(seqs_unaligned_curated)) / len(seqs_unaligned)
    if frac_filtered >= filter_warn:
        logger.warning(("A fraction %f of sequences were filtered out "
            "during curation for tax %d (segment: %s) using reference %s") %
            (frac_filtered, taxid, segment, ref_acc))

    # Produce clusters of unaligned sequences
    logger.info(("Clustering %d sequences"), len(seqs_unaligned_curated))
    clusters = cluster.cluster_with_minhash_signatures(
        seqs_unaligned_curated)

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

    return len(clusters)

