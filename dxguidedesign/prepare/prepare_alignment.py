"""Functions for preparing an alignment for input.

This includes downloading, curating, and aligning sequences.
"""

import logging
import random

from dxguidedesign.prepare import align
from dxguidedesign.prepare import ncbi_neighbors
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def prepare_for(taxid, segment, ref_acc, out,
        aln_memoizer=None, aln_stat_memoizer=None,
        limit_seqs=None, filter_warn=0.25):
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
        out: path to write FASTA file of aligned sequences
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
    """
    logger.info(("Preparing an alignment for tax %d (segment: %s) with "
        "reference %s") % (taxid, segment, ref_acc))

    # Download neighbors for taxid
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
            raise Exception(("limit_seqs for tax %id (segment: %s) is %d and "
                "is greater than number of sequences available (%d); it must "
                "be at most %d") % (taxid, segment, limit_seqs, len(neighbors),
                len(neighbors)))
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

    # Align the curated sequences
    seqs_aligned = align.align(seqs_unaligned_curated, am=aln_memoizer)

    seq_io.write_fasta(seqs_aligned, out)

