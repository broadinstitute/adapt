"""Functions for preparing an alignment for input.

This includes downloading, curating, and aligning sequences.
"""

from dxguidedesign.prepare import align
from dxguidedesign.prepare import ncbi_neighbors
from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'


def prepare_for(taxid, segment, ref_acc, out,
        aln_memoizer=None, aln_stat_memoizer=None):
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
    """
    # Download neighbors for taxid
    neighbors = ncbi_neighbors.construct_neighbors(taxid)

    # Filter neighbors by segment
    if segment != None and segment != '':
        neighbors = [n for n in neighbors if n.segment == segment]

    # Fetch FASTAs for the neighbors; also do so for ref_acc if it
    # is not included
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
        seqs_unaligned, ref_acc, asm=aln_stat_memoizer)

    # If ref_acc was added to seqs_unaligned (because it needed
    # to be fetched), remove it
    if added_ref_acc_to_fetch:
        del seqs_unaligned_curated[ref_acc]

    # TODO: warn if too many (>25%) are filtered out
    # Note that len(accessions) may be >> len(seqs_unaligned) because
    # an accession can show up multiple times (as separate neighbors)
    # for different RefSeq entries, but seqs_unaligned only stores
    # unique accessions. So compare against len(seq_unaligned)

    # Align the curated sequences
    seqs_aligned = align.align(seqs_unaligned_curated, am=aln_memoizer)

    seq_io.write_fasta(seqs_aligned, out)

