"""Functions for performing an alignment of sequences, with curation.
"""

from collections import OrderedDict
import hashlib
import os
import re
import subprocess
import tempfile

from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'


def read_unaligned_seqs(tmp):
    """Parse temp fasta file of sequences.

    Args:
        tmp: tempfile, which has a name attribute
            (tempfile.NamedTemporaryFile object)

    Returns:
        dict mapping accessions (without version) to sequences
        (as str)
    """
    # TODO: Use accession.version instead of just accession
    # Each sequence header is in the format '[accession].[version] [name, etc.]'
    # Parse them to extract the accession
    acc_p = re.compile("(.+?)(?:\.| )")

    seqs = seq_io.read_fasta(tmp.name)
    seqs_by_acc = OrderedDict()
    for name, seq in seqs.items():
        acc = acc_p.match(name).group(1)
        seqs_by_acc[acc] = seq
    return seqs_by_acc


class AlignmentMemoizer:
    """Memoizer for alignments.

    This stores an alignment using a hash of the accessions
    in the alignments.

    This stores alignments as fasta files named by the hash.
    """

    def __init__(self, path):
        """
            Args:
                path: path to directory in which to read and store
                    memoized alignments
        """
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _hash_accs_filepath(self, accs):
        """Generate a path to use as the filename for an alignment.

        Args:
            accs: collection of accessions in an alignment

        Returns:
            path to file that should store accs
        """
        h = hashlib.md5(str(sorted(set(accs))).encode('utf-8')).hexdigest()
        return os.path.join(self.path, h)

    def get(self, accs):
        """Get memoized alignment.

        Args:
            accs: collection of accessions in an alignment

        Returns:
            OrderedDict mapping accessions to sequences; or None
            if accs is not memoized
        """
        p = self._hash_accs_filepath(accs)

        if os.path.exists(p):
            # Read the fasta, and verify that the accessions in it
            # match accs (i.e., that there is not a collision)
            seqs = seq_io.read_fasta(p)
            if set(accs) == set(seqs.keys()):
                return seqs
            else:
                return None
        else:
            return None

    def save(self, seqs):
        """Memoize alignment by saving it to a file.

        Args:
            seqs: dict mapping accessions to sequences
        """
        p = self._hash_accs_filepath(seqs.keys())
        seq_io.write_fasta(seqs, p)
        

class AlignmentStatMemoizer:
    """Memoizer for statistics on alignments.

    This stores, for each alignment, the tuple (aln_identity, aln_identity_ccg).
    """

    def __init__(self, path):
        """
            Args:
                path: path to file at which to read and store memoized
                    stats
        """
        self.path = path
        self.memoized = {}

        if os.path.isfile(self.path):
            # Read stats from the file
            with open(self.path) as f:
                for line in f:
                    ls = line.split('\t')
                    accs = tuple(ls[0].split(','))
                    stats = float(ls[1]), float(ls[2])
                    self.memoized[accs] = stats

    def get(self, accs):
        """Get memoized stats for alignment.

        Args:
            accs: collection of accessions in an alignment

        Returns:
            (aln_identity, aln_identity_ccg) for alignment of
            accs; or None if accs is not memoized
        """
        accs_sorted = tuple(sorted(accs))
        if accs_sorted in self.memoized:
            return self.memoized[accs_sorted]
        else:
            return None

    def add(self, accs, stats):
        """Memoize statistics for an alignment.

        Args:
            accs: collection of accessions in an alignment
            stats: tuple (aln_identity, aln_identity_ccg)
        """
        assert len(stats) == 2
        accs_sorted = tuple(sorted(accs))
        self.memoized[accs_sorted] = stats

    def save(self):
        """Save memoized values to file."""
        with open(self.path + '.tmp', 'w') as f:
            for accs, stats in self.memoized.items():
                accs_str = ','.join(accs)
                f.write(('\t'.join([accs_str, str(stats[0]), str(stats[1])]) +
                    '\n'))
        os.rename(self.path + '.tmp', self.path)


_mafft_exec = None
def set_mafft_exec(mafft_path):
    """Set path to the mafft executable.

    Args:
        mafft_path: path to mafft executable
    """
    global _mafft_exec
    # Verify the path exists and is an executable
    if not os.path.isfile(mafft_path) or not os.access(mafft_path, os.X_OK):
        raise Exception(("Path to mafft (%s) does not exist or is not "
            "executable") % mafft_path)
    _mafft_exec = mafft_path


def align(seqs, am=None):
    """Align sequences using mafft.

    Args:
        seqs: dict mapping sequence header to sequences
        am: AlignmentMemoizer object to use for memoizing alignments;
            or None to not memoize

    Returns:
        dict mapping sequence header to sequences, where all the sequences
        are aligned
    """
    if am is not None:
        seqs_aligned = am.get(seqs.keys())
        if seqs_aligned is not None:
            return seqs_aligned

    global _mafft_exec
    # Check that mafft executable has been set
    if _mafft_exec is None:
        raise Exception(("Path to mafft executable must be set using "
            "_set_mafft_exec()"))

    # Write a fasta of these sequences to a named temp file, and
    # create one to write the output fasta to
    in_fasta = tempfile.NamedTemporaryFile()
    seq_io.write_fasta(seqs, in_fasta.name)
    out_fasta = tempfile.NamedTemporaryFile()

    # Setup arguments to mafft
    params = ['--maxiterate', '1000', '--preservecase']
    if len(seqs) < 100:
        # Accuracy oriented
        params += ['--localpair']
    else:
        # Speed oriented
        params += ['--retree', '2']

    # Call mafft
    cmd = [_mafft_exec] + params + [in_fasta.name]
    subprocess.call(cmd, stdout=out_fasta, stderr=subprocess.DEVNULL)

    # Read the output fasta into a dict
    seqs_aligned = seq_io.read_fasta(out_fasta.name)

    # Close temp files
    in_fasta.close()
    out_fasta.close()

    if am is not None:
        # Memoize the alignment
        am.save(seqs_aligned)

    return seqs_aligned


def _aln_identity(a, b):
    """Compute identity of alignment between two sequences.

    Args:
        a, b: two aligned sequences

    Returns:
        float representing the fraction of the alignment that is
        identical between a and b
    """
    assert len(a) == len(b)
    identical_count = sum(1 for i in range(len(a)) if a[i] == b[i])
    return identical_count / float(len(a))


def _collapse_consecutive_gaps(a, b):
    """Collapse consecutive gaps in an alignment between two sequences.

    For example, the alignment
        ATC----GA
        ATCATCGGA
    would become
        ATC-GA
        ATCAGA

    Args:
        a, b: two aligned sequences

    Returns:
        tuple (a', b') where a' and b' represents an alignment of a
        and b, with gaps collapsed
    """
    assert len(a) == len(b)
    a_ccg, b_ccg = a[0], b[0]
    for i in range(len(a)):
        if a[i-1] == '-':
            if a[i] == '-':
                # Skip this position; we are already in a gap
                continue
            else:
                # We are out of the gap for a; be sure to include
                # this position
                a_ccg += a[i]
                b_ccg += b[i]
        elif b[i-1] == '-':
            if b[i] == '-':
                # Skip this position; we are already in a gap
                continue
            else:
                # We are out of the gap for b; be sure to include
                # this position
                a_ccg += a[i]
                b_ccg += b[i]
        else:
            a_ccg += a[i]
            b_ccg += b[i]
    return (a_ccg, b_ccg)


def curate_against_ref(seqs, ref_acc, asm=None,
        aln_identity_thres=0.5, aln_identity_ccg_thres=0.6):
    """Curate sequences by aligning pairwise with a reference sequence.

    This can make use of an object of AlignmentStatMemoizer to
    memoize values, since the pairwise alignment can be slow.

    Args:
        seqs: dict mapping sequence accession to sequences
        ref_acc: accession of reference sequence to use for curation; must
            be a key in seqs
        asm: AlignmentStatMemoizer to use for memoization of values; or
            None to not memoize
        aln_identity_thres: filter out any sequences with less than
            this fraction identity to ref_acc (taken over the whole alignment)
        aln_identity_ccg_thres: filter out any sequences with less than
            this fraction identity to ref_acc, after collapsing
            consecutive gaps to a single gap; this should be at least
            aln_identity_thres

    Returns:
        dict mapping sequence accession to sequences, filtered by
        ones that can align reasonably well with ref_acc
    """
    if ref_acc not in seqs:
        raise Exception("ref_acc must be in seqs")

    seqs_filtered = OrderedDict()
    for acc, seq in seqs.items():
        if acc == ref_acc:
            # ref_acc will align well with ref_acc, so include it automatically
            seqs_filtered[acc] = seq
            continue

        stats = asm.get((ref_acc, acc)) if asm is not None else None
        if stats is not None:
            aln_identity, aln_identity_ccg = stats
        else:
            # Align ref_acc with acc
            to_align = {ref_acc: seqs[ref_acc], acc: seq}
            aligned = align(to_align)
            ref_acc_aln = aligned[ref_acc]
            acc_aln = aligned[acc]
            assert len(ref_acc_aln) == len(acc_aln)

            # Compute the identity of the alignment
            aln_identity = _aln_identity(ref_acc_aln, acc_aln)

            # Compute the identity of the alignment, after collapsing
            # consecutive gaps (ccg) to a single gap
            ref_acc_aln_ccg, acc_aln_ccg = _collapse_consecutive_gaps(
                ref_acc_aln, acc_aln)
            aln_identity_ccg = _aln_identity(ref_acc_aln_ccg, acc_aln_ccg)

            if asm is not None:
                # Memoize these statistics, so the alignment does not need
                # to be performed again
                stats = (aln_identity, aln_identity_ccg)
                asm.add((ref_acc, acc), stats)

        if (aln_identity >= aln_identity_thres and
                aln_identity_ccg >= aln_identity_ccg_thres):
            seqs_filtered[acc] = seq

    if asm is not None:
        # Save the new memoized values
        asm.save()

    return seqs_filtered
