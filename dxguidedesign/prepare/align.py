"""Functions for performing an alignment of sequences, with curation.
"""

from collections import OrderedDict
import hashlib
import logging
import os
import re
import subprocess
import tempfile

from dxguidedesign.utils import seq_io

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def read_unaligned_seqs(tmp):
    """Parse temp fasta file of sequences.

    Args:
        tmp: tempfile, which has a name attribute
            (tempfile.NamedTemporaryFile object)

    Returns:
        dict mapping accession.version to sequences (as str)
    """
    logger.debug(("Reading and parsing unaligned sequences from "
        "tmp file at %s") % tmp.name)

    # Each sequence header is in the format '[accession].[version] [name, etc.]'
    # Parse them to extract the accession.version (everything before the
    # first space)
    accver_p = re.compile('([^\s]+)')

    seqs = seq_io.read_fasta(tmp.name)
    seqs_by_accver = OrderedDict()
    for name, seq in seqs.items():
        accver = accver_p.match(name).group(1)
        seqs_by_accver[accver] = seq
    return seqs_by_accver


class AlignmentMemoizer:
    """Memoizer for alignments.

    This stores an alignment using a hash of the accession.version
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

    def _hash_accvers_filepath(self, accvers):
        """Generate a path to use as the filename for an alignment.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            path to file that should store accs
        """
        h = hashlib.md5(str(sorted(set(accvers))).encode('utf-8')).hexdigest()
        return os.path.join(self.path, h)

    def get(self, accvers):
        """Get memoized alignment.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            OrderedDict mapping accession.version to sequences; or None
            if accvers is not memoized
        """
        p = self._hash_accvers_filepath(accvers)

        if os.path.exists(p):
            # Read the fasta, and verify that the accessions in it
            # match accs (i.e., that there is not a collision)
            seqs = seq_io.read_fasta(p)
            if set(accvers) == set(seqs.keys()):
                return seqs
            else:
                return None
        else:
            return None

    def save(self, seqs):
        """Memoize alignment by saving it to a file.

        Args:
            seqs: dict mapping accession.version to sequences
        """
        p = self._hash_accvers_filepath(seqs.keys())
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
                    accvers = tuple(ls[0].split(','))
                    stats = float(ls[1]), float(ls[2])
                    self.memoized[accvers] = stats

    def get(self, accvers):
        """Get memoized stats for alignment.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            (aln_identity, aln_identity_ccg) for alignment of
            accvers; or None if accs is not memoized
        """
        accvers_sorted = tuple(sorted(accvers))
        if accvers_sorted in self.memoized:
            return self.memoized[accvers_sorted]
        else:
            return None

    def add(self, accvers, stats):
        """Memoize statistics for an alignment.

        Args:
            accvers: collection of accessions in an alignment
            stats: tuple (aln_identity, aln_identity_ccg)
        """
        assert len(stats) == 2
        accvers_sorted = tuple(sorted(accvers))
        self.memoized[accvers_sorted] = stats

    def save(self):
        """Save memoized values to file."""
        with open(self.path + '.tmp', 'w') as f:
            for accvers, stats in self.memoized.items():
                accvers_str = ','.join(accvers)
                f.write(('\t'.join([accvers_str, str(stats[0]), str(stats[1])]) +
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
    for i in range(1, len(a)):
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
        aln_identity_thres=0.5, aln_identity_ccg_thres=0.6,
        remove_ref_acc=False):
    """Curate sequences by aligning pairwise with a reference sequence.

    This can make use of an object of AlignmentStatMemoizer to
    memoize values, since the pairwise alignment can be slow.

    Args:
        seqs: dict mapping sequence accession to sequences
        ref_acc: accession of reference sequence to use for curation; either
            itself or ref_acc.[version] for some version must be a key
            in seqs
        asm: AlignmentStatMemoizer to use for memoization of values; or
            None to not memoize
        aln_identity_thres: filter out any sequences with less than
            this fraction identity to ref_acc (taken over the whole alignment)
        aln_identity_ccg_thres: filter out any sequences with less than
            this fraction identity to ref_acc, after collapsing
            consecutive gaps to a single gap; this should be at least
            aln_identity_thres
        remove_ref_acc: do not include ref_acc in the output

    Returns:
        dict mapping sequence accession.version to sequences, filtered by
        ones that can align reasonably well with ref_acc
    """
    logger.debug(("Curating %d sequences against reference %s") %
        (len(seqs), ref_acc))

    # Find ref_acc in seqs
    if ref_acc in seqs:
        ref_acc_key = ref_acc
    else:
        # Look for a key in seqs that is ref_acc.[version] for some version
        ref_acc_key = None
        for k in seqs.keys():
            if k.split('.')[0] == ref_acc:
                ref_acc_key = k
                break
        if ref_acc_key is None:
            raise Exception("ref_acc must be in seqs")

    seqs_filtered = OrderedDict()
    for accver, seq in seqs.items():
        if accver == ref_acc_key:
            # ref_acc_key will align well with ref_acc_key, so include it
            # automatically
            seqs_filtered[accver] = seq
            continue

        stats = asm.get((ref_acc_key, accver)) if asm is not None else None
        if stats is not None:
            aln_identity, aln_identity_ccg = stats
        else:
            # Align ref_acc_key with accver
            to_align = {ref_acc_key: seqs[ref_acc_key], accver: seq}
            aligned = align(to_align)
            ref_acc_aln = aligned[ref_acc_key]
            accver_aln = aligned[accver]
            assert len(ref_acc_aln) == len(accver_aln)

            # Compute the identity of the alignment
            aln_identity = _aln_identity(ref_acc_aln, accver_aln)

            # Compute the identity of the alignment, after collapsing
            # consecutive gaps (ccg) to a single gap
            ref_acc_aln_ccg, accver_aln_ccg = _collapse_consecutive_gaps(
                ref_acc_aln, accver_aln)
            aln_identity_ccg = _aln_identity(ref_acc_aln_ccg, accver_aln_ccg)

            if asm is not None:
                # Memoize these statistics, so the alignment does not need
                # to be performed again
                stats = (aln_identity, aln_identity_ccg)
                asm.add((ref_acc_key, accver), stats)

        logger.debug(("Alignment between %s and reference %s has identity "
            "%f and identity (after collapsing consecutive gaps) %f") %
            (accver, ref_acc_key, aln_identity, aln_identity_ccg))

        if (aln_identity >= aln_identity_thres and
                aln_identity_ccg >= aln_identity_ccg_thres):
            seqs_filtered[accver] = seq

    if asm is not None:
        # Save the new memoized values
        asm.save()

    if remove_ref_acc:
        # Do not include ref_acc_key in the output
        del seqs_filtered[ref_acc_key]

    return seqs_filtered
