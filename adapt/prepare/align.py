"""Functions for performing an alignment of sequences, with curation.
"""

from collections import OrderedDict
import hashlib
import logging
import os
import random
import re
import subprocess
import tempfile
try:
    import boto3
    from botocore.exceptions import ClientError
except:
    cloud = False
else:
    cloud = True

from adapt.utils import seq_io

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

    def __init__(self, path, aws_access_key_id=None, aws_secret_access_key=None):
        """
            Args:
                path: path to directory in which to read and store
                    memoized alignments; if path begins with "s3://",
                    stores bucket
        """
        if path[:5] == "s3://":
            # Store path as an S3 Bucket Object
            folders = path.split("/")
            self.path = "/".join(folders[3:])
            self.bucket = folders[2]
            if aws_access_key_id is not None and aws_secret_access_key is not None:
                self.S3 = boto3.client("s3", 
                    aws_access_key_id=aws_access_key_id, 
                    aws_secret_access_key=aws_secret_access_key)
            else:
                self.S3 = boto3.client("s3")
        else:
            self.path = path
            self.bucket = None
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def _hash_accvers_filepath(self, accvers):
        """Generate a path or S3 key to use as the filename for an alignment.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            path to file or S3 key that should store accvers
        """
        h = hashlib.md5(str(sorted(set(accvers))).encode('utf-8')).hexdigest()
        if self.bucket:
            return "/".join([self.path, h])
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
        seqs = None

        if self.bucket:
            # Download file from source if it exists,
            # otherwise return None
            try:
                f = self.S3.get_object(Bucket = self.bucket, Key = p)
            except ClientError as e:
                if e.response['Error']['Code'] == "NoSuchKey":
                    return None
                else:
                    raise
            else:
                lines = [line.decode('utf-8') for line in f["Body"].iter_lines()]
                seqs = seq_io.process_fasta(lines)

        elif os.path.exists(p):
            # Read the fasta
            seqs = seq_io.read_fasta(p)

        if seqs:
            # Verify that the accessions in the file match accs 
            # (i.e., that there is not a collision)
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

        if self.bucket:
            # Make temporary file to store FASTA, then upload to S3
            p_tmp = tempfile.NamedTemporaryFile(delete=False)
            p_tmp.close()
            seq_io.write_fasta(seqs, p_tmp.name)
            with open(p_tmp.name, 'rb') as f:
                self.S3.put_object(Bucket=self.bucket, Key=p, Body=f)
            os.unlink(p_tmp.name)
        else:
            # Generate a random 8-digit hex to append to p temporarily, so that
            # two files don't write to p at the same time
            p_tmp = "%s.%08x" % (p, random.getrandbits(32))
            seq_io.write_fasta(seqs, p_tmp)
            os.replace(p_tmp, p)
        

class AlignmentStatMemoizer:
    """Memoizer for statistics on alignments.

    This stores an alignment using a hash of accession.version
    in the alignments.

    This stores, for each alignment, the tuple (aln_identity, aln_identity_ccg).
    """

    def __init__(self, path, aws_access_key_id=None, aws_secret_access_key=None):
        """
            Args:
                path: path to directory in which to read and store memoized
                    stats; if path begins with "s3://", stores bucket
        """
        if path[:5] == "s3://":
            # Store path as an S3 Bucket Object
            folders = path.split("/")
            self.path = "/".join(folders[3:])
            self.bucket = folders[2]
            if aws_access_key_id is not None and aws_secret_access_key is not None:
                self.S3 = boto3.client("s3", 
                    aws_access_key_id=aws_access_key_id, 
                    aws_secret_access_key=aws_secret_access_key)
            else:
                self.S3 = boto3.client("s3")
        else:
            self.path = path
            self.bucket = None
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def _hash_accvers_filepath(self, accvers):
        """Generate a path or S3 key to use as the filename for an alignment.

        Let h be the hash of accers and and h_2 be the first two
        (hexadecimal) digits of h. This stores h in a directory
        corresponding to h_2, in order to decrease the number of
        files per directory.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            path to file or S3 key that should store stats on accvers
        """
        h = hashlib.md5(str(sorted(set(accvers))).encode('utf-8')).hexdigest()

        h_2 = h[:2]

        if self.bucket:
            return "/".join([self.path, h_2, h])

        h_dir = os.path.join(self.path, h_2)
        if not os.path.exists(h_dir):
            os.makedirs(h_dir)

        return os.path.join(h_dir, h)

    def get(self, accvers):
        """Get memoized stats for alignment.

        Args:
            accvers: collection of accession.version in an alignment

        Returns:
            (aln_identity, aln_identity_ccg) for alignment of
            accvers; or None if accvers is not memoized
        """
        p = self._hash_accvers_filepath(accvers)
        ls = None

        if self.bucket:
            # Download file from source if it exists,
            # otherwise return None
            try:
                f = self.S3.get_object(Bucket = self.bucket, Key = p)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return None
                else:
                    raise e
            else:
                lines = f["Body"].read().decode('utf-8').rstrip()
                ls = lines.split('\t')

        elif os.path.exists(p):
            # Read the file
            with open(p) as f:
                lines = [line.rstrip() for line in f]
            assert len(lines) == 1  # should be just 1 line
            ls = lines[0].split('\t')
        
        if ls:
            # Verify that the accessions in file match accvers
            # (i.e., that there is not a collision)
            accvers_in_p = ls[0].split(',')
            if set(accvers) == set(accvers_in_p):
                stats = (float(ls[1]), float(ls[2]))
                return stats
            else:
                return None
        else:
            return None

    def save(self, accvers, stats):
        """Memoize statistics for an alignment.

        Args:
            accvers: collection of accession.version in an alignment
            stats: tuple (aln_identity, aln_identity_ccg)
        """
        p = self._hash_accvers_filepath(accvers)
        accvers_str = ','.join(accvers)

        if self.bucket:
            body = '\t'.join([accvers_str, str(stats[0]), str(stats[1])]) + '\n'
            self.S3.put_object(Bucket=self.bucket, Key=p,
                Body=body.encode('utf-8'))
        else:
            # Generate a random 8-digit hex to append to p temporarily, so that
            # two files don't write to p at the same time
            p_tmp = "%s.%08x" % (p, random.getrandbits(32))
            with open(p_tmp, 'w') as fw:
                fw.write(('\t'.join([accvers_str, str(stats[0]), str(stats[1])]) +
                    '\n'))
            os.rename(p_tmp, p)


_mafft_exec = None
def set_mafft_exec(mafft_path):
    """Set path to the mafft executable.

    Args:
        mafft_path: path to mafft executable
    """
    global _mafft_exec
    # Verify the path exists and is an executable
    if not os.path.isfile(mafft_path) or not os.access(mafft_path, os.X_OK):
        raise FileNotFoundError(("Path to mafft (%s) does not exist or is not "
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

    if len(seqs) == 1:
        # There's one sequence; simply output it
        seqs_aligned = OrderedDict(seqs)
        if am is not None:
            am.save(seqs_aligned)
        return OrderedDict(seqs_aligned)

    global _mafft_exec
    # Check that mafft executable has been set
    if _mafft_exec is None:
        raise Exception(("Path to mafft executable must be set using "
            "_set_mafft_exec()"))

    # Write a fasta of these sequences to a named temp file, and
    # create one to write the output fasta to
    # Files closed for Windows, which will not allow files to be 
    # opened twice (files opened again in write_fasta/read_fasta)
    in_fasta = tempfile.NamedTemporaryFile(delete=False)
    in_fasta.close()
    seq_io.write_fasta(seqs, in_fasta.name)
    out_fasta = tempfile.NamedTemporaryFile(delete=False)

    # Setup arguments to mafft
    params = ['--preservecase', '--thread', '-1']
    max_seq_len = max(len(seq) for seq in seqs.values())
    if len(seqs) < 10 and max_seq_len < 10000:
        # Accuracy oriented
        params += ['--maxiterate', '1000', '--localpair']
    elif len(seqs) > 50000 or max_seq_len > 200000:
        # Speed oriented
        params += ['--retree', '1', '--maxiterate', '0']
    elif len(seqs) > 1000 or max_seq_len > 100000:
        # Speed oriented
        params += ['--retree', '2', '--maxiterate', '0']
    else:
        # Standard
        params += ['--retree', '2', '--maxiterate', '2']

    # Call mafft
    cmd = [_mafft_exec] + params + [in_fasta.name]
    subprocess.call(cmd, stdout=out_fasta, stderr=subprocess.DEVNULL)
    out_fasta.close()

    # Read the output fasta into a dict
    seqs_aligned = seq_io.read_fasta(out_fasta.name)

    # Delete temp files
    os.unlink(in_fasta.name)
    os.unlink(out_fasta.name)

    if len(seqs) > 0 and len(seqs_aligned) == 0:
        logger.critical(("The generated alignment contains no sequences; "
            "it is possible that mafft failed, e.g., due to running out "
            "of memory"))

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


def curate_against_ref(seqs, ref_accs, asm=None,
        aln_identity_thres=0.5, aln_identity_ccg_thres=0.6,
        aln_identity_long_thres=(100000, 0.9, 0.9),
        remove_ref_accs=[]):
    """Curate sequences by aligning pairwise with reference sequences.

    This can make use of an object of AlignmentStatMemoizer to
    memoize values, since the pairwise alignment can be slow.

    This compares each sequence against a list of reference accessions,
    and keeps any that satisfies the criteria against any reference
    sequence. In other words, it filters out any sequences that fail to
    meet the criteria against all reference sequences.

    Args:
        seqs: dict mapping sequence accession to sequences
        ref_accs: list of accessions of reference sequences to use for curation;
            for each ref_acc in ref_accs, either itself or ref_acc.[version]
            for some version must be a key in seqs
        asm: AlignmentStatMemoizer to use for memoization of values; or
            None to not memoize
        aln_identity_thres: filter out any sequences with less than
            this fraction identity to all ref_acc in ref_accs (taken over the
            whole alignment)
        aln_identity_ccg_thres: filter out any sequences with less than
            this fraction identity to all ref_acc in ref_accs, after collapsing
            consecutive gaps to a single gap; this should be at least
            aln_identity_thres
        aln_identity_long_thres: if set, then adjust aln_identity_thres
            and aln_identity_ccf_thres for long genomes. This is a tuple
            (len, a, b) such that, if the length of the longest reference
            sequence is >= len, then aln_identity_thres takes on the
            value a and aln_identity_ccf_thres takes on the value b. This
            is useful to be stronger in curating long genomes (ideally,
            removing sequences with large structural changes), which will
            take long to align in a multiple sequence alignment
        remove_ref_accs: a list of ref_acc, specifying to not include each
            ref_acc in the output

    Returns:
        dict mapping sequence accession.version to sequences, filtered by
        ones that can align reasonably well with some ref_acc in ref_accs
    """
    logger.info(("Curating %d sequences against references %s "
        "(this can take a while if alignment stats are not memoized)") %
        (len(seqs), ref_accs))

    # Find each ref_acc in seqs
    ref_acc_to_key = {}
    max_ref_seq_len = 0
    for ref_acc in ref_accs:
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
                raise Exception(("Each ref_acc must be in seqs; it is possible "
                    "that the reference accession '%s' is invalid or could not "
                    "be found") % ref_acc)
        ref_acc_to_key[ref_acc] = ref_acc_key
        max_ref_seq_len = max(max_ref_seq_len, len(seqs[ref_acc_key]))

    if aln_identity_long_thres is not None:
        len_thres, a, b = aln_identity_long_thres
        if max_ref_seq_len >= len_thres:
            # This is a long sequence; be more strict in curation
            aln_identity_thres = a
            aln_identity_ccf_thres = b

    seqs_filtered = OrderedDict()
    for accver, seq in seqs.items():
        for ref_acc in ref_accs:
            ref_acc_key = ref_acc_to_key[ref_acc]
            if accver == ref_acc_key:
                # ref_acc_key will align well with ref_acc_key, so include it
                # automatically
                seqs_filtered[accver] = seq
                break

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
                    asm.save((ref_acc_key, accver), stats)

            logger.debug(("Alignment between %s and reference %s has identity "
                "%f and identity (after collapsing consecutive gaps) %f") %
                (accver, ref_acc_key, aln_identity, aln_identity_ccg))

            if (aln_identity >= aln_identity_thres and
                    aln_identity_ccg >= aln_identity_ccg_thres):
                # Include accver in the filtered output
                seqs_filtered[accver] = seq
                break

    logger.info(("After curation, %d of %d sequences (with unique accession) "
        "were kept; %d of these are references that will be removed"),
        len(seqs_filtered), len(seqs), len(remove_ref_accs))

    for remove_ref_acc in remove_ref_accs:
        # Do not include the ref_acc_key corresponding to remove_ref_acc
        # in the output
        ref_acc_key = ref_acc_to_key[remove_ref_acc]
        del seqs_filtered[ref_acc_key]

    return seqs_filtered
