"""Structure(s) and functions for working with alignments of sequences.
"""

import logging

from probedesign.utils import probe

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class Alignment:
    """Immutable collection of sequences that have been aligned.

    This stores sequences in column-major order, which should make it more
    efficient to extract pieces of the alignment by position and to generate
    consensus sequences.
    """

    def __init__(self, seqs):
        """
        Args:
            seqs: list of str representing an alignment in column-major order
                (i.e., seqs[i] is a string giving the bases in the sequences
                at the i'th position of the alignment; it is not the i'th sequence)
        """
        self.seq_length = len(seqs)
        self.num_sequences = len(seqs[0])
        for s in seqs:
            assert len(s) == self.num_sequences

        self.seqs = seqs

    def extract_range(self, pos_start, pos_end):
        """Extract range of positions from alignment.

        Args:
            pos_start: start position of extraction (inclusive)
            pos_end: end position of extraction (exclusive)

        Returns:
            object of type Alignment including only the specified range
        """
        return Alignment(self.seqs[pos_start:pos_end])

    def has_indel(self):
        """Determine whether there is an indel in the alignment.

        Returns:
            True iff there is an indel in the alignment
        """
        for s in self.seqs:
            if '-' in s:
                return True
        return False

    def construct_probe(self, start, probe_length, seqs_to_consider, mismatches):
        """Construct a single probe to target a set of sequences in the alignment.

        This constructs a probe to target sequence within the range [start,
        start+probe_length]. It only considers the sequences with indices given in
        seqs_to_consider.

        Args:
            start: start position in alignment at which to target
            probe_length: length of the probe
            seqs_to_consider: collection of indices of sequences to use when
                constructing the probe
            mismatches: threshold on number of mismatches for determining whether
                a probe would hybridize to a target sequence

        Returns:
            tuple (x, y) where:
                x is the sequence of the constructed probe
                y is a list of indices of sequences (a subset of
                    seqs_to_consider) to which the probe x will hybridize
            (Note that it is possible that x binds to no sequences and that
            y will be empty.)
        """
        assert start + probe_length <= self.seq_length
        assert len(seqs_to_consider) > 0

        aln_for_probe = self.extract_range(start, start + probe_length)

        # If this region in the alignment has an indel, do not attempt to
        # construct a probe that covers it (even if the alignment formed by just
        # the sequences in seqs_to_consider do not have an indel)
        if aln_for_probe.has_indel():
            raise CannotConstructProbeError("Region in alignment has indel")

        seqs_to_consider = sorted(list(seqs_to_consider))

        # First construct the optimal probe to cover the sequences. This would be
        # a string x that maximizes the number of sequences s_i such that x and
        # s_i are equal to within 'mismatches' mismatches; it's called the "max
        # close string" or "close to most strings" problem. For simplicity, let's
        # assume for now that the optimal probe is just the consensus sequence.
        consensus = aln_for_probe.determine_consensus_sequence(seqs_to_consider)
        prb = consensus

        # If all that exists at a position in the alignment is 'N', then do
        # not attempt to cover the sequences because we do not know which
        # base to put in the probe at that position. In this case, the
        # consensus will have 'N' at that position.
        if 'N' in prb:
            raise CannotConstructProbeError("A position has all 'N'")

        seq_rows = aln_for_probe.make_list_of_seqs(seqs_to_consider)
        def determine_binding_seqs(prb_sequence):
            binding_seqs = []
            for seq_idx, seq in zip(seqs_to_consider, seq_rows):
                if probe.probe_binds(prb_sequence, seq, mismatches):
                    binding_seqs += [seq_idx]
            return binding_seqs

        binding_seqs = determine_binding_seqs(prb)

        # It's possible that the consensus sequence (probe) does not bind to
        # any of the sequences. In this case, simply select the first
        # sequence from seq_row that has no ambiguity and make this the probe;
        # this is guaranteed to have at least one binding sequence (itself)
        if len(binding_seqs) == 0:
            for s in seq_rows:
                if sum(s.count(c) for c in ['A', 'T', 'C', 'G']) == len(s):
                    # s has no ambiguity and is a suitable probe
                    prb = s
                    binding_seqs = determine_binding_seqs(prb)
                    break
            # If it made it here, then all of the sequences have ambiguity
            # (so none are suitable probes); prb will remain the consensus and
            # binding_seqs will still be empty

        return (prb, binding_seqs)

    def make_list_of_seqs(self, seqs_to_consider=None):
        """Construct list of sequences from the alignment.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if None,
                use all)

        Returns:
            list of str giving the sequences in the alignment
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)
        return [''.join(self.seqs[j][i] for j in range(self.seq_length))
                for i in seqs_to_consider]

    def determine_consensus_sequence(self, seqs_to_consider=None):
        """Determine consensus sequence from the alignment.

        At each position, the consensus is the most common allele even if it is
        not the majority; the consensus will not be an ambiguity code (except N
        if all bases are N). Ties are broken arbitrarily.

        This ignores 'N' bases at each position, determining the consensus from
        the other (non-'N') bases. An 'N' will only be output in the consensus
        if all bases at a position are 'N'.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if None,
                use all)

        Returns:
            str representing the consensus of the alignment
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        consensus = ''
        for i in range(self.seq_length):
            counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
            for b in [self.seqs[i][j] for j in seqs_to_consider]:
                if b in counts:
                    counts[b] += 1
                elif b == 'N':
                    # skip N
                    continue
                elif b in probe.FASTA_CODES:
                    for c in probe.FASTA_CODES[b]:
                        counts[c] += 1.0 / len(probe.FASTA_CODES[b])
                else:
                    raise ValueError("Unknown base call %s" % b)
            max_base = max(counts, key=counts.get)
            if counts[max_base] == 0:
                consensus += 'N'
            else:
                consensus += max_base

        return consensus

    def sequences_bound_by_probe(self, prb_seq, prb_start, mismatches):
        """Determine the sequences to which a probe hybridizes.

        Args:
            prb_seq: seequence of the probe
            prb_start: start position of the probe in the alignment
            mismatches: threshold on number of mismatches for determining whether
                a probe would hybridize to a target sequence

        Returns:
            collection of indices of sequences to which the probe will
            hybridize
        """
        assert prb_start + len(prb_seq) <= self.seq_length

        aln_for_probe = self.extract_range(prb_start, prb_start + len(prb_seq))
        seq_rows = aln_for_probe.make_list_of_seqs()

        binding_seqs = []
        for seq_idx, seq in enumerate(seq_rows):
            if probe.probe_binds(prb_seq, seq, mismatches):
                binding_seqs += [seq_idx]
        return binding_seqs

    @staticmethod
    def from_list_of_seqs(seqs):
        """Construct a Alignment from aligned list of sequences.

        If seqs is stored in row-major order, this converts to column-major order
        and creates a Alignment object.

        Args:
            seqs: list of str, all the same length, representing an alignment;
                seqs[i] is the i'th sequence

        Returns:
            object of type Alignment
        """
        num_sequences = len(seqs)
        seq_length = len(seqs[0])
        for s in seqs:
            if len(s) != seq_length:
                raise ValueError("Sequences must be the same length")

        seqs_col = ['' for _ in range(seq_length)]
        for j in range(seq_length):
            seqs_col[j] = ''.join(seqs[i][j] for i in range(num_sequences))

        return Alignment(seqs_col)


class CannotConstructProbeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
