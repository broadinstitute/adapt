"""Structure(s) and functions for working with alignments of sequences.
"""

from collections import defaultdict
import logging
import statistics

import numpy as np
from math import log2, isclose

from adapt.utils import oligo
from adapt.utils import lsh
from adapt.utils import predict_activity
from adapt.utils import search

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class SequenceList:
    """Immutable collection of sequences.
    """

    def __init__(self, seqs):
        """
        Args:
            seqs: list of str, where seqs[i] is the i'th sequence
        """
        self.seqs = seqs
        self.num_sequences = len(seqs)

    def make_list_of_seqs(self, seqs_to_consider=None, include_idx=False,
            remove_gaps=False):
        """Construct list of sequences.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if None,
                use all)
            include_idx: instead of a list of str giving the sequences in
                the alignment, return a list of tuples (seq, idx) where seq
                is a str giving a sequence and idx is the index in the
                list
            remove_gaps: if True, remove gaps ('-') from the returned
                sequences; note that there should not be gaps because,
                in general, this should represent unaligned sequences

        Returns:
            list of str giving the sequences (or, list of
            tuples if include_idx is True)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        def seq_str(i):
            # Get the str of the sequence with index i
            s = self.seqs[i]
            if remove_gaps:
                s = s.replace('-', '')
            return s

        if include_idx:
            return [(seq_str(i), i) for i in seqs_to_consider]
        else:
            return [seq_str(i) for i in seqs_to_consider]


class Alignment(SequenceList):
    """Immutable collection of sequences that have been aligned.

    This stores sequences in column-major order, which should make it more
    efficient to extract pieces of the alignment by position and to generate
    consensus sequences.
    """

    def __init__(self, seqs, seq_norm_weights=None):
        """
        Args:
            seqs: list of str representing an alignment in column-major order
                (i.e., seqs[i] is a string giving the bases in the sequences
                at the i'th position of the alignment; it is not the i'th
                sequence)
            seq_norm_weights: list of normalized weights, where
                seq_norm_weights[j] is the weight for the j'th sequence, Should
                sum to 1
        """
        self.seq_length = len(seqs)
        self.num_sequences = len(seqs[0])
        for s in seqs:
            assert len(s) == self.num_sequences

        self.seqs = seqs
        if seq_norm_weights == None:
            self.seq_norm_weights = [1/self.num_sequences for _ in seqs[0]]
        else:
            assert isclose(sum(seq_norm_weights), 1)
            self.seq_norm_weights = seq_norm_weights

        # Memoize information missing data at each position
        self._frac_missing = None
        self._median_missing = None

    def extract_range(self, pos_start, pos_end):
        """Extract range of positions from alignment.

        Args:
            pos_start: start position of extraction (inclusive)
            pos_end: end position of extraction (exclusive)

        Returns:
            object of type Alignment including only the specified range
        """
        return Alignment(self.seqs[pos_start:pos_end],
                         seq_norm_weights=self.seq_norm_weights)

    def _compute_frac_missing(self):
        """Compute fraction of sequences with missing data at each position.
        """
        self._frac_missing = [0 for _ in range(self.seq_length)]
        for j in range(self.seq_length):
            num_n = sum(1 for i in range(self.num_sequences)
                        if self.seqs[j][i] == 'N')
            self._frac_missing[j] = float(num_n) / self.num_sequences

    def seq_idxs_weighted(self, seq_idxs):
        """Find the total weight of a subset of sequences in the alignment

        Args:
            seq_idxs: indexes of the sequences to get the weight of

        Returns:
            sum of the weights of the sequences specified by seq_idxs
        """
        return sum(self.seq_norm_weights[seq_idx]
                   for seq_idx in seq_idxs)

    def median_sequences_with_missing_data(self):
        """Compute the median fraction of sequences with missing data, across
        positions.

        Returns:
            median fraction of sequences that have 'N', taken across the
            positions in the alignment
        """
        if self._median_missing is None:
            if self._frac_missing is None:
                self._compute_frac_missing()
            self._median_missing = statistics.median(self._frac_missing)
        return self._median_missing

    def frac_missing_at_pos(self, pos):
        """Find fraction of sequences with missing data at position.

        Args:
            pos: position in the alignment at which to return the fraction
                of sequences with 'N'

        Returns:
            fraction of sequences at pos with 'N'
        """
        if self._frac_missing is None:
            self._compute_frac_missing()
        return self._frac_missing[pos]

    def seqs_with_gap(self, seqs_to_consider=None):
        """Determine sequences in the alignment that have a gap.

        Args:
            seqs_to_consider: only look within seqs_to_consider for
                sequences with a gap; if None, then look in all
                sequences

        Returns:
            set of indices of sequences that contain a gap
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        has_gap = set()
        for j in range(self.seq_length):
            has_gap.update(i for i in seqs_to_consider if self.seqs[j][i] == '-')
        return has_gap

    def seqs_with_required_flanking(self, oligo_start, oligo_length,
            required_flanking_seqs, seqs_to_consider=None):
        """Determine sequences in the alignment with required flanking sequence.

        If no flanking sequences are required, this says that all sequences
        have the required flanking sequence.

        Args:
            oligo_start: start position of the oligo in the alignment
            oligo_length: length of the oligo
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the oligo (in the target, not the oligo) that must be
                required for a oligo to bind; if either is None, no
                flanking sequence is required for that end
            seqs_to_consider: only look within seqs_to_consider for
                sequences with the flanking sequences; if None, then
                look in all sequences

        Returns:
            set of indices of sequences that have the required flanking
            sequences
        """
        if seqs_to_consider is None:
            seqs_to_consider = set(range(self.num_sequences))

        def seqs_that_equal(query, start_pos, end_pos):
            # Return set of sequence indices from seqs_to_consider that,
            # in the range [start_pos, end_pos) do not contain a gap and
            # are equal to query
            if start_pos < 0 or end_pos > self.seq_length:
                return set()
            query_matches = set()
            for i in seqs_to_consider:
                s = ''.join(self.seqs[j][i] for j in range(start_pos, end_pos))
                if '-' not in s and oligo.query_target_eq(query, s):
                    query_matches.add(i)
            return query_matches

        required_flanking5, required_flanking3 = required_flanking_seqs

        seqs_with_required_flanking = set(seqs_to_consider)
        if required_flanking5 is not None and len(required_flanking5) > 0:
            query = required_flanking5
            seqs_with_required_flanking &= seqs_that_equal(
                required_flanking5, oligo_start - len(required_flanking5),
                oligo_start)
        if required_flanking3 is not None and len(required_flanking3) > 0:
            seqs_with_required_flanking &= seqs_that_equal(
                required_flanking3, oligo_start + oligo_length,
                oligo_start + oligo_length + len(required_flanking3))
        return seqs_with_required_flanking

    def make_list_of_seqs(self, seqs_to_consider=None, include_idx=False,
            remove_gaps=False):
        """Construct list of sequences from the alignment.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if None,
                use all)
            include_idx: instead of a list of str giving the sequences in
                the alignment, return a list of tuples (seq, idx) where seq
                is a str giving a sequence and idx is the index in the
                alignment
            remove_gaps: if True, remove gaps ('-') from the returned
                sequences (i.e., so the sequences are as they appeared
                prior to alignment); in this case, the returned sequences
                may be of differing lengths

        Returns:
            list of str giving the sequences in the alignment (or, list of
            tuples if include_idx is True)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        def seq_str(i):
            # Construct the str of the sequence with index i
            s = ''.join(self.seqs[j][i] for j in range(self.seq_length))
            if remove_gaps:
                s = s.replace('-', '')
            return s

        if include_idx:
            return [(seq_str(i), i) for i in seqs_to_consider]
        else:
            return [seq_str(i) for i in seqs_to_consider]

    def determine_consensus_sequence(self, seqs_to_consider=None):
        """Determine consensus sequence from the alignment.

        At each position, the consensus is the most common allele even if it is
        not the majority; the consensus will not be an ambiguity code (except N
        if all bases are N). Ties are broken arbitrarily but deterministically.

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
            for j in seqs_to_consider:
                b = self.seqs[i][j]
                if b in counts:
                    counts[b] += self.seq_norm_weights[j]
                elif b == 'N':
                    # skip N
                    continue
                elif b in oligo.FASTA_CODES:
                    for c in oligo.FASTA_CODES[b]:
                        counts[c] += (self.seq_norm_weights[j] /
                                      len(oligo.FASTA_CODES[b]))
                else:
                    raise ValueError("Unknown base call %s" % b)
            counts_sorted = sorted(list(counts.items()))
            max_base = max(counts_sorted, key=lambda x: x[1])[0]
            if counts[max_base] == 0:
                consensus += 'N'
            else:
                consensus += max_base

        return consensus

    def determine_most_common_sequences(self, seqs_to_consider=None,
            skip_ambiguity=False, n=1):
        """Determine the most common of the sequences from the alignment.

        If we imagine each sequence in the alignment as existing some
        number of times in the alignment, this effectively picks the
        mode when n is 1.

        Ties are broken arbitrarily but deterministically.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if
                None, use all)
            skip_ambiguity: if True, ignore any sequences that contain
                an ambiguity code (i.e., only count sequences where all
                bases are 'A', 'T', 'C', or 'G')
            n: number of sequences to return. If there are <n unique sequences,
                all sequences are returned

        Returns:
            list of str representing the n most common of the sequences in
            order of count (or None if there are no suitable strings)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        allowed_chars = ['A', 'T', 'C', 'G', '-']

        # Convert all sequences into strings, and count the number of each
        seqs_str = self.make_list_of_seqs(seqs_to_consider=seqs_to_consider)
        seq_count = defaultdict(lambda: 0)
        for j, seq_str in enumerate(seqs_str):
            if skip_ambiguity:
                # Skip over this sequence if it contains any base that is
                # ambiguous
                skip = False
                for i in range(len(seq_str)):
                    if seq_str[i] not in allowed_chars:
                        skip = True
                        break
                if skip:
                    continue

            seq_count[seq_str] += self.seq_norm_weights[j]

        if len(seq_count) == 0:
            # There are no suitable strings (e.g., all contain ambiguity)
            return None

        # Find the most common sequence (break ties by arbitrarily by sequence)
        counts_sorted = sorted(list(seq_count.items()),
                               key=lambda x: (-x[1], x[0]))
        max_seq_strs = [count_sorted[0] for count_sorted in counts_sorted[0:n]]
        return max_seq_strs

    def determine_representative_oligos(self, start, oligo_length,
        seqs_to_consider, clusterer, missing_threshold=1,
        pre_filter_fns=[], required_flanking_seqs=(None, None),
        include_overall_consensus=True):
        """Construct a set of oligos representative of the target sequences.

        This is similar to construct_oligo(), except returns a set of
        representative sequences rather than a single best one.

        Args:
            start: start position in alignment at which to target
            oligo_length: length of the representative sequence (oligo)
            seqs_to_consider: dict mapping universe group ID to collection of
                indices to use when constructing the representative sequences
            clusterer: object of SequenceClusterer to use for clustering
                potential oligo sequences; it must have been initialized with
                a family suitable for oligos of length oligo_length; if None,
                then don't cluster, and instead draw a consensus from all
                the sequences
            missing_threshold: do not construct representative sequences if
                the fraction of sequences with missing data, at any position
                in the target range, exceeds this threshold
            pre_filter_fns: if set, the value of this argument is a list
                of functions f(x) such that this will only construct a oligo x
                for which each f(x) is True
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the oligo (in the target, not the oligo) that must be
                present for a oligo to bind; if either is None, no
                flanking sequence is required for that end
            include_overall_consensus: includes, as a representative
                sequence, the consensus across all sequences; this is
                optional because it may not be "representative" of any
                sequences as well as the cluster consensuses

        Returns:
            set of representative sequences
        """
        assert start + oligo_length <= self.seq_length
        assert len(seqs_to_consider) > 0

        for pos in range(start, start + oligo_length):
            if self.frac_missing_at_pos(pos) > missing_threshold:
                raise search.CannotConstructOligoError(("Too much missing data at "
                    "a position in the target range"))

        aln_for_oligo = self.extract_range(start, start + oligo_length)

        # Before modifying seqs_to_consider, make a copy of it
        seqs_to_consider_cp = {}
        for group_id in seqs_to_consider.keys():
            seqs_to_consider_cp[group_id] = set(seqs_to_consider[group_id])
        seqs_to_consider = seqs_to_consider_cp

        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Ignore any sequences in the alignment that have a gap in
        # this region
        seqs_with_gap = set(aln_for_oligo.seqs_with_gap(all_seqs_to_consider))
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].difference_update(seqs_with_gap)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Only consider sequences in the alignment that have the
        # required flanking sequence(s)
        seqs_with_flanking = self.seqs_with_required_flanking(
            start, oligo_length, required_flanking_seqs,
            seqs_to_consider=all_seqs_to_consider)
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].intersection_update(seqs_with_flanking)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # If every sequence in this region has a gap or does not contain
        # required flanking sequence, then there are none left to consider
        if len(all_seqs_to_consider) == 0:
            raise search.CannotConstructOligoError(("All sequences in region have "
                "a gap and/or do not contain required flanking sequences"))

        representatives = set()
        def consider_and_add_oligo(olg):
            if 'N' in olg:
                # Skip this guide; all sequences at a position in the cluster
                # under consideration are 'N'
                return False
            for is_suitable_fn in pre_filter_fns:
                if is_suitable_fn(olg) is False:
                    # Skip this oligo/cluster
                    return False
            representatives.add(olg)
            return True

        if include_overall_consensus:
            # Make a representative guide be the consensus of all
            # sequences
            olg = aln_for_oligo.determine_consensus_sequence(
                    all_seqs_to_consider)
            consider_and_add_oligo(olg)

        # Cluster the sequences
        seq_rows = aln_for_oligo.make_list_of_seqs(all_seqs_to_consider,
            include_idx=True)
        clusters = clusterer.cluster(seq_rows)

        # Take the consensus of each cluster to be the representative
        for cluster_idxs in clusters:
            olg = aln_for_oligo.determine_consensus_sequence(
                cluster_idxs)
            consider_and_add_oligo(olg)

        return representatives

    def compute_activity(self, start, olg_sequence, predictor, mutator=None):
        """Compute activity between an oligo sequence and every target sequence
        in the alignment.

        This only considers the region of the alignment within the range
        [start, start + len(olg_sequence)), along with sequence context.

        Args:
            start: start position in alignment at which to target
            olg_sequence: str representing oligo sequence
            predictor: a adapt.utils.predict_activity.Predictor object
            mutator: a adapt.utils.mutate.Mutator object

        Returns:
            numpy array x where x[i] gives the predicted activity between
            olg_sequence and the sequence in the alignment at index i. If
            mutator is not None, x[i] gives the predicted activity after the
            mutations specified in the mutator.
        """
        oligo_length = len(olg_sequence)
        assert start + oligo_length <= self.seq_length

        if isinstance(predictor, predict_activity.SimpleBinaryPredictor):
            if mutator:
                # Extract the target sequences, including context to use with
                # prediction (i.e. the flanking sequences, if they exist)
                left_context = 0
                right_context = 0
                if predictor:
                    if predictor.required_flanking_seqs[0]:
                        left_context = len(predictor.required_flanking_seqs[0])
                    if predictor.required_flanking_seqs[1]:
                        right_context = len(predictor.required_flanking_seqs[1])

                aln_for_oligo_with_context = self.extract_range(
                        start - left_context,
                        start + oligo_length + right_context)
                seq_rows_with_context = aln_for_oligo_with_context.make_list_of_seqs()

                # Start array of predicted activities; make this all 0s so that
                # sequences for which activities are not computed (e.g., gap)
                # have activity=0
                activities = np.zeros(self.num_sequences)
                for i, seq_with_context in enumerate(seq_rows_with_context):
                    activity = mutator.compute_mutated_activity(predictor,
                                                                seq_with_context,
                                                                olg_sequence,
                                                                start=start)
                    activities[i] = activity
                return activities
            # Do not use a model; just predict binary activity (1 or 0)
            # based on distance between oligo and targets
            return predictor.compute_activity(start, olg_sequence, self)

        # Extract the target sequences, including context to use with
        # prediction
        if (start - predictor.context_nt < 0 or
                start + oligo_length + predictor.context_nt > self.seq_length):
            raise search.CannotConstructOligoError(("The context needed for "
                "the target to predict activity falls outside the "
                "range of the alignment at this position"))
        aln_for_oligo_with_context = self.extract_range(
                start - predictor.context_nt,
                start + oligo_length + predictor.context_nt)

        # Ignore any sequences in the alignment that have a gap in
        # this region
        all_seqs_to_consider = set(range(self.num_sequences))
        seqs_with_gap = set(aln_for_oligo_with_context.seqs_with_gap(
            all_seqs_to_consider))
        seqs_to_consider = all_seqs_to_consider.difference(seqs_with_gap)

        seq_rows_with_context = aln_for_oligo_with_context.make_list_of_seqs(
                seqs_to_consider, include_idx=True)

        # Start array of predicted activities; make this all 0s so that
        # sequences for which activities are not computed (e.g., gap)
        # have activity=0
        activities = np.zeros(self.num_sequences)

        # Determine what calls to make to predictor.compute_activity(); it
        # is best to batch these
        pairs_to_eval = []
        pairs_to_eval_seq_idx = []
        evals = []

        if mutator:
            for seq_with_context, seq_idx in seq_rows_with_context:
                activity = mutator.compute_mutated_activity(predictor,
                                                            seq_with_context,
                                                            olg_sequence,
                                                            start=start)
                evals.append(activity)
                pairs_to_eval_seq_idx.append(seq_idx)
        else:
            for seq_with_context, seq_idx in seq_rows_with_context:
                pair = (seq_with_context, olg_sequence)
                pairs_to_eval.append(pair)
                pairs_to_eval_seq_idx.append(seq_idx)
            # Evaluate activity
            evals = predictor.compute_activity(start, pairs_to_eval)

        for activity, seq_idx in zip(evals, pairs_to_eval_seq_idx):
            # Fill in the activity for seq_idx
            activities[seq_idx] = activity

        return activities

    def sequences_bound_by_oligo(self, olg_seq, olg_start, mismatches,
            allow_gu_pairs, required_flanking_seqs=(None, None)):
        """Determine the sequences to which a oligo hybridizes.

        Args:
            olg_seq: sequence of the oligo
            olg_start: start position of the oligo in the alignment
            mismatches: threshold on number of mismatches for determining whether
                a oligo would hybridize to a target sequence
            allow_gu_pairs: if True, tolerate G-U base pairs between a
                oligo and target when computing whether a oligo binds
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the oligo (in the target, not the oligo) that must be
                present for a oligo to bind; if either is None, no
                flanking sequence is required for that end

        Returns:
            collection of indices of sequences to which the oligo will
            hybridize
        """
        assert olg_start + len(olg_seq) <= self.seq_length

        aln_for_oligo = self.extract_range(olg_start, olg_start + len(olg_seq))
        seq_rows = aln_for_oligo.make_list_of_seqs(include_idx=True)

        seqs_with_flanking = self.seqs_with_required_flanking(
            olg_start, len(olg_seq), required_flanking_seqs)

        binding_seqs = []
        for seq, seq_idx in seq_rows:
            if (seq_idx in seqs_with_flanking and
                    oligo.binds(olg_seq, seq, mismatches, allow_gu_pairs)):
                binding_seqs += [seq_idx]
        return binding_seqs

    def position_entropy(self):
        """Determine the entropy at each position in the alignment.

        Returns:
            list with the entropy of position i listed at index i
        """

        position_entropy = []
        for i in range(self.seq_length):
            counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0, '-': 0}
            for j in range(self.num_sequences):
                b = self.seqs[i][j]
                if b in counts:
                    counts[b] += self.seq_norm_weights[j]
                elif b in oligo.FASTA_CODES:
                    for c in oligo.FASTA_CODES[b]:
                        counts[c] += (self.seq_norm_weights[j] /
                                      len(oligo.FASTA_CODES[b]))
                else:
                    raise ValueError("Unknown base call %s" % b)

            # Calculate entropy
            probabilities = [counts[base] for base in counts]
            this_position_entropy = sum([-p*log2(p) for p in probabilities if p > 0])

            position_entropy.append(this_position_entropy)

        return position_entropy

    def base_percentages(self):
        """Determines the percentage of each base pair in the alignment.

        Returns:
            dictionary of base pair to its percentage in the alignment
        """
        counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
        for i in range(self.seq_length):
            for j in range(self.num_sequences):
                b = self.seqs[i][j]
                if b in counts:
                    counts[b] += self.seq_norm_weights[j]
                elif b in oligo.FASTA_CODES:
                    for c in oligo.FASTA_CODES[b]:
                        counts[c] += (self.seq_norm_weights[j] /
                                      len(oligo.FASTA_CODES[b]))
                elif b != '-':
                    raise ValueError("Unknown base call %s" % b)

        # Needs to be normalized because this is across all locations in the
        # alignment and gaps are not counted
        total = sum(counts.values())
        for base in counts:
            counts[base] /= total
        return counts

    @staticmethod
    def from_list_of_seqs(seqs, seq_norm_weights=None):
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

        if num_sequences == 0:
            raise Exception(("Trying to construct an alignment consisting "
                "of 0 sequences"))

        seq_length = len(seqs[0])
        for s in seqs:
            if len(s) != seq_length:
                raise ValueError("Sequences must be the same length")

        seqs_col = ['' for _ in range(seq_length)]
        for j in range(seq_length):
            seqs_col[j] = ''.join(seqs[i][j] for i in range(num_sequences))

        return Alignment(seqs_col, seq_norm_weights=seq_norm_weights)


class SequenceClusterer:
    """Supports clustering sequences using LSH.

    This is useful for approximating an optimal oligo over a collection of
    sequences.
    """

    def __init__(self, family, k=10):
        """
        Args:
            family: object of hash family from which to draw hash functions
            k: number of hash functions to draw from a family of
                hash functions for amplification; each hash function is then
                the concatenation (h_1, h_2, ..., h_k)
        """
        self.hash_concat = lsh.HashConcatenation(family, k)

    def cluster(self, seqs_with_idx):
        """Generate clusters of given sequences.

        Args:
            seqs_with_idx: collection of tuples (seq, idx) where seq
                is a str giving a sequence and idx is an identifier
                of the sequence (e.g., index in an alignment); all idx
                in the collection must be unique

        Returns:
            collection {S_i} where each S_i is a set of idx from
            seqs_with_idx that form a cluster; the S_i's are disjoint and
            their union is all the idx
        """
        def g(seq):
            # Hash a sequence; since the output of self.hash_concat is a
            # tuple of strings, we can join it into one string
            return ''.join(self.hash_concat.g(seq))

        d = defaultdict(set)
        for seq, idx in seqs_with_idx:
            d[g(seq)].add(idx)
        return d.values()

    def largest_cluster(self, seqs_with_idx):
        """Find sequences that form the largest cluster.

        Args:
            seqs_with_idx: collection of tuples (seq, idx) where seq
                is a str giving a sequence and idx is an identifier
                of the sequence (e.g., index in an alignment); all idx
                in the collection must be unique

        Returns:
            set of idx from seqs_with_idx whose sequences form the largest
            cluster
        """
        clusters = self.cluster(seqs_with_idx)
        return max(clusters, key=lambda c: len(c))

