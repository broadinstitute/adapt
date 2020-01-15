"""Structure(s) and functions for working with alignments of sequences.
"""

from collections import defaultdict
import logging
import statistics

from adapt.utils import guide
from adapt.utils import lsh

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
        return Alignment(self.seqs[pos_start:pos_end])

    def _compute_frac_missing(self):
        """Compute fraction of sequences with missing data at each position.
        """
        self._frac_missing = [0 for _ in range(self.seq_length)]
        for j in range(self.seq_length):
            num_n = sum(1 for i in range(self.num_sequences)
                        if self.seqs[j][i] == 'N')
            self._frac_missing[j] = float(num_n) / self.num_sequences

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

    def seqs_with_required_flanking(self, guide_start, guide_length,
            required_flanking_seqs, seqs_to_consider=None):
        """Determine sequences in the alignment with required flanking sequence.

        If no flanking sequences are required, this says that all sequences
        have the required flanking sequence.

        Args:
            guide_start: start position of the guide in the alignment
            guide_length: length of the guide
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the guide (in the target, not the guide) that must be
                required for a guide to bind; if either is None, no
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
                if '-' not in s and guide.query_target_eq(query, s):
                    query_matches.add(i)
            return query_matches

        required_flanking5, required_flanking3 = required_flanking_seqs

        seqs_with_required_flanking = set(seqs_to_consider)
        if required_flanking5 is not None and len(required_flanking5) > 0:
            query = required_flanking5
            seqs_with_required_flanking &= seqs_that_equal(
                required_flanking5, guide_start - len(required_flanking5),
                guide_start)
        if required_flanking3 is not None and len(required_flanking3) > 0:
            seqs_with_required_flanking &= seqs_that_equal(
                required_flanking3, guide_start + guide_length,
                guide_start + guide_length + len(required_flanking3))
        return seqs_with_required_flanking

    def construct_guide(self, start, guide_length, seqs_to_consider, mismatches,
            allow_gu_pairs, guide_clusterer, num_needed=None,
            missing_threshold=1, guide_is_suitable_fn=None,
            required_flanking_seqs=(None, None),
            predictor=None, stop_early=True):
        """Construct a single guide to target a set of sequences in the alignment.

        This constructs a guide to target sequence within the range [start,
        start+guide_length]. It only considers the sequences with indices given in
        seqs_to_consider.

        Args:
            start: start position in alignment at which to target
            guide_length: length of the guide
            seqs_to_consider: dict mapping universe group ID to collection of
                indices to use when constructing the guide
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence
            allow_gu_pairs: if True, tolerate G-U base pairs between a guide
                and target when computing whether a guide binds
            guide_clusterer: object of SequenceClusterer to use for clustering
                potential guide sequences; it must have been initialized with
                a family suitable for guides of length guide_length; if None,
                then don't cluster, and instead draw a consensus from all
                the sequences
            num_needed: dict mapping universe group ID to the number of sequences
                from the group that are left to cover in order to achieve
                a desired coverage; these are used to help construct a
                guide
            missing_threshold: do not construct a guide if the fraction of
                sequences with missing data, at any position in the target
                range, exceeds this threshold
            guide_is_suitable_fn: if set, a function f(x) such that this
                will only construct a guide x for which f(x) is True
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the guide (in the target, not the guide) that must be
                present for a guide to bind; if either is None, no
                flanking sequence is required for that end
            predictor: if set, a adapt.utils.predict_activity.Predictor
                object, with evaluate() function to predict activity and
                determine whether a guide has sufficiently high activity
                for a target. If None, do not predict activity.
            stop_early: if True, impose early stopping criteria while iterating
                over clusters to improve runtime

        Returns:
            tuple (x, y) where:
                x is the sequence of the constructed guide
                y is a list of indices of sequences (a subset of
                    values in seqs_to_consider) to which the guide x will
                    hybridize
            (Note that it is possible that x binds to no sequences and that
            y will be empty.)
        """
        # TODO: There are several optimizations that can be made to
        # this function that take advantage of G-U pairing in order
        # to lower the number of guides that need to be designed.
        # Two are:
        #  1) The function SequenceClusterer.cluster(..), which is
        #     used here, can cluster accounting for G-U pairing (e.g.,
        #     such that 'A' hashes to 'G' and 'C' hashes to 'T',
        #     so that similar guide sequences hash to the same
        #     value tolerating G-U similarity).
        #  2) Instead of taking a consensus sequence across a
        #     cluster, this can take a pseudo-consensus that
        #     accounts for G-U pairing (e.g., in which 'A' is
        #     treated as 'G' and 'C' is treated as 'T' for
        #     the purposes of generating a consensus sequence).

        assert start + guide_length <= self.seq_length
        assert len(seqs_to_consider) > 0

        for pos in range(start, start + guide_length):
            if self.frac_missing_at_pos(pos) > missing_threshold:
                raise CannotConstructGuideError(("Too much missing data at "
                    "a position in the target range"))

        aln_for_guide = self.extract_range(start, start + guide_length)

        if predictor is not None:
            # Extract the target sequences, including context to use with
            # prediction
            if (start - predictor.context_nt < 0 or
                    start + guide_length + predictor.context_nt > self.seq_length):
                raise CannotConstructGuideError(("The context needed for "
                    "the target to predict activity falls outside the "
                    "range of the alignment at this position"))
            aln_for_guide_with_context = self.extract_range(
                    start - predictor.context_nt,
                    start + guide_length + predictor.context_nt)

        # Before modifying seqs_to_consider, make a copy of it
        seqs_to_consider_cp = {}
        for group_id in seqs_to_consider.keys():
            seqs_to_consider_cp[group_id] = set(seqs_to_consider[group_id])
        seqs_to_consider = seqs_to_consider_cp

        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Ignore any sequences in the alignment that have a gap in
        # this region
        if predictor is not None:
            seqs_with_gap = set(aln_for_guide_with_context.seqs_with_gap(all_seqs_to_consider))
        else:
            seqs_with_gap = set(aln_for_guide.seqs_with_gap(all_seqs_to_consider))
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].difference_update(seqs_with_gap)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Only consider sequences in the alignment that have the
        # required flanking sequence(s)
        seqs_with_flanking = self.seqs_with_required_flanking(
            start, guide_length, required_flanking_seqs,
            seqs_to_consider=all_seqs_to_consider)
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].intersection_update(seqs_with_flanking)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # If every sequence in this region has a gap or does not contain
        # required flanking sequence, then there are none left to consider
        if len(all_seqs_to_consider) == 0:
            raise CannotConstructGuideError(("All sequences in region have "
                "a gap and/or do not contain required flanking sequences"))

        seq_rows = aln_for_guide.make_list_of_seqs(all_seqs_to_consider,
            include_idx=True)
        if predictor is not None:
            seq_rows_with_context = aln_for_guide_with_context.make_list_of_seqs(
                    all_seqs_to_consider, include_idx=True)

        if predictor is not None:
            # Memoize activity evaluations
            pair_eval = {}
        def determine_binding_and_active_seqs(gd_sequence):
            binding_seqs = []
            num_bound = 0
            if predictor is not None:
                num_passed_predict_active = 0
                # Determine what calls to make to predictor.evaluate(); it
                # is best to batch these
                pairs_to_eval = []
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if guide.guide_binds(gd_sequence, seq, mismatches,
                            allow_gu_pairs):
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        pairs_to_eval += [pair]
                # Evaluate activity
                evals = predictor.evaluate(start, pairs_to_eval)
                for pair, y in zip(pairs_to_eval, evals):
                    pair_eval[pair] = y
                # Fill in binding_seqs
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if guide.guide_binds(gd_sequence, seq, mismatches,
                            allow_gu_pairs):
                        num_bound += 1
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        if pair_eval[pair]:
                            num_passed_predict_active += 1
                            binding_seqs += [seq_idx]
            else:
                num_passed_predict_active = None
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if guide.guide_binds(gd_sequence, seq, mismatches,
                            allow_gu_pairs):
                        num_bound += 1
                        binding_seqs += [seq_idx]
            return binding_seqs, num_bound, num_passed_predict_active

        # Define a score function (higher is better) for a collection of
        # sequences covered by a guide
        if num_needed is not None:
            # This is the number of sequences it contains that are needed to
            # achieve the partial cover; we can compute this by summing over
            # the number of needed sequences it contains, taken across the
            # groups in the universe
            # Memoize the scores because this computation might be expensive
            seq_idxs_scores = {}
            def seq_idxs_score(seq_idxs):
                seq_idxs = set(seq_idxs)
                tc = tuple(seq_idxs)
                if tc in seq_idxs_scores:
                    return seq_idxs_scores[tc]
                score = 0
                for group_id, needed in num_needed.items():
                    contained_in_seq_idxs = seq_idxs & seqs_to_consider[group_id]
                    score += min(needed, len(contained_in_seq_idxs))
                seq_idxs_scores[tc] = score
                return score
        else:
            # Score by the number of sequences it contains
            def seq_idxs_score(seq_idxs):
                return len(seq_idxs)

        # First construct the optimal guide to cover the sequences. This would be
        # a string x that maximizes the number of sequences s_i such that x and
        # s_i are equal to within 'mismatches' mismatches; it's called the "max
        # close string" or "close to most strings" problem. For simplicity, let's
        # do the following: cluster the sequences (just the portion with
        # potential guides) with LSH, choose a guide for each cluster to be
        # the consensus of that cluster, and choose the one (across clusters)
        # that has the highest score
        if guide_clusterer is None:
            # Don't cluster; instead, draw the consensus from all the
            # sequences. Effectively, treat it as there being just one
            # cluster, which consists of all sequences to consider
            clusters_ordered = [all_seqs_to_consider]
        else:
            # Cluster the sequences
            clusters = guide_clusterer.cluster(seq_rows)

            # Sort the clusters by score, from highest to lowest
            # Here, the score is determined by the sequences in the cluster
            clusters_ordered = sorted(clusters, key=seq_idxs_score, reverse=True)

        best_gd = None
        best_gd_binding_seqs = None
        best_gd_score = 0
        stopped_early = False
        for cluster_idxs in clusters_ordered:
            if stop_early and best_gd_score > seq_idxs_score(cluster_idxs):
                # The guide from this cluster is unlikely to exceed the current
                # score; stop early
                stopped_early = True
                break

            gd = aln_for_guide.determine_consensus_sequence(
                cluster_idxs)
            if guide_is_suitable_fn is not None:
                if guide_is_suitable_fn(gd) is False:
                    # Skip this cluster
                    continue
            if 'N' in gd:
                # Skip this; all sequences at a position in this cluster
                # are 'N'
                continue
            # Determine the sequences that are bound by this guide (and
            # where it is 'active', if predictor is set)
            binding_seqs, num_bound, num_passed_predict_active = \
                    determine_binding_and_active_seqs(gd)
            score = seq_idxs_score(binding_seqs)
            if score > best_gd_score:
                best_gd = gd
                best_gd_binding_seqs = binding_seqs
                best_gd_score = score

            # Impose an early stopping criterion if predictor is
            # used, because using it is slow
            if predictor is not None and stop_early:
                if (num_bound >= 0.5*len(cluster_idxs) and
                        num_passed_predict_active < 0.1*len(cluster_idxs)):
                    # gd binds (according to guide.guide_binds()) many
                    # sequences, but is not predicted to be active against
                    # many; it is likely that this region is poor according to
                    # the predictor (e.g., due to sequence composition). Rather
                    # than trying other clusters at this site, just skip it.
                    # Note that this is just a heuristic; it can help runtime
                    # when the predictor is used, but is not needed and may
                    # hurt the optimality of the solution
                    stopped_early = True
                    break
        gd = best_gd
        binding_seqs = best_gd_binding_seqs

        # It's possible that the consensus sequence (guide) of no cluster
        # binds to any of the sequences. In this case, simply go through all
        # sequences and find the first that has no ambiguity and is suitable
        # and active, and make this the guide
        if gd is None and not stopped_early:
            for i, (s, idx) in enumerate(seq_rows):
                if sum(s.count(c) for c in ['A', 'T', 'C', 'G']) != len(s):
                    # s has ambiguity; skip it
                    continue
                if (guide_is_suitable_fn is not None and
                        guide_is_suitable_fn(s) is False):
                    # Skip s, which is not suitable
                    continue
                if predictor is not None:
                    s_with_context, _ = seq_rows_with_context[i]
                    if not predictor.evaluate(start, [(s_with_context, s)])[0]:
                        # s is not active against itself; skip it
                        continue
                # s has no ambiguity and is a suitable guide; use it
                gd = s
                binding_seqs, _, _ = determine_binding_and_active_seqs(gd)
                break

        if gd is None:
            raise CannotConstructGuideError(("No guides are suitable or "
                "active"))

        return (gd, binding_seqs)

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
            for b in [self.seqs[i][j] for j in seqs_to_consider]:
                if b in counts:
                    counts[b] += 1
                elif b == 'N':
                    # skip N
                    continue
                elif b in guide.FASTA_CODES:
                    for c in guide.FASTA_CODES[b]:
                        counts[c] += 1.0 / len(guide.FASTA_CODES[b])
                else:
                    raise ValueError("Unknown base call %s" % b)
            counts_sorted = sorted(list(counts.items()))
            max_base = max(counts_sorted, key=lambda x: x[1])[0]
            if counts[max_base] == 0:
                consensus += 'N'
            else:
                consensus += max_base

        return consensus

    def determine_most_common_sequence(self, seqs_to_consider=None,
            skip_ambiguity=False):
        """Determine the most common of the sequences from the alignment.

        If we imagine each sequence in the alignment as existing some
        number of times in the alignment, this effectively picks the
        mode.

        Ties are broken arbitrarily but deterministically.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if
                None, use all)
            skip_ambiguity: if True, ignore any sequences that contain
                an ambiguity code (i.e., only count sequences where all
                bases are 'A', 'T', 'C', or 'G')

        Returns:
            str representing the mode of the sequences (or None if there
            are no suitable strings)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        allowed_chars = ['A', 'T', 'C', 'G', '-']

        # Convert all sequences into strings, and count the number of each
        seqs_str = self.make_list_of_seqs(seqs_to_consider=seqs_to_consider)
        seq_count = defaultdict(int)
        for seq_str in seqs_str:
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

            seq_count[seq_str] += 1

        if len(seq_count) == 0:
            # There are no suitable strings (e.g., all contain ambiguity)
            return None

        # Find the most common sequence (sort to break ties deterministically)
        counts_sorted = sorted(list(seq_count.items()))
        max_seq_str = max(counts_sorted, key=lambda x: x[1])[0]
        return max_seq_str

    def sequences_bound_by_guide(self, gd_seq, gd_start, mismatches,
            allow_gu_pairs, required_flanking_seqs=(None, None)):
        """Determine the sequences to which a guide hybridizes.

        Args:
            gd_seq: seequence of the guide
            gd_start: start position of the guide in the alignment
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence
            allow_gu_pairs: if True, tolerate G-U base pairs between a
                guide and target when computing whether a guide binds
            required_flanking_seqs: tuple (s5, s3) that specifies sequences
                on the 5' (left; s5) end and 3' (right; s3) end flanking
                the guide (in the target, not the guide) that must be
                present for a guide to bind; if either is None, no
                flanking sequence is required for that end

        Returns:
            collection of indices of sequences to which the guide will
            hybridize
        """
        assert gd_start + len(gd_seq) <= self.seq_length

        aln_for_guide = self.extract_range(gd_start, gd_start + len(gd_seq))
        seq_rows = aln_for_guide.make_list_of_seqs(include_idx=True)

        seqs_with_flanking = self.seqs_with_required_flanking(
            gd_start, len(gd_seq), required_flanking_seqs)

        binding_seqs = []
        for seq, seq_idx in seq_rows:
            if (seq_idx in seqs_with_flanking and
                    guide.guide_binds(gd_seq, seq, mismatches, allow_gu_pairs)):
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

        return Alignment(seqs_col)


class CannotConstructGuideError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class SequenceClusterer:
    """Supports clustering sequences using LSH.

    This is useful for approximating an optimal guide over a collection of
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


class AlignmentQuerier:
    """Supports queries for potential guide sequences across a set of
    alignments.

    This uses LSH to find near neighbors of queried guide sequences. It
    constructs a data structure containing all subsequences in a given
    collection of alignments (subsequences of length equal to the guide
    length). These are sequences that could potentially be "hit" by a
    guide. Then, on query, it performs approximate near neighbor search
    and calculates the fraction of sequences in each alignment that are
    hit by the queried guide.

    Note that, when building the data structure, this removes all gaps
    from sequences in the alignment. Thus, all subsequences stored are
    gapless. In terms of querying for potential hits, this is more
    realistic because the actual sequences do not have gaps.

    This currently stores in the near-neighbor data structure a tuple
    (subsequence string s, index i of alignment with s) -- i.e., all
    unique subsequences for each alignment. It stores an additional data
    structure that contains, for each of those tuples, all of the
    sequences in alignment i that contain s. Together, these can be used
    to query for all sequences that contain a subsequence that is 'near'
    the queried guide.
    """

    def __init__(self, alns, guide_length, dist_thres, allow_gu_pairs, k=22,
                 reporting_prob=0.95):
        """
        Args:
            alns: list of Alignment objects
            guide_length: length of guide sequences
            dist_thres: detect a queried guide sequence as hitting a
                sequence in an alignment if it is within a distance
                of dist_thres (where the distance is measured with
                guide.seq_mismatches if G-U base pairing is disabled, and
                guide.seq_mismatches_with_gu_pairs is G-U base pairing
                is enabled; effectively these are Hamming distance either
                tolerating or not tolerating G-U base pairing)
            allow_gu_pairs: if True, tolerate G-U base pairs when computing
                whether a guide binds to a target
            k: number of hash functions to draw from a family of
                hash functions for amplification; each hash function is then
                the concatenation (h_1, h_2, ..., h_k)
            reporting_prob: ensure that any guide within dist_thres of
                a queried guide is detected as such with this probability;
                this constructs multiple hash functions (each of which is a
                concatenation of k functions drawn from the family) to achieve
                this probability
        """
        self.alns = alns
        self.guide_length = guide_length

        if allow_gu_pairs:
            # Measure distance while tolerating G-U base pairing, and
            # use an appropriate LSH family
            # Note that guide.seq_mismatches_with_gu_pairs is appropriate
            # for the distance function here (although it is asymmetric)
            # because it takes the arguments (guide_seq, target_seq), in
            # that order. This will work with near neighbor queries
            # because lsh.NearNeighborLookup.query(q) calls the dist_fn
            # as dist_fn(q, possible-hit); here, q always represents a
            # potential guide sequence and possible-hit is always a
            # potential target sequence
            dist_fn = guide.seq_mismatches_with_gu_pairs
            family = lsh.HammingWithGUPairsDistanceFamily(guide_length)
        else:
            # Do not tolerate G-U base pairing when measure distance,
            # and use a standard Hamming distance LSH family
            dist_fn = guide.seq_mismatches
            family = lsh.HammingDistanceFamily(guide_length)
        self.nnr = lsh.NearNeighborLookup(family, k, dist_thres, dist_fn,
            reporting_prob, hash_idx=0, join_concat_as_str=True)

        self.seqs_with_subseq = defaultdict(set)

        self.is_setup = False

    def setup(self):
        """Build data structure for near neighbor lookup of guide sequences.

        The data structure in self.nnr stores only unique subsequences in
        each alignment; it does not store the particular sequences of
        the alignment that contain the subsequences. A separate hashtable
        stored here, self.seqs_with_subseq, tracks the sequence indices
        in alignment i that contain a subsequence s (key'd on (s, i)).
        """
        for aln_idx, aln in enumerate(self.alns):
            # Convert aln.seqs from column-major order to row-major order
            # Also, remove all gaps from guide sequences because the
            # true sequences are gapless, so queries for potential hits
            # should be against gapless sequence
            seqs = aln.make_list_of_seqs(remove_gaps=True)

            for seq_idx, seq in enumerate(seqs):
                logger.debug(("Indexing for queries: alignment %d of %d, "
                    "sequence %d of %d"),
                    aln_idx + 1, len(self.alns), seq_idx + 1, len(seqs))

                # Add all possible guide sequences g as:
                #   (g, aln_idx)
                pts = set()
                for j in range(len(seq) - self.guide_length + 1):
                    g = seq[j:(j + self.guide_length)]
                    pts.add((g, aln_idx))
                    self.seqs_with_subseq[(g, aln_idx)].add(seq_idx)
                self.nnr.add(pts)
        self.is_setup = True

    def mask_aln(self, aln_idx):
        """Mask an alignment from being reported in near neighbor lookup.

        Note that masking an alignment A can be much more efficient than
        leaving it in the near neighbor data structures and filtering
        the output of queries that match A. This is especially true if
        many neighbors of queries will be from A because the near neighbor
        query function will spend a lot of time comparing the query against
        guides (neighbors) from A.

        Args:
            aln_idx: index of alignment in self.alns to mask from lookups
        """
        logger.debug("Masking alignment with index %d from alignment queries",
            aln_idx)

        # The alignment index is stored in index 1 of the tuple
        mask_idx = 1

        self.nnr.mask(mask_idx, aln_idx)

    def unmask_all_aln(self):
        """Unmask all alignments that may have been masked in the near neighbor
        lookup.
        """
        logger.debug("Unmasking all alignments from alignment queries")

        self.nnr.unmask_all()

    def frac_of_aln_hit_by_guide(self, guide):
        """Calculate how many sequences in each alignment are hit by a query.

        Args:
            guide: guide sequence to query

        Returns:
            list of fractions f where f[i] gives the fraction of sequences
            in the i'th alignment (self.alns[i]) that contain a subsequence
            which is found to be a near neighbor of guide
        """
        if not self.is_setup:
            raise Exception(("AlignmentQuerier.setup() must be called before "
                "querying for near neighbors"))

        assert len(guide) == self.guide_length

        neighbors = self.nnr.query(guide)
        seqs_hit_by_aln = defaultdict(set)
        for neighbor in neighbors:
            _, aln_idx = neighbor
            seq_idxs = self.seqs_with_subseq[neighbor]
            seqs_hit_by_aln[aln_idx].update(seq_idxs)

        frac_of_aln_hit = []
        for i, aln in enumerate(self.alns):
            num_hit = len(seqs_hit_by_aln[i])
            frac_hit = float(num_hit) / aln.num_sequences
            frac_of_aln_hit += [frac_hit]
        return frac_of_aln_hit

    def guide_is_specific_to_alns(self, guide, aln_idxs, frac_hit_thres):
        """Determine if guide is specific to a particular alignment.

        Note that this does *not* verify whether guide hits the alignment
        with an index in aln_idx -- only that it does not hit all the others.

        Args:
            guide: guide sequence to check
            aln_idxs: check if guide is specific to all the alignments with
                these indices (self.alns[aln_idxx[i]])
            frac_hit_thres: say that a guide "hits" an alignment A if the
                fraction of sequences in A that it hits is > this value

        Returns:
            True iff guide does not hit alignments other than aln_idx
        """
        frac_of_aln_hit = self.frac_of_aln_hit_by_guide(guide)
        for j, frac in enumerate(frac_of_aln_hit):
            if j in aln_idxs:
                # This frac is for alignment in aln_idxs; it should be high and
                # is irrelevant for specificity (but we won't check that
                # it is high)
                # Note that if aln_idxs[i] has already been masked, then this
                # check should not be necessary (j should never equal
                # aln_idxs[i])
                continue
            if frac > frac_hit_thres:
                # guide hits too many sequences in alignment j
                return False
        return True

