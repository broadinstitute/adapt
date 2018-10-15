"""Structure(s) and functions for working with alignments of sequences.
"""

from collections import defaultdict
import logging
import statistics

from dxguidedesign.utils import guide
from dxguidedesign.utils import lsh

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
            list of indices of sequences that contain a gap
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        has_gap = set()
        for j in range(self.seq_length):
            has_gap.update(i for i in seqs_to_consider if self.seqs[j][i] == '-')
        return has_gap

    def construct_guide(self, start, guide_length, seqs_to_consider, mismatches,
            guide_clusterer, num_needed=None, missing_threshold=1,
            guide_is_suitable_fn=None):
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

        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Ignore any sequences in the alignment that have a gap in
        # this region
        seqs_to_ignore = set(aln_for_guide.seqs_with_gap(all_seqs_to_consider))
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].difference_update(seqs_to_ignore)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # If every sequence in this region has a gap, then there are none
        # left to consider
        if len(all_seqs_to_consider) == 0:
            raise CannotConstructGuideError("All sequences in region have a gap")

        seq_rows = aln_for_guide.make_list_of_seqs(all_seqs_to_consider,
            include_idx=True)

        # First construct the optimal guide to cover the sequences. This would be
        # a string x that maximizes the number of sequences s_i such that x and
        # s_i are equal to within 'mismatches' mismatches; it's called the "max
        # close string" or "close to most strings" problem. For simplicity, let's
        # do the following: cluster the sequences (just the portion with
        # potential guides) with LSH, pick a cluster (the "best" cluster)
        # according to some heuristic, and take the consensus of that cluster.
        # In particular, we'll do this by sorting the clusters (from best
        # to worst) and taking the consensus until it yields a suitable
        # guide.
        if guide_clusterer is None:
            # Don't cluster; instead, draw the consensus from all the
            # sequences. Effectively, treat it as there being just one
            # cluster, which consists of all sequences to consider
            clusters_ordered = [all_seqs_to_consider]
        else:
            # Cluster and pick the cluster that contains the highest number
            # of needed sequences to achieve the partial cover
            # Cluster the sequences
            clusters = guide_clusterer.cluster(seq_rows)

            # Define a score function for each cluster
            if num_needed is not None:
                # Score a cluster (higher is better) by the number of
                # sequences it contains that are needed to achieve the
                # partial cover; do this by summing over the number
                # of needed sequences it contains, taken across the
                # groups in the universe
                # Memoize the scores because this computation might be
                # expensive
                cluster_scores = {}
                def cluster_score(cluster_idxs):
                    tc = tuple(cluster_idxs)
                    if tc in cluster_scores:
                        return cluster_scores[tc]
                    score = 0
                    for group_id, needed in num_needed.items():
                        contained_in_cluster = cluster_idxs & seqs_to_consider[group_id]
                        score += min(needed, len(contained_in_cluster))
                    cluster_scores[tc] = score
                    return score
            else:
                # Score a cluster by the number of sequences it contains
                def cluster_score(cluster_idxs):
                    return len(cluster_idxs)

            # Sort the clusters by score, from highest to lowest
            clusters_ordered = sorted(clusters, key=cluster_score, reverse=True)

        # Create a guide from each cluster, until one is suitable
        selected_cluster_idxs = None
        if guide_is_suitable_fn is None:
            # Create a guide from the best cluster (first in the list)
            selected_cluster_idxs = clusters_ordered[0]
            consensus = aln_for_guide.determine_consensus_sequence(
                selected_cluster_idxs)
            gd = consensus
        else:
            gd = None
            for cluster_idxs in clusters_ordered:
                consensus = aln_for_guide.determine_consensus_sequence(
                    cluster_idxs)
                if guide_is_suitable_fn(consensus) is True:
                    gd = consensus
                    selected_cluster_idxs = cluster_idxs
                    break
            if gd is None:
                raise CannotConstructGuideError("No guides are suitable")

        # If all that exists at a position in the alignment is 'N', then do
        # not attempt to cover the sequences because we do not know which
        # base to put in the guide at that position. In this case, the
        # consensus will have 'N' at that position.
        if 'N' in gd:
            raise CannotConstructGuideError("A position has all 'N'")

        def determine_binding_seqs(gd_sequence):
            binding_seqs = []
            for seq, seq_idx in seq_rows:
                if guide.guide_binds(gd_sequence, seq, mismatches):
                    binding_seqs += [seq_idx]
            return binding_seqs

        binding_seqs = determine_binding_seqs(gd)

        # It's possible that the consensus sequence (guide) of a cluster does
        # not bind to any of the sequences. In this case, simply select the first
        # sequence from the selected cluster that has no ambiguity and make this
        # the guide; this is guaranteed to have at least one binding sequence
        # (itself)
        if len(binding_seqs) == 0:
            suitable_guides = []
            for s, idx in seq_rows:
                if sum(s.count(c) for c in ['A', 'T', 'C', 'G']) == len(s):
                    # s has no ambiguity and is a suitable guide
                    if idx in selected_cluster_idxs:
                        # Pick s as the guide
                        gd = s
                        binding_seqs = determine_binding_seqs(gd)
                        break
                    else:
                        suitable_guides += [s]
            if len(binding_seqs) == 0:
                # All sequences in the selected cluster have ambiguity, so now
                # simply pick any suitable guide
                if len(suitable_guides) > 0:
                    gd = suitable_guides[0]
                    binding_seqs = determine_binding_seqs(gd)

            # If it made it here, then all of the sequences have ambiguity
            # (so none are suitable guides); gd will remain the consensus and
            # binding_seqs will still be empty

        return (gd, binding_seqs)

    def make_list_of_seqs(self, seqs_to_consider=None, include_idx=False):
        """Construct list of sequences from the alignment.

        Args:
            seqs_to_consider: collection of indices of sequences to use (if None,
                use all)
            include_idx: instead of a list of str giving the sequences in
                the alignment, return a list of tuples (seq, idx) where seq
                is a str giving a sequence and idx is the index in the
                alignment

        Returns:
            list of str giving the sequences in the alignment (or, list of
            tuples if include_idx is True)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)
        if include_idx:
            return [(''.join(self.seqs[j][i] for j in range(self.seq_length)), i)
                    for i in seqs_to_consider]
        else:
            return [''.join(self.seqs[j][i] for j in range(self.seq_length))
                    for i in seqs_to_consider]

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

    def sequences_bound_by_guide(self, gd_seq, gd_start, mismatches):
        """Determine the sequences to which a guide hybridizes.

        Args:
            gd_seq: seequence of the guide
            gd_start: start position of the guide in the alignment
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence

        Returns:
            collection of indices of sequences to which the guide will
            hybridize
        """
        assert gd_start + len(gd_seq) <= self.seq_length

        aln_for_guide = self.extract_range(gd_start, gd_start + len(gd_seq))
        seq_rows = aln_for_guide.make_list_of_seqs(include_idx=True)

        binding_seqs = []
        for seq, seq_idx in seq_rows:
            if guide.guide_binds(gd_seq, seq, mismatches):
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

    This might use considerable memory, depending on the guide length,
    reporting probability, etc. It currently stores in the data structure
    a tuple (subsequence string s, index i of alignment with s, index j of
    sequence in alignment i that has subsequence s). One option to reduce
    memory usage would be to still key on the subsequence string, but
    not store it in the tuple and, in its place, store the index x of the
    subsequence in the j'th sequence of alignment i, so that the subsequence
    can be found as: self.alns[i].seqs[x:(x + guide_length)][j].
    """

    def __init__(self, alns, guide_length, dist_thres, k=15,
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

        if guide.get_allow_gu_pairs():
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

        self.is_setup = False

    def setup(self):
        """Build data structure for near neighbor lookup of guide sequences.
        """
        for aln_idx, aln in enumerate(self.alns):
            # Convert aln.seqs from column-major order to row-major order
            seqs = aln.make_list_of_seqs()

            for seq_idx, seq in enumerate(seqs):
                # Add all possible guide sequences g as:
                #   (g, aln_idx, seq_idx)
                pts = []
                for j in range(aln.seq_length - self.guide_length + 1):
                    g = seq[j:(j + self.guide_length)]
                    pts += [(g, aln_idx, seq_idx)]
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
        # The alignment index is stored in index 1 of the tuple
        mask_idx = 1

        self.nnr.mask(mask_idx, aln_idx)

    def unmask_all_aln(self):
        """Unmask all alignments that may have been masked in the near neighbor
        lookup.
        """
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
            _, aln_idx, seq_idx = neighbor
            seqs_hit_by_aln[aln_idx].add(seq_idx)

        frac_of_aln_hit = []
        for i, aln in enumerate(self.alns):
            num_hit = len(seqs_hit_by_aln[i])
            frac_hit = float(num_hit) / aln.num_sequences
            frac_of_aln_hit += [frac_hit]
        return frac_of_aln_hit

    def guide_is_specific_to_aln(self, guide, aln_idx, frac_hit_thres):
        """Determine if guide is specific to a particular alignment.

        Note that this does *not* verify whether guide hits the alignment
        with index aln_idx -- only that it does not hit all the others.

        Args:
            guide: guide sequence to check
            aln_idx: check if guide is specific to the alignment with this
                index (self.alns[aln_dx])
            frac_hit_thres: say that a guide "hits" an alignment A if the
                fraction of sequences in A that it hits is > this value

        Returns:
            True iff guide does not hit alignments other than aln_idx
        """
        frac_of_aln_hit = self.frac_of_aln_hit_by_guide(guide)
        for j, frac in enumerate(frac_of_aln_hit):
            if aln_idx == j:
                # This frac is for alignment aln_idx; it should be high and
                # is irrelevant for specificity (but we won't check that
                # it is high)
                # Note that if aln_idx has already been masked, then this
                # check should not be necessary (j should never equal aln_idx)
                continue
            if frac > frac_hit_thres:
                # guide hits too many sequences in alignment j
                return False
        return True

