"""Classes for querying the specificity of guides against target sequences.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging

from adapt.utils import guide
from adapt.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class AlignmentQuerier(metaclass=ABCMeta):
    """Abstract class supporting queries for potential guide sequences.
    """

    def __init__(self, alns, guide_length, dist_thres, allow_gu_pairs):
        """
        Args:
            alns: list of Alignment objects
            guide_length: length of guide sequences
            dist_thres: detect a queried guide sequence as hitting a
                sequence in an alignment if it is within a distance
                of dist_thres (where the distance is measured with
                guide.seq_mismatches if G-U base pairing is disabled, and
                guide.seq_mismatches_with_gu_pairs if G-U base pairing
                is enabled; effectively these are Hamming distance either
                tolerating or not tolerating G-U base pairing)
            allow_gu_pairs: if True, tolerate G-U base pairs when computing
                whether a guide binds to a target -- i.e., be sensitive to
                G-U pairing when computing mismatches or, in other words,
                do not count a G-U pair as a mismatch
        """
        self.alns = alns
        self.guide_length = guide_length
        self.dist_thres = dist_thres
        self.allow_gu_pairs = allow_gu_pairs

    @abstractmethod
    def setup(self): raise NotImplementedError

    @abstractmethod
    def mask_aln(self, aln_idx): raise NotImplementedError

    @abstractmethod
    def unmask_all_aln(self): raise NotImplementedError

    @abstractmethod
    def frac_of_aln_hit_by_guide(self, guide): raise NotImplementedError

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


class AlignmentQuerierWithLSHNearNeighbor(AlignmentQuerier):
    """Supports queries for potential guide sequences across a set of
    alignments using near-neighbor lookups with LSH.

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
            [See AlignmentQuerier.__init__()]
            k: number of hash functions to draw from a family of
                hash functions for amplification; each hash function is then
                the concatenation (h_1, h_2, ..., h_k)
            reporting_prob: ensure that any guide within dist_thres of
                a queried guide is detected as such with this probability;
                this constructs multiple hash functions (each of which is a
                concatenation of k functions drawn from the family) to achieve
                this probability
        """
        super().__init__(alns, guide_length, dist_thres, allow_gu_pairs)

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
