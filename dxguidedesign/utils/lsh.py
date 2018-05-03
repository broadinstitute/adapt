"""Classes and methods for applying locality-sensitive hashing.

This code is modified from CATCH (https://github.com/broadinstitute/catch/blob/master/catch/utils/lsh.py),
which is released under the following license:
MIT License

Copyright (c) 2018 Hayden Metsky, Broad Institute, Inc., and Massachusetts
Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
import logging
import math
import random
import zlib

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class HammingDistanceFamily:
    """An LSH family that works with Hamming distance by sampling bases."""

    def __init__(self, dim):
        self.dim = dim

    def make_h(self):
        """Construct a random hash function for this family.

        Returns:
            hash function
        """
        i = random.randint(0, self.dim - 1)
        def h(x):
            assert len(x) == self.dim
            return x[i]
        return h

    def P1(self, dist):
        """Calculate lower bound on probability of collision for nearby sequences.

        Args:
            dist: Hamming distance; suppose two sequences are within this
                distance of each other

        Returns:
            lower bound on probability that two sequences (e.g., probes) hash
            to the same value if they are within dist of each other
        """
        return 1.0 - float(dist) / float(self.dim)


class MinHashFamily:
    """An LSH family that works by taking the minimum permutation of
    k-mers in a string/sequence (MinHash).

    See (Broder et al. 1997) and (Andoni and Indyk 2008) for details.
    """

    def __init__(self, kmer_size):
        self.kmer_size = kmer_size

    def make_h(self):
        """Construct a random hash function for this family.

        Here, we treat a sequence as being a set of k-mers. We calculate
        a hash value for each k-mer and the hash function on the sequence
        returns the minimum of these.

        Returns:
            hash function
        """
        # First construct a random hash function for a k-mer that
        # is a universal hash function (effectively a "permutation"
        # of the k-mer)
        # Constrain all values to be in [0, 2^31 - 1] to have a bound
        # on the output of the universal hash function; this upper bound
        # is nice because it is also a prime, so we can simply work
        # modulo (2^31 - 1)
        p = 2**31 - 1
        # Let the random hash function be:
        #   (a*x + b) mod p
        # for random integers a, b (a in [1, p] and b in [0, p])
        a = random.randint(1, p)
        b = random.randint(0, p)
        def kmer_hash(x):
            # Hash a k-mer x with the random hash function
            # hash(..) uses a random seed in Python 3.3+, so its output
            # varies across Python processes; use zlib.adler32(..) for a
            # deterministic hash value of the k-mer
            x_hash = zlib.adler32(x.encode('utf-8'))
            return (a * x_hash + b) % p

        def h(s):
            # For a string/sequence s, have the MinHash function be the minimum
            # hash over all the k-mers in it
            assert self.kmer_size <= len(s)
            if self.kmer_size >= len(s) / 2:
                logger.warning(("The k-mer size %d is large (> (1/2)x) "
                    "compared to the size of a sequence to hash (%d), which "
                    "might make it difficult for MinHash to find similar "
                    "sequence"), self.kmer_size, len(s))
            kmer_hashes = []
            for i in range(len(s) - self.kmer_size + 1):
                kmer = s[i:(i + self.kmer_size)]
                kmer_hashes += [kmer_hash(kmer)]
            return min(kmer_hashes)
        return h

    def P1(self, dist):
        """Calculate lower bound on probability of collision for nearby sequences.

        Args:
            dist: Jaccard distance (1 minus Jaccard similarity); suppose
                two sequences are within this distance of each other. The
                Jaccard similarity can be thought of as the overlap in k-mers
                between the two sequences

        Returns:
            lower bound on probability that two sequences (e.g., probes) hash
            to the same value if they are within dist of each other
        """
        # With MinHash, the collision probability is the Jaccard similarity
        return 1.0 - dist


class HashConcatenation:
    """Concatenated hash functions (AND constructions)."""

    def __init__(self, family, k, join_as_str=False):
        """
        Args:
            family: hash family object; must have a make_h() function
            k: number of hash functions to concatenate
            join_as_str: if True, concatenate the output of the k hash
                functions into a string before returning the concatenated
                result; if False (default), simply return a tuple of the
                k outputs
        """
        self.family = family
        self.k = k
        self.join_as_str = join_as_str
        self.hs = [family.make_h() for _ in range(k)]

    def g(self, x):
        """Evaluate random hash functions and concatenate the result.

        Args:
            x: point (e.g., probe) on which to evaluate hash functions

        Returns:
            concatenation of the result of the self.k random hash functions
            evaluated at x
        """
        if self.join_as_str:
            return ''.join(h(x) for h in self.hs)
        else:
            return tuple([h(x) for h in self.hs])


class NearNeighborLookup:
    """Support for approximate near neighbor lookups.

    This implements the R-near neighbor reporting problem described in
    Andoni and Indyk 2008.
    """

    def __init__(self, family, k, dist_thres, dist_fn, reporting_prob,
                 hash_idx=None, join_concat_as_str=False):
        """
        This selects a number of hash tables (defined as L in the above
        reference) according to the strategy it outlines: we want any
        neighbor (within dist_thres) of a query to be reported with
        probability at least reporting_prob; thus, the number of
        tables should be [log_{1 - (P1)^k} (1 - reporting_prob)]. In
        the above reference, delta is 1.0 - reporting_prob.

        Args:
            family: object giving family of hash functions
            k: number of hash functions from family to concatenate
            dist_thres: consider any two objects within this threshold
                of each other to be neighbors
            dist_fn: function f(a, b) that calculates the distance between
                a and b, to compare against dist_thres
            reporting_prob: report any neighbor of a query with
                probability at least equal to this
            hash_idx: if set, the inserted points are tuples and should
                be key'd on the hash_idx'd index; e.g., (A, B, C) might
                be a point and if hash_idx is 0, it is hashed only based on A,
                B and C simply store additional information along with A,
                and queries are based on distance to A
            join_concat_as_str: if True, have concatenated hash functions
                return a string rather than tuple (this can be more
                efficient, but works only if each hash function from the
                family returns a string)
        """
        self.family = family
        self.k = k
        self.dist_thres = dist_thres
        self.dist_fn = dist_fn
        self.hash_idx = hash_idx

        P1 = self.family.P1(dist_thres)
        if P1 == 1.0:
            # dist_thres might be 0, and any number of hash tables can
            # satisfy the reporting probability
            self.num_tables = 1
        else:
            self.num_tables = math.log(1.0 - reporting_prob, 1.0 - math.pow(P1, k))
            self.num_tables = int(math.ceil(self.num_tables))

        # Setup self.num_tables hash tables, each with a corresponding
        # function for hashing into it (the functions are concatenations
        # of k hash functions from the given family)
        self.hashtables = []
        self.hashtables_masked = []
        self.hashtables_g = []
        for j in range(self.num_tables):
            g = HashConcatenation(self.family, self.k,
                join_as_str=join_concat_as_str)
            self.hashtables += [defaultdict(set)]
            self.hashtables_masked += [defaultdict(set)]
            self.hashtables_g += [g]

    def add(self, pts):
        """Insert given points into each of the hash tables.

        Args:
            pts: collection of points (e.g., probes) to add to the hash
                tables
        """
        for j in range(self.num_tables):
            ht = self.hashtables[j]
            g = self.hashtables_g[j].g
            for p in pts:
                p_key = p[self.hash_idx] if self.hash_idx is not None else p
                ht[g(p_key)].add(p)

    def mask(self, mask_idx, mask_val):
        """Mask points from the hash tables that meet given criteria.

        The points stored must be tuples. This moves the points that meet
        the given criteria (according to mask_idx and mask_val) into separate
        hash tables, so that they can be returned to the main hash tables
        when they should be unmasked.

        Args:
            mask_idx: mask according to this index in a tuple specifying
                a point
            mask_val: mask all points p where p[mask_idx] == mask_val
        """
        for j in range(self.num_tables):
            ht = self.hashtables[j]
            ht_masked = self.hashtables_masked[j]
            keys_to_del = set()
            for k in ht.keys():
                # Find all points p in bucket with key k to mask
                p_to_mask = set()
                for p in ht[k]:
                    if p[mask_idx] == mask_val:
                        p_to_mask.add(p)

                for p in p_to_mask:
                    # Delete p from bucket with key k
                    ht[k].remove(p)
                    # Add p to the mask hash table
                    ht_masked[k].add(p)

                if len(ht[k]) == 0:
                    keys_to_del.add(k)
            # Delete empty buckets in ht
            for k in keys_to_del:
                del ht[k]

    def unmask_all(self):
        """Unmask all points that have been masked.

        This moves all points from the mask hash tables into the main
        hash tables.
        """
        for j in range(self.num_tables):
            ht = self.hashtables[j]
            ht_masked = self.hashtables_masked[j]
            for k in ht_masked.keys():
                for p in ht_masked[k]:
                    ht[k].add(p)

            # Reset this mask hash table
            self.hashtables_masked[j] = defaultdict(set)

    def query(self, q):
        """Find neighbors of a query point.

        Args:
            q: query point (e.g., probe); if self.hash_idx is set and
                the inserted points are tuples, q should only be the
                key of what to search for (i.e., distance is measured
                between q and p[self.hash_idx] for a stored point p)

        Returns:
            collection of stored points that are within self.dist_thres of
            q; all returned points are within this distance, but the
            returned points might not include all that are
        """
        neighbors = set()
        for j in range(self.num_tables):
            ht = self.hashtables[j]
            g = self.hashtables_g[j].g
            for p in ht[g(q)]:
                p_key = p[self.hash_idx] if self.hash_idx is not None else p
                if self.dist_fn(q, p_key) <= self.dist_thres:
                    neighbors.add(p)
        return neighbors

