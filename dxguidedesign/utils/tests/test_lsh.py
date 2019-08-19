"""Tests for lsh module.

This code is modified from CATCH (https://github.com/broadinstitute/catch/blob/master/catch/utils/tests/test_lsh.py),
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

import random
import unittest

from adapt.utils import guide
from adapt.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestHammingDistanceFamily(unittest.TestCase):
    """Tests family of hash functions for Hamming distance.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.HammingDistanceFamily(20)

    def test_identical(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)

        # Identical strings should hash to the same value
        for i in range(30):
            h = self.family.make_h()
            self.assertEqual(h(a), h(b))

    def test_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'ATCGACATGGGCACTGGTAT'

        # a and b should probably collide
        collision_count = 0
        for i in range(10):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertGreater(collision_count, 8)

    def test_not_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'AGTTGTCACCCTTGACGATA'

        # a and b should rarely collide
        collision_count = 0
        for i in range(10):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertLess(collision_count, 2)

    def test_collision_prob(self):
        # Collision probability for 2 mismatches should be
        # 1 - 2/20
        self.assertEqual(self.family.P1(2), 0.9)


class TestHammingWithGUPairsDistanceFamily(unittest.TestCase):
    """Tests family of hash functions for Hamming distance after
    transformations that account for G-U pairing.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.HammingWithGUPairsDistanceFamily(20)

    def test_identical(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)

        # Identical strings should hash to the same value
        for i in range(30):
            h = self.family.make_h()
            self.assertEqual(h(a), h(b))

    def test_identical_gu(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'GCCAATGTGGACGTTACTGC'

        # Identical strings, after 'A'->'G' and 'C'->'T' transformations,
        # should hash to the same value
        for i in range(30):
            h = self.family.make_h()
            self.assertEqual(h(a), h(b))

    def test_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'GTCGGTATGAGCATCGGTGG'

        # a and b should probably collide
        collision_count = 0
        for i in range(10):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertGreater(collision_count, 8)

    def test_not_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'AGTTGTCACCCTTGACGATA'

        # a and b should rarely collide
        collision_count = 0
        for i in range(10):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertLess(collision_count, 2)

    def test_collision_prob(self):
        # Collision probability for 2 mismatches should be
        # 1 - 2/20
        self.assertEqual(self.family.P1(2), 0.9)


class TestMinHashFamilyWithSingleHash(unittest.TestCase):
    """Tests family of hash functions for MinHash, where the hash function
    only returns a single (minimum) value.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.MinHashFamily(3, N=1)

    def test_identical(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)

        # Identical strings should hash to the same value
        h1 = self.family.make_h()
        self.assertEqual(h1(a), h1(b))
        h2 = self.family.make_h()
        self.assertEqual(h2(a), h2(b))

    def test_similar(self):
        a = 'ATCGATATGGGCACTGCTATGTAGCGC'
        b = 'ATCGACATGGGCACTGGTATGTAGCGC'

        # a and b should probably collide; the Jaccard similarity
        # of a and b is ~67% (with 3-mers being the elements that
        # make up each sequence) so they should collide with that
        # probability (check that it is >60%)
        collision_count = 0
        for i in range(100):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertGreater(collision_count, 60)

    def test_not_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'AGTTGTCACCCTTGACGATA'

        # a and b should rarely collide
        collision_count = 0
        for i in range(100):
            h = self.family.make_h()
            if h(a) == h(b):
                collision_count += 1
        self.assertLess(collision_count, 30)

    def test_collision_prob(self):
        # Collision probability for two sequences with a Jaccard
        # distance of 0.2 should be 0.8
        self.assertEqual(self.family.P1(0.2), 0.8)


class TestMinHashFamilySignatures(unittest.TestCase):
    """Tests family of hash functions for MinHash, where the hash function
    returns a signature (multiple hash values).
    """
    
    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.MinHashFamily(4, N=10)

    def test_identical(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)

        # Identical strings should yield the same signature, for the same
        # hash function
        for i in range(10):
            h = self.family.make_h()
            self.assertEqual(h(a), h(b))
            self.assertEqual(self.family.estimate_jaccard_dist(h(a), h(b)), 0.0)

    def test_jaccard_dist_similar(self):
        a = 'ATCGATATGGGCACTGCTATGTAGCGCAAATACGATCGCTAATGCGGATCGGATCGAATG'
        b = 'ATCGACATGGGCACTGGTATGTAGCGCAAATACGATCGCTATTGCGGATCGGATCGAATG'

        # These strings are very similar, but since N is small
        # the Jaccard distance estimate may in some cases be a
        # significant overestimate; test that most of
        # the time, the distance is <=0.5
        num_close = 0
        for i in range(100):
            h = self.family.make_h()
            if self.family.estimate_jaccard_dist(h(a), h(b)) <= 0.5:
                num_close += 1
        self.assertGreaterEqual(num_close, 80)

    def test_jaccard_dist_not_similar(self):
        a = 'ATCGATATGGGCACTGCTATGTAGCGCAAATACGATCGCTAATGCGGATCGGATCGAATG'
        b = 'TCGATCGAATCGAAGGTCGATCGGCGCAATACGGATCGCATTCGATCGGTTATAACGTGA'

        # These strings are far apart, and the estimated Jaccard distance
        # should usually be high
        num_far = 0
        for i in range(100):
            h = self.family.make_h()
            if self.family.estimate_jaccard_dist(h(a), h(b)) > 0.5:
                num_far += 1
        self.assertGreaterEqual(num_far, 80)


class TestHammingHashConcatenation(unittest.TestCase):
    """Tests concatenations of hash functions with Hamming distance.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.HammingDistanceFamily(20)
        self.G = lsh.HashConcatenation(self.family, 100)
        self.G_join = lsh.HashConcatenation(self.family, 100, join_as_str=True)

    def test_identical(self):
        # Identical a and b should collide even with large k
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)
        self.assertEqual(self.G.g(a), self.G.g(b))
        self.assertEqual(self.G_join.g(a), self.G_join.g(b))

    def test_similar(self):
        # Similar (but not identical) a and b should rarely
        # collide when k is large
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'ATCGACATGGGCACTGGTAT'

        collision_count = 0
        collision_count_join = 0
        for i in range(10):
            if self.G.g(a) == self.G.g(b):
                collision_count += 1
            if self.G_join.g(a) == self.G_join.g(b):
                collision_count_join += 1
        self.assertLess(collision_count, 2)
        self.assertLess(collision_count_join, 2)

    def test_not_similar(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = 'AGTTGTCACCCTTGACGATA'

        # a and b should rarely collide
        collision_count = 0
        collision_count_join = 0
        for i in range(10):
            if self.G.g(a) == self.G.g(b):
                collision_count += 1
            if self.G_join.g(a) == self.G_join.g(b):
                collision_count_join += 1
        self.assertLess(collision_count, 2)
        self.assertLess(collision_count_join, 2)


class TestHammingNearNeighborLookup(unittest.TestCase):
    """Tests approximate near neighbor lookups with Hamming distance."""

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.HammingDistanceFamily(20)
        self.dist_thres = 5
        def f(a, b):
            assert len(a) == len(b)
            return sum(1 for i in range(len(a)) if a[i] != b[i])
        self.dist_fn = f

    def test_varied_k(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)  # identical to a
        c = 'ATCGACATGGGCACTGGTAT'  # similar to a
        d = 'AGTTGTCACCCTTGACGATA'  # not similar to a
        e = 'AGTTGTCACCCTTGACGATA'  # similar to d

        for k in [2, 5, 10]:
            nnl = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95)
            nnl.add([a, b, c, d])

            # b and c are within self.dist_thres of a, so only these
            # should be returned (along with a); note that since
            # a==b, {a,b,c}=={a,c}=={b,c} and nnl.query(a) returns
            # a set, which will be {a,c} or {b,c}
            self.assertCountEqual(nnl.query(a), {a, b, c})

            # Although e was not added, a query for it should return d
            self.assertCountEqual(nnl.query(e), {d})

    def test_masking(self):
        a = ('ATCGATATGGGCACTGCTAT', 1)
        b = (str(a[0]), 2)  # identical to a
        c = ('ATCGACATGGGCACTGGTAT', 3)  # similar to a
        d = ('AGTTGTCACCCTTGACGATA', 1)  # not similar to a
        e = ('AGTTGTCACCCTTGACGATA', 4)  # similar to d

        for k in [2, 5, 10]:
            nnl = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95, hash_idx=0)
            nnl.add([a, b, c, d])

            # As with above test (test_varied_key()), the following should
            # hold true:
            self.assertCountEqual(nnl.query(a[0]), {a, b, c})
            self.assertCountEqual(nnl.query(e[0]), {d})

            # Mask where the second index of the tuple (index 1) is 1
            nnl.mask(1, 1)

            # Now a and d should not be reported
            self.assertCountEqual(nnl.query(a[0]), {b, c})
            self.assertCountEqual(nnl.query(e[0]), {})

            # Unmask all
            nnl.unmask_all()

            # Now the same results as above (before masking) should hold
            self.assertCountEqual(nnl.query(a[0]), {a, b, c})
            self.assertCountEqual(nnl.query(e[0]), {d})


class TestHammingWithGUPairsNearNeighborLookup(unittest.TestCase):
    """Tests approximate near neighbor lookups with Hamming distance
    that supports G-U pairing."""

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.family = lsh.HammingWithGUPairsDistanceFamily(20)
        self.dist_thres = 5
        self.dist_fn = guide.seq_mismatches_with_gu_pairs

    def test_varied_k_without_gu_pairs(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)  # identical to a
        c = 'ATCGACATGGGCACTGGTAT'  # similar to a
        d = 'AGTTGTCACCCTTGACGATA'  # not similar to a
        e = 'AGTAGTCACCCTTGACGCTA'  # similar to d

        for k in [2, 5, 10]:
            nnl = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95)
            nnl.add([a, b, c, d])

            # b and c are within self.dist_thres of a, so only these
            # should be returned (along with a); note that since
            # a==b, {a,b,c}=={a,c}=={b,c} and nnl.query(a) returns
            # a set, which will be {a,c} or {b,c}
            self.assertCountEqual(nnl.query(a), {a, b, c})

            # Although e was not added, a query for it should return d
            self.assertCountEqual(nnl.query(e), {d})

    def test_varied_k_with_gu_pairs(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)  # identical to a
        c = 'ACCAACACAAACACCACCAC'  # identical to a in {G,T} space
        d = 'AGTTGTCACCCTTGACGATA'  # not similar to a
        e = 'AGTAGTCACCCTTGACGCTA'  # similar to d
        f = 'AACCACCACGCCCACCACGA'  # similar to d in {G,T} space
        g = 'GGGGGGGGGGTTTTTTTTTT'  # not similar to anything else
        h = 'AAAAAAAAAACCCCCCCCCC'  # similar to g in {G,T} space

        for k in [2, 5, 10]:
            nnl1 = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95)
            nnl1.add([a, b, c, d, g])

            # b is within self.dist_thres of a, so it should be in the
            # output
            # Although c is similar to a in {G,T} space, it is *not*
            # within self.dist_thres of a according to self.dist_fn(a, c),
            # which is what a query for a would compute, so c should not
            # be in the output
            # Note that since a==b, {a,b}=={a}=={b} and nnl.query(a) returns a
            # set, which will be {a} or {b}
            self.assertCountEqual(nnl1.query(a), {a, b})

            # c is within self.dist_thres of a according to self.dist_fn(c, a),
            # which is what a query for c would compute
            # Since a==b, it is also within self.dist_thres of b
            self.assertCountEqual(nnl1.query(c), {a, b, c})

            # Although e was not added, a query for it should return d because
            # they are so similar
            # f is similar to d according to self.dist_fn(f, d), so a
            # query for f should return d
            self.assertCountEqual(nnl1.query(e), {d})
            self.assertCountEqual(nnl1.query(f), {d})

            # h is similar to g according to self.dist_fn(h, g), so a
            # query for h should return g
            self.assertCountEqual(nnl1.query(h), {g})

            # Now make a few changes to what is included in the data
            # structure, and re-test
            nnl2 = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95)
            nnl2.add([a, b, c, d, e, f, h])

            # Since e is now included in the data structure, and f is
            # similar to d according to self.dist_fn(f, d) (as well as to
            # e), a query for f should return both; it should also
            # of course return f since f is included now too
            self.assertCountEqual(nnl2.query(f), {d, e, f})

            # d is *not* similar to f according to self.dist_fn(d, f), so
            # a query for d should not return f (but should return e,
            # and of course d, now that e is included in the data structure)
            self.assertCountEqual(nnl2.query(d), {d, e})

            # Although h is similar to g in {G,T} space, it is *not*
            # similar according to self.dist_fn(g, h), so a query for g
            # should not return h (note that, above, in nnl1, where g is
            # included but not h, a query for h should return g)
            self.assertCountEqual(nnl2.query(g), {})


class TestMinHashNearNeighborLookup(unittest.TestCase):
    """Tests approximate near neighbor lookups with MinHash."""

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        kmer_size = 3
        self.family = lsh.MinHashFamily(kmer_size, N=1)
        self.dist_thres = 0.5
        def f(a, b):
            a_kmers = [a[i:(i + kmer_size)] for i in range(len(a) - kmer_size + 1)]
            b_kmers = [b[i:(i + kmer_size)] for i in range(len(b) - kmer_size + 1)]
            a_kmers = set(a_kmers)
            b_kmers = set(b_kmers)
            jaccard_sim = float(len(a_kmers & b_kmers)) / len(a_kmers | b_kmers)
            return 1.0 - jaccard_sim
        self.dist_fn = f

    def test_varied_k(self):
        a = 'ATCGATATGGGCACTGCTAT'
        b = str(a)  # identical to a
        c = 'ATCGACATGGGCACTGGTAT'  # similar to a
        d = 'AGTTGTCACCCTTGACGATA'  # not similar to a
        e = 'AGTTGTCACCCTTGACGATA'  # similar to d

        for k in [2, 5, 10]:
            nnl = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95)
            nnl.add([a, b, c, d])

            # b and c are within self.dist_thres of a, so only these
            # should be returned (along with a); note that since
            # a==b, {a,b,c}=={a,c}=={b,c} and nnl.query(a) returns
            # a set, which will be {a,c} or {b,c}
            self.assertCountEqual(nnl.query(a), {a, b, c})

            # Although e was not added, a query for it should return d
            self.assertCountEqual(nnl.query(e), {d})

    def test_hash_idx(self):
        a = ('ATCGATATGGGCACTGCTAT', 'abc', 'def')
        b = tuple(a)  # identical to a
        c = ('ATCGACATGGGCACTGGTAT', 'ghi', 'jkl')  # similar to a
        d = ('AGTTGTCACCCTTGACGATA', 'mno', 'pqr')  # not similar to a
        e = ('AGTTGTCACCCTTGACGATA', 'stu', 'vwx')  # similar to d

        for k in [2, 5, 10]:
            nnl = lsh.NearNeighborLookup(self.family, k, self.dist_thres,
                self.dist_fn, 0.95, hash_idx=0)
            nnl.add([a, b, c, d])

            # b and c are within self.dist_thres of a, so only these
            # should be returned (along with a); note that since
            # a==b, {a,b,c}=={a,c}=={b,c} and nnl.query(a) returns
            # a set, which will be {a,c} or {b,c}
            self.assertCountEqual(nnl.query(a[0]), {a, b, c})

            # Although e was not added, a query for it should return d
            self.assertCountEqual(nnl.query(e[0]), {d})
