"""Classes and methods to shard (or split) k-mers across separate tries.

The basic idea is as follows. Without loss of generality, consider a 28-mer S
that we split into 2 parts.  S has two signatures: s_1 and s_2. s_1 is the
first 14-mer of S collapsed down to a 2-letter alphabet (to account for G-U
pairing). s_2 is the second 14-mer of S collapsed down to a 2-letter alphabet
(to account for G-U pairing). We have a separate trie for every possible
signature, and S gets stored twice: once for s_1 and again for s_2. Now
consider a 28-mer query Q, which similarly has signatures q_1 and q_2. Let's
only query for Q up to 3 mismatches. By the pigeonhole principle, either
Q[0:14] or Q[14:28] has <= 1 mismatch with any possible result. So we can query
in 30 tries (corresponding to 30 signatures): q_1, as well as q_1 where at
every position we flip the letter (i.e., introduce a mismatch); and the same
for q_2. On average, each trie should have 1/2^14 of the total number of
28-mers, so the total space of 28-mers we query is 30/2^14 = 0.002.

In general, consider p partitions (above, p=2) of a k-mer, where p is a small
constant. A loose bound on the query runtime in a single trie is O(# k-mers
in the trie), owing to considerable branching due to G-U pairing. Let n be the
total number of k-mers in the data structure; each trie has on average
n/2^{k/p} k-mers. So a loose bound on the query runtime is
O(p*(n/2^{k/p})*(sum_{i=0}^{floor(m/p)} {k/p choose i})), where m is the number
of mismatches we query up to.
"""

from abc import ABCMeta, abstractmethod
from itertools import combinations

from adapt.specificity import kmer_leaf
from adapt.utils import trie

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TrieSpace:
    """A space of tries, where each trie is addressed by a signature.
    """

    def __init__(self):
        self.tries = {}

    def get(self, sig, make=True):
        """Fetch a trie with a given signature.
        Args:
            sig: signature (e.g., string)
            make: if True, make a trie if one does not exist with sig
        Returns:
            Trie object (or, if one does not exist and make is False, None)
        """
        if sig not in self.tries:
            if make:
                # Make an empty trie, and insert it
                t = trie.Trie()
                self.tries[sig] = t
            else:
                return None
        return self.tries[sig]

    def all_tries(self):
        """Iterate over all tries.
        Yields:
            (signature, trie)
        """
        return self.tries.items()

    def mask(self, d):
        """Mask an object from all leaves in all tries.

        Args:
            d: object to remove from all leaves
        """
        for sig, t in self.all_tries():
            t.mask(d)

    def unmask_all(self):
        """Unmask all masked objects.
        """
        for sig, t in self.all_tries():
            t.unmask_all()


def _signatures_with_mismatches(sig, m):
    """Generate signatures within m mismatches given a signature.

    Args:
        sig: signature as string (alphabet is G and T)
        m: number of mismatches such that all signatures have 0,1,...,m
            mismatches relative to kmer

    Iterates:
        signatures as strings
    """
    yield sig
    for mi in range(1, m+1):
        # Introduce mi mismatches to sig
        for pos in combinations(range(len(sig)), mi):
            # Introduce mismatches to sig at positions in pos
            # Note that len(pos) == mi
            sig_mi = list(str(sig))
            for j in pos:
                # Introduce a mismatch at position j
                if sig_mi[j] == 'G':
                    sig_mi[j] = 'T'
                else:
                    sig_mi[j] = 'G'
            sig_mi_str = ''.join(sig_mi)
            yield sig_mi_str


def _full_signature(kmer):
    """Generate a 2-letter alphabet signature of a full k-mer.

    The transformations are A->G and C->T.

    Args:
        kmer: k-mer

    Returns:
        signature as a string
    """
    return kmer.replace('A', 'G').replace('C', 'T')


def _full_signatures_with_mismatches(kmer, m):
    """Generate signatures within m mismatches given a k-mer.

    Args:
        kmer: k-mer
        m: number of mismatches such that all signatures have 0,1,...,m
            mismatches relative to kmer

    Returns:
        list of signatures as strings
    """
    sig = _full_signature(kmer)
    return _signatures_with_mismatches(sig, m)


def _split_signature(kmer, half):
    """Generate a 2-letter alphabet signature of one half of a k-mer.

    The transformations are A->G and C->T.

    Args:
        kmer: k-mer
        half: 0 or 1, denoting the first or second half

    Returns:
        signature as a string
    """
    k = len(kmer)
    if k % 2 == 0:
        # Make each half be exactly half
        k_half = int(k / 2)
    else:
        # Make the first 'half' be 1 letter longer than the second
        k_half = int(k / 2) + 1
    assert half in [0, 1]
    kmer_half = kmer[0:k_half] if half == 0 else kmer[k_half:k]
    return kmer_half.replace('A', 'G').replace('C', 'T')


def _split_signatures_with_mismatches(kmer, half, m):
    """Generate signatures within one mismatch given a k-mer.

    Args:
        kmer: k-mer, where k is divisible by 2
        half: 0 or 1, denoting the first or second half
        m: number of mismatches such that all signatures have 0,1,...,m
            mismatches relative to kmer

    Returns:
        list of signatures as strings
    """
    sig = _split_signature(kmer, half)
    return _signatures_with_mismatches(sig, m)


class TrieSpaceOfKmers(metaclass=ABCMeta):
    """Abstract class for a space of tries storing k-mers, key'd by signatures
    computed from those k-mers.
    """

    @abstractmethod
    def add(self, kmers): raise NotImplementedError

    @abstractmethod
    def query(self, q, m, gu_pairing): raise NotImplementedError

    @abstractmethod
    def mask(self, taxid): raise NotImplementedError

    @abstractmethod
    def unmask_all(self): raise NotImplementedError


class TrieSpaceOfKmersFullSig(TrieSpaceOfKmers):
    """Space of tries storing k-mers, key'd by the full k-mer.
    """

    def __init__(self):
        self.ts = TrieSpace()
        self.kmer_len = None
        self.contains_kmers = False

    def add(self, kmers):
        """Add to space of tries.

        Args:
            kmers: iterator over (k-mer, {(taxonomy identifier, sequence id)});
                each k-mer must not have ambiguity
        """
        for kmer, seqs_with_kmer in kmers:
            if self.kmer_len is None:
                self.kmer_len = len(kmer)
            if len(kmer) != self.kmer_len:
                raise ValueError("Length of k-mer is variable")
            self.contains_kmers = True

            sig = _full_signature(kmer)
            leaf = kmer_leaf.KmerLeaf(seqs_with_kmer)
            self.ts.get(sig).insert([(kmer, leaf)])

    def query(self, q, m=0, gu_pairing=False):
        """Query a k-mer.

        Args:
            q: query string; must not have ambiguity
            m: number of mismatches to tolerate
            gu_pairing: whether to tolerate G-U base pairing

        Returns:
            dict {taxonomy identifier: {sequence identifiers}}
        """
        if self.contains_kmers is False:
            # Nothing was added; this prevents an error below when
            # self.kmer_len is None if nothing is added
            return {}
        if len(q) != self.kmer_len:
            raise ValueError("Query length must be %d" % self.kmer_len)

        all_results = []
        for sig in  _full_signatures_with_mismatches(q, m):
            t = self.ts.get(sig, make=False)
            if t is None:
                # No results
                pass
            else:
                results = t.query(q, mismatches=m, gu_pairing=gu_pairing)
                all_results.extend(results)
        leaf_union = kmer_leaf.KmerLeaf.union(all_results)
        return leaf_union.d

    def mask(self, taxid):
        """Mask taxid from data structure.

        Args:
            taxid: taxonomic identifier to mask
        """
        self.ts.mask(taxid)

    def unmask_all(self):
        """Unmask all taxonomic identifiers.
        """
        self.ts.unmask_all()


class TrieSpaceOfKmersSplitSig(TrieSpaceOfKmers):
    """Space of tries storing k-mers, key'd by each half of a k-mer.
    """

    def __init__(self):
        # TrieSpace containing first 1/2 of k-mer, forward direction
        self.ts_0 = TrieSpace()
        # TrieSpace containing second 1/2 of k-mer, reverse direction
        self.ts_1 = TrieSpace()

        self.kmer_len = None
        self.contains_kmers = False

    def add(self, kmers):
        """Add to space of tries.

        Args:
            kmers: iterator over (k-mer, {(taxonomy identifier, sequence id)})
                where the latter describes the sequences containing the k-mer;
                each k-mer must not have ambiguity
        """
        for kmer, seqs_with_kmer in kmers:
            if self.kmer_len is None:
                self.kmer_len = len(kmer)
            if len(kmer) != self.kmer_len:
                raise ValueError("Length of k-mer is variable")
            self.contains_kmers = True

            sig_0 = _split_signature(kmer, 0)
            sig_1 = _split_signature(kmer, 1)
            kmer_rev = kmer[::-1]

            leaf_0 = kmer_leaf.KmerLeaf(seqs_with_kmer)
            leaf_1 = kmer_leaf.KmerLeaf(seqs_with_kmer)
            self.ts_0.get(sig_0).insert([(kmer, leaf_0)])
            self.ts_1.get(sig_1).insert([(kmer_rev, leaf_1)])
    
    def query(self, q, m=0, gu_pairing=False):
        """Query a k-mer.

        Args:
            q: query string; must not have ambiguity
            m: number of mismatches to tolerate
            gu_pairing: whether to tolerate G-U base pairing

        Returns:
            dict {taxonomy identifier: {sequence identifiers}}
        """
        if self.contains_kmers is False:
            # Nothing was added; this prevents an error below when
            # self.kmer_len is None if nothing is added
            return {}
        if len(q) != self.kmer_len:
            raise ValueError("Query length must be %d" % self.kmer_len)

        # By the pigeonhole principle, there will be at most floor(m/2)
        # mismatches in one of the two halves of the query
        m_half = int(m / 2)

        # Determine the signatures of each half of the query to use

        if m_half == 0:
            sigs_0 = [_split_signature(q, 0)]
            sigs_1 = [_split_signature(q, 1)]
        else:
            sigs_0 = _split_signatures_with_mismatches(q, 0, m_half)
            sigs_1 = _split_signatures_with_mismatches(q, 1, m_half)

        q_rev = q[::-1]

        all_results = []

        # Query for tries maybe containing each half of the query
        for reverse, sigs, ts in [(False, sigs_0, self.ts_0), (True, sigs_1,
                self.ts_1)]:
            q_q = q_rev if reverse else q

            # Only allow up to m_half mismatches for the first
            # half of the search down the trie
            if self.kmer_len % 2 == 0:
                # The first half of the search down the trie contains
                # levels 0 through k/2 - 1 (inclusive), where k is
                # k-mer length
                first_half_level = int(self.kmer_len/2 - 1)
            else:
                # Since the k-mer length is odd, the first 'half' of
                # the k-mer is 1 letter longer than the second 'half'
                # of the k-mer
                if reverse:
                    # The first half of the search is intended to
                    # match the second 'half' of the k-mer; this
                    # is levels 0 through floor(k/2) - 1 (inclusive)
                    first_half_level = int(self.kmer_len/2) - 1
                else:
                    # The first half of the search is intended to
                    # match the first 'half' of the k-mer; this is
                    # levels 0 through floor(k/2) (inclusive)
                    first_half_level = int(self.kmer_len/2)

            # Lookup each signature, and query the resulting trie (if there
            # is one) for each
            for sig in sigs:
                t = ts.get(sig, make=False)
                if t is None:
                    # No results
                    pass
                else:
                    results = t.query(q_q,
                            mismatches=m, gu_pairing=gu_pairing,
                            mismatches_to_level=(m_half, first_half_level))
                    all_results.extend(results)

        leaf_union = kmer_leaf.KmerLeaf.union(all_results)
        return leaf_union.d

    def mask(self, taxid):
        """Mask taxid from data structure.

        Args:
            taxid: taxonomic identifier to mask
        """
        self.ts_0.mask(taxid)
        self.ts_1.mask(taxid)

    def unmask_all(self):
        """Unmask all taxonomic identifiers.
        """
        self.ts_0.unmask_all()
        self.ts_1.unmask_all()

