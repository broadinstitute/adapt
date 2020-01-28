"""Data structure for leaf node of tries.

The leaf nodes should support specificity queries.
"""

from adapt.utils import trie

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class KmerLeaf(trie.LeafInfo):
    """Store info in each leaf node on the k-mer leading to the leaf.

    This stores a dict, d: {taxonomy identifier: {sequence identifiers}}
    giving the taxonomies and sequences containing the k-mer.
    """
    def __init__(self, xs):
        """
        Args:
            xs: collection of tuples (a, b) where a gives taxonomy identifier
                and b gives sequence identifier for a sequence containing
                this k-mer
        """
        self.d = {}
        for x in xs:
            taxid, seqid = x
            if taxid not in self.d:
                self.d[taxid] = set()
            self.d[taxid].add(seqid)

    def extend(self, other):
        """Extend self to contain what is in other.

        Args:
            other: KmerLeaf object
        """
        for taxid, seqids in other.d.items():
            if taxid not in self.d:
                self.d[taxid] = set()
            self.d[taxid].update(seqids)

    def remove(self, taxid):
        """Remove taxonomy identifier from self.

        Args:
            taxid: taxonomy identifier
        """
        if taxid in self.d:
            del self.d[taxid]

    def is_empty(self):
        """Determine if self has any information stored.

        Returns:
            True iff self.d is empty
        """
        return len(self.d) == 0

    def __contains__(self, taxid):
        """Determine if a taxonomy identifier is stored by this leaf.

        Args:
            taxid: taxonomy identifier

        Returns:
            True iff taxid is in self.d
        """
        return taxid in self.d

    def copy(self):
        """Copy self.

        Returns:
            KmerLeaf object, identical to this one
        """
        d_new = {}
        for taxid in self.d.keys():
            d_new[taxid] = set(self.d[taxid])
        new = KmerLeaf({(-1,-1)})
        new.d = d_new
        return new

    def __eq__(self, other):
        """Check equality of self and other.

        Args:
            other: KmerLeaf object

        Returns:
            True or False
        """
        return self.d == other.d

    def __len__(self):
        """Sum number of sequences stored.

        Returns:
            total number of sequences stored across taxids
        """
        n = 0
        for taxid, seqids in self.d.items():
            n += len(seqids)
        return n

    @staticmethod
    def union(kmer_leaves):
        """Construct a KmerLeaf object representing a union of others.

        Args:
            kmer_leaves: collection of KmerLeaf objects

        Returns:
            KmerLeaf object
        """
        union = KmerLeaf([])
        for kl in kmer_leaves:
            union.extend(kl)
        return union

