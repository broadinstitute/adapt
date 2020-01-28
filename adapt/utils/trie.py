"""Classes and methods for using a trie to query for similar DNA/RNA strings.

The trie here is intended to store relatively short strings (e.g., k-mers where
k is about 28). It is expected that many of the strings will be highly similar
to other strings (i.e., there are dense regions in the space of strings) --
for example, this can be the case if they are k-mers drawn from many genomes of
a diverse viral species will mutations throughout the genome. Also, the trie
should easily support insert/delete operations. For these reasons (in particular,
there being dense regions in the space of strings), the trie here does not
compress redundant edges as it typical in a compressed trie (i.e., there may be
internal nodes with one child). It is expected that compression would not have
much of a benefit because: (a) k-mers are similar but with mutations throughout;
(b) the depth of the trie is a small constant (e.g., 28); and (c) compression
would be insertions/deletions more tricky.
"""

from abc import ABCMeta, abstractmethod

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class LeafInfo(metaclass=ABCMeta):
    """Abstract base class of information stored at each leaf of a trie.
    """
    @abstractmethod
    def extend(self, other): raise NotImplementedError
    @abstractmethod
    def remove(self, x): raise NotImplementedError
    @abstractmethod
    def is_empty(self): raise NotImplementedError
    @abstractmethod
    def __contains__(self, x): raise NotImplementedError
    @abstractmethod
    def copy(self): raise NotImplementedError


class Node:
    """The root, an internal node, or leaf of the trie.

    Because there are many of these, it should use minimal space.
    """

    def __init__(self, char):
        """
        Args:
            char: character ('A', 'C', 'G', or 'T') that this node represents;
                or None, if this represents the root
        """
        assert char in (None, 'A', 'C', 'G', 'T')
        self.char = char

        # Store children as a list x where x[0], x[1], x[2], and
        # x[3] is a pointer to a child node corresponding to
        # 'A', 'C', 'G', and 'T' respectively (or None if there
        # is no child node under the corresponding character)
        # In other words, self.children gives the edges off of this node
        # and each edge represents 'A'/'C'/'G'/'T'
        self.children = [None, None, None, None]

        # If this is a leaf, store a pointer to an object giving
        # more information about the string; this object must be a
        # subclass of LeafInfo
        self.leaf_info = None

    def children_of(self, char, gu_pairing=True):
        """Determine children of this node for a query character.

        In the canonical case of DNA, this would only return one child
        corresponding to char (e.g., if char=='A', then self.children[0]).
        This still returns that child, but in the case of G-U base
        pairing there can be another child.
        An RNA guide with U can bind to target RNA with G, and vice-versa.
        The guide sequence that is synthesized to RNA is the reverse-complement
        of the guide sequence constructed here so that it will hybridize
        to the target. All guides constructed here are in the same
        sense as the target sequence. Therefore, for permitting G-U
        pairing, we consider a base X in the guide sequence as matching
        a base Y in the target if either: (X=='A' and Y=='G') or
        (X=='C' and Y=='T'). If X is either 'A' or 'C', this returns
        the child representing 'G' or 'T' respectively.

        Args:
            char: character ('A', 'C', 'G', or 'T') for which to
                find children that could bind to char
            gu_pairing: tolerate G-U pairing when determining whether a
                particular character may bind to char

        Returns:
            list of Node objects that match char
        """
        # Find the child below the edge that represents char
        # This should be faster than using a dict
        if char == 'A':
            char_child = self.children[0]
        elif char == 'C':
            char_child = self.children[1]
        elif char == 'G':
            char_child = self.children[2]
        elif char == 'T':
            char_child = self.children[3]
        else:
            raise ValueError(("Unknown char '%s'"), char)

        if char_child is not None:
            children = [char_child]
        else:
            children = []

        if gu_pairing:
            gu_child = None
            if char == 'A':
                # Matches 'G'
                gu_child = self.children[2]
            elif char == 'C':
                # Matches 'T'
                gu_child = self.children[3]
            if gu_child is not None:
                children += [gu_child]

        return children

    def all_children(self):
        """Determine all children of this node.

        Returns:
            list of Node objects that are children of this node
        """
        children = []
        for i in range(4):
            if self.children[i] is not None:
                children += [self.children[i]]
        return children

    def query(self, q, level, mismatches=0, gu_pairing=True,
            mismatches_used=0, mismatches_to_level=None):
        """Query a given string in the tree rooted at this node.

        Due to mismatches and G-U pairing, this can return multiple
        (or many) results for a query.

        Because the depth of the trie should be a small constant,
        recursion should be acceptable.

        Args:
            q: string to query for in the tree rooted at self
            level: current level of the trie (root is at level 0)
            mismatches: number of mismatches to permit when
                determining whether q matches a string stored
                in the trie rooted at self
            gu_pairing: tolerate G-U pairing when determining what
                strings match q
            mismatches_used: number of mismatches that have been used so far
            mismatches_to_level: if set, a tuple (m, l) such that only
                <= m mismatches are permitted at all levels <= l of the
                trie (the root is at level 0)

        Returns:
            list of x.leaf_info for all leaves x that match q
        """
        if self.leaf_info is not None:
            # This node is a leaf
            if len(q) > 0:
                # There is still a string to query, so the
                # leaf is not a valid result
                return []
            else:
                return [self.leaf_info]

        if len(q) == 0:
            # We are not yet at a leaf, but there is no string to
            # query
            return []

        if mismatches_to_level is not None:
            m, l = mismatches_to_level
            if level <= l and mismatches_used > m:
                # Too many mismatches have been used for what is allowed at
                # this level; return no results
                return []

        results = []

        # q[0] gives the first char of the string to query, and q[1:]
        # gives everything left to query after q[0]
        q_0 = q[0]
        q_remain = q[1:]

        # Find the children for the first char, and results under
        # these children
        matching_children = self.children_of(q_0, gu_pairing=gu_pairing)
        for child in matching_children:
            # When querying the remaining string, the number of
            # mismatches permitted stays the same because there
            # was a match for q_0
            results.extend(child.query(q_remain,
                level+1,
                mismatches=mismatches,
                gu_pairing=gu_pairing,
                mismatches_used=mismatches_used,
                mismatches_to_level=mismatches_to_level))

        if mismatches > 0:
            # Query for q_remain in children of this node, using
            # up a mismatch (i.e., under edges that do not match
            # q_0)
            for child in self.all_children():
                if child not in matching_children:
                    # This child is under a edge that represents a mismatch
                    # with q_0
                    results.extend(child.query(q_remain,
                        level+1,
                        mismatches=mismatches-1,
                        gu_pairing=gu_pairing,
                        mismatches_used=mismatches_used+1,
                        mismatches_to_level=mismatches_to_level))

        return results

    def insert(self, s, leaf_info, replace=False):
        """Insert a string into the trie rooted at this node.

        Args:
            s: string to insert
            leaf_info: object to store in the leaf; this must be a
                subclass of LeafInfo and implement its abstract methods
            replace: if True, instead of calling x.leaf_info.extend(leaf_info)
                for an existing leaf x, this replaces x.leaf_info with
                leaf_info
        """
        curr_node = self
        for char in s:
            # Determine the index of the edge that represents char
            # This should be faster than using a dict
            if char == 'A':
                char_idx = 0
            elif char == 'C':
                char_idx = 1
            elif char == 'G':
                char_idx = 2
            elif char == 'T':
                char_idx = 3
            else:
                raise ValueError(("Unknown char '%s' in '%s'"), char, s)

            child_node = curr_node.children[char_idx]
            if child_node is None:
                # A child does not exist under the edge representing char
                # Make it
                child_node = Node(char)
                curr_node.children[char_idx] = child_node
            curr_node = child_node

        # curr_node should represent the leaf node
        if curr_node.leaf_info is not None:
            # A string s was also present in the trie; the object
            # curr_node.leaf_info should support a merge with the
            # new leaf_info to be added (via extend())
            if replace:
                curr_node.leaf_info = leaf_info
            else:
                curr_node.leaf_info.extend(leaf_info)
        else:
            curr_node.leaf_info = leaf_info

    def remove(self, s):
        """Delete a string from the trie rooted at this node.

        Args:
            s: string to delete

        Returns:
            True/False indicating whether s is present in the trie
        """
        curr_node = self
        path = []
        for char in s:
            # Determine the index of the edge that represents char
            # This should be faster than using a dict
            if char == 'A':
                char_idx = 0
            elif char == 'C':
                char_idx = 1
            elif char == 'G':
                char_idx = 2
            elif char == 'T':
                char_idx = 3
            else:
                raise ValueError(("Unknown char '%s' in '%s'"), char, s)

            path += [(curr_node, char, char_idx)]

            child_node = curr_node.children[char_idx]
            if child_node is None:
                # A child does not exist under the edge representing char,
                # so s is not present in the trie
                return False
            curr_node = child_node

        # curr_node should represent the leaf node
        if curr_node.leaf_info is not None:
            # Delete the leaf
            curr_node.leaf_info = None

            # Step back up along the path, to delete nodes if they
            # have no children
            for node, c, c_idx in path[::-1]:
                # We branched off of node to the edge representing c
                # Delete the edge representing c
                node.children[c_idx] = None

                # See if node has other edges
                has_other_edges = False
                for i in range(4):
                    if i != c_idx and node.children[i] is not None:
                        # There is an edge representing i
                        has_other_edges = True
                        break
                if has_other_edges:
                    # Don't go further up the path, as node itself
                    # should not be deleted
                    break

            # s was present in the trie
            return True
        else:
            # curr_node is not a leaf, so s is not present in the trie
            return False

    def traverse_and_find(self, f):
        """Find strings whose leaves contain f for all leaves rooted at this node.

        Each object in a leaf x (x.leaf_info) should have a __contains__()
        method. This checks `f in x` for each leaf in the trie rooted at self.
        It returns the strings (constructed over the path to x) for each x
        for which f is in x.

        Args:
            f: object to check in each leaf_info

        Returns:
            collection of tuples (s, li) where s is a string leading to a leaf
            that contain f and li is the leaf_info of that leaf
        """
        children = self.all_children()

        if self.leaf_info is not None:
            # This is a leaf node
            assert len(children) == 0

            # Check if f is in this leaf
            if f in self.leaf_info:
                return [(self.char, self.leaf_info)]
            else:
                return []
        else:
            results = []
            for child in children:
                results.extend(child.traverse_and_find(f))

            # Prepend the char that this node represents to each
            # resulting string (unless this is the root)
            if self.char is not None:
                for i in range(len(results)):
                    s, li = results[i]
                    results[i] = (self.char + s, li)

            return results

    def traverse_and_remove(self, d):
        """Call x.leaf_info.remove(d) for all leaves x rooted at this node.

        Each object in a leaf x (x.leaf_info) should have a method
        x.remove(). This calls x.remove(d) for each leaf in the trie
        rooted at self. Furthermore, x should have a method x.is_empty(),
        and if x.is_empty() is true this removes the leaf node x and
        cleans up the path that leads to x (removing nodes along the way
        if they have no other edges).

        Args:
            d: object to remove from each leaf_info

        Returns:
            True iff this node should be deleted
        """
        children = self.all_children()

        if self.leaf_info is not None:
            # This is a leaf node
            assert len(children) == 0

            # Remove d (if needed)
            self.leaf_info.remove(d)

            if self.leaf_info.is_empty():
                # Remove this leaf and note that this node should be removed
                self.leaf_info = None
                return True
            else:
                # This leaf does not need to be removed
                return False
        else:
            for child in children:
                remove_child = child.traverse_and_remove(d)
                if remove_child:
                    # Remove the edge that points to child
                    for i in range(4):
                        if (self.children[i] is not None and
                                self.children[i].char == child.char):
                            self.children[i] = None

            # Check if there are any children left after removals
            if len(self.all_children()) == 0:
                # This node should be removed
                return True
            else:
                # This node still has >= 1 child, so it should stay
                return False

    def __len__(self):
        """Count the number of leaf nodes in the trie rooted at this node.

        Returns:
            number of leaf nodes
        """
        children = self.all_children()

        if self.leaf_info is not None:
            # This is a leaf node
            assert len(children) == 0
            return 1

        num_leaves = 0
        for child in children:
            num_leaves += len(child)
        return num_leaves


class Trie:
    """Trie, focused on storing k-mers for small k.

    See description at top of this module for why the trie is uncompressed.
    """

    def __init__(self):
        self.root_node = Node(None)
        self.masked = {}

    def insert(self, kvs):
        """Insert keys/values into the trie.

        Note that each insertion will take constant time if the trie is
        of a constant depth. If this were used to store a suffix tree
        (it should not be!) for a string of length N, then each insertion
        would take O(N) time and all the insertions would take O(N^2) time.

        Args:
            kvs: collection of (s, li) where s is a string to insert into
                the trie and li is a value to store in a leaf
        """
        for kv in kvs:
            s, li = kv
            self.root_node.insert(s, li)

    def query(self, q, mismatches=0, gu_pairing=True,
            mismatches_to_level=None):
        """Query a given string in the trie.

        Args:
            q: string to query for in the trie
            mismatches: number of mismatches to permit when
                determining whether q matches a string stored
                in the trie
            gu_pairing: tolerate G-U pairing when determining what
                strings match q
            mismatches_to_level: if set, a tuple (m, l) such that only
                <= m mismatches are permitted at all levels <= l of the trie
                (the root is at level 0)

        Returns:
            list of values corresponding to strings that match q
        """
        return self.root_node.query(q, 0, mismatches=mismatches,
                gu_pairing=gu_pairing, mismatches_used=0,
                mismatches_to_level=mismatches_to_level)

    def mask(self, d):
        """Mask an object from all leaves and cleanup trie.

        This calls x.leaf_info.remove(d) for all leaves x in the trie,
        and cleans up the trie (i.e., removing paths that do not lead
        to leaves).

        This additionally saves information about the masked objects, so
        that they can be unmasked (re-inserted into the trie) later.

        Args:
            d: object to remove from each leaf_info
        """
        # Find all (s, li) corresponding to leaves that contain d in their
        # leaf_info; li is the full leaf_info and may contain objects
        # that were not masked/removed but this is ok because they can
        # be re-inserted with a replacement operation
        masked_leaves = self.root_node.traverse_and_find(d)
        for s, li in masked_leaves:
            if s in self.masked:
                # If mask() is called successively, a more recent li could
                # replace an older one; instead, merge them via extend()
                self.masked[s].extend(li)
            else:
                # Make a copy of li because it will be modified below by
                # traverse_and_remove()
                self.masked[s] = li.copy()

        # Remove d from each leaf, and cleanup the trie
        self.root_node.traverse_and_remove(d)

    def unmask_all(self):
        """Unmask all masked objects.
        """
        for s, li in self.masked.items():
            # The leaf in the trie corresponding to string s, which had
            # leaf_info li, had objects that were masked; replace the
            # current one with li (an old copy)
            self.root_node.insert(s, li, replace=True)

        # Nothing is masked anymore
        self.masked = {}
