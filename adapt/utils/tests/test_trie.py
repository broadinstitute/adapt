"""Tests for trie module.
"""

import unittest

from adapt.utils import trie

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class LeafInfo(trie.LeafInfo):
    """Mock class (wrapping a list) to use for leaf_info.
    """
    def __init__(self, l=[]):
        self.l = l
    def extend(self, other):
        self.l.extend(other.l)
    def remove(self, x):
        if x in self.l:
            self.l.remove(x)
    def is_empty(self):
        return len(self.l) == 0
    def copy(self):
        return LeafInfo(list(self.l))
    def __contains__(self, x):
        return x in self.l
    def __eq__(self, other):
        return frozenset(tuple(self.l)) == frozenset(tuple(other.l))
    def __repr__(self):
        return str(self.l)
    def __hash__(self):
        return hash(frozenset(tuple(self.l)))


def all_3mers():
    r = []
    for c1 in ['A', 'C', 'G', 'T']:
        for c2 in ['A', 'C', 'G', 'T']:
            for c3 in ['A', 'C', 'G', 'T']:
                r += [c1 + c2 + c3]
    return r


class TestNode(unittest.TestCase):
    """Tests Node objects used in Trie.
    """


    def test_insert(self):
        r = trie.Node(None)
        r.insert('AC', LeafInfo([1]))
        r.insert('AC', LeafInfo([2]))
        r.insert('AT', LeafInfo([3]))
        r.insert('GG', LeafInfo([4]))

        # Root should only point to A and G edges (0 and 2)
        self.assertIsNone(r.leaf_info)
        self.assertIsNone(r.children[1])
        self.assertIsNone(r.children[3])

        child_A = r.children[0]
        child_G = r.children[2]

        # A should only point to C and T edges (1 and 3); G should only
        # point to G edge (2)
        self.assertIsNone(child_A.leaf_info)
        self.assertIsNone(child_G.leaf_info)
        self.assertIsNone(child_A.children[0])
        self.assertIsNone(child_A.children[2])
        self.assertIsNone(child_G.children[0])
        self.assertIsNone(child_G.children[1])
        self.assertIsNone(child_G.children[3])

        # Now check the leaves
        child_AC = child_A.children[1]
        child_AT = child_A.children[3]
        child_GG = child_G.children[2]
        self.assertEqual(child_AC.children, [None, None, None, None])
        self.assertEqual(child_AT.children, [None, None, None, None])
        self.assertEqual(child_GG.children, [None, None, None, None])
        self.assertEqual(child_AC.leaf_info, LeafInfo([1,2]))
        self.assertEqual(child_AT.leaf_info, LeafInfo([3]))
        self.assertEqual(child_GG.leaf_info, LeafInfo([4]))

    def test_all_children(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))

        root_children = r.all_children()
        self.assertCountEqual([c.char for c in root_children], ['A', 'G'])

        child_A_children = r.children[0].all_children()
        self.assertCountEqual([c.char for c in child_A_children], ['A', 'C'])

        child_G_children = r.children[2].all_children()
        self.assertCountEqual([c.char for c in child_G_children], ['G', 'T'])

    def test_children_of_without_gu(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))

        # Check for 'AA' and 'AC'
        children_A = r.children_of('A', gu_pairing=False)
        self.assertCountEqual([c.char for c in children_A], ['A'])
        children_AA = children_A[0].children_of('A', gu_pairing=False)
        self.assertCountEqual([c.char for c in children_AA], ['A'])
        children_AC = children_A[0].children_of('C', gu_pairing=False)
        self.assertCountEqual([c.char for c in children_AC], ['C'])

        # There is no 'AG'
        children_AG = children_A[0].children_of('G', gu_pairing=False)
        self.assertCountEqual(children_AG, [])

    def test_children_of_with_gu(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))

        # 'A' should match 'A' and 'G' with G-U pairing
        children_A = r.children_of('A', gu_pairing=True)
        self.assertCountEqual([c.char for c in children_A], ['A', 'G'])

    def test_len(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))
        r.insert('ATC', LeafInfo([8]))

        # 'ACC' only gets counted once
        self.assertEqual(len(r), 7)

    def test_query_no_mismatches_without_gu(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))

        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([3,6]),
                'ACG': LeafInfo([4]),
                'GTG': LeafInfo([5]),
                'TCC': LeafInfo([7])}
        for query in all_3mers():
            if query in true:
                expected = [true[query]]
            else:
                expected = []
            self.assertCountEqual(r.query(query, 0, 0, False), expected)

    def test_query_no_mismatches_without_gu_with_insert_replace(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]), replace=True)
        r.insert('GGG', LeafInfo([2]), replace=True)
        r.insert('ACC', LeafInfo([3]), replace=True)
        r.insert('ACG', LeafInfo([4]), replace=True)
        r.insert('GTG', LeafInfo([5]), replace=True)
        r.insert('ACC', LeafInfo([6]), replace=True)
        r.insert('TCC', LeafInfo([7]), replace=True)
        r.insert('GTG', LeafInfo([8]), replace=True)
        r.insert('ACG', LeafInfo([9]))

        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([6]),
                'ACG': LeafInfo([4,9]),
                'GTG': LeafInfo([8]),
                'TCC': LeafInfo([7])}
        for query in all_3mers():
            if query in true:
                expected = [true[query]]
            else:
                expected = []
            self.assertCountEqual(r.query(query, 0, 0, False), expected)

    def test_query_no_mismatches_with_gu(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))
        r.insert('ATC', LeafInfo([8]))

        # 'A' can match 'A' or 'G'
        self.assertCountEqual(r.query('AAA', 0, 0, True),
                [LeafInfo([1]), LeafInfo([2])])
        self.assertCountEqual(r.query('ATG', 0, 0, True),
                [LeafInfo([5])])

        # 'G' can only match 'G'
        self.assertCountEqual(r.query('GGG', 0, 0, True),
                [LeafInfo([2])])

        # 'C' can match 'T'
        self.assertCountEqual(r.query('CCC', 0, 0, True),
                [LeafInfo([7])])
        self.assertCountEqual(r.query('ACC', 0, 0, True),
                [LeafInfo([3,6]), LeafInfo([8])])

    def test_query_1_mismatch_without_gu(self):
        r = trie.Node(None)
        r.insert('AAAAAAA', LeafInfo([1]))
        r.insert('AAATAAA', LeafInfo([2]))
        r.insert('AAATTAA', LeafInfo([3]))
        r.insert('CCCCCCC', LeafInfo([4]))
        r.insert('CCCGCCC', LeafInfo([5]))
        r.insert('CCCGGCC', LeafInfo([6]))
        r.insert('GGGGGGG', LeafInfo([7]))
        r.insert('GGGTGGG', LeafInfo([8]))
        r.insert('GGGTTGG', LeafInfo([9]))
        r.insert('TTTTTTT', LeafInfo([10]))
        r.insert('TTTGTTT', LeafInfo([11]))
        r.insert('TTTGGTT', LeafInfo([12]))

        # Test above keys with mismatches
        expected = {'AAAAAAA': [LeafInfo([1]), LeafInfo([2])],
                    'AAAGAAA': [LeafInfo([1]), LeafInfo([2])],
                    'CAAAAAA': [LeafInfo([1])],
                    'ATATATA': [],
                    'CCCCCCC': [LeafInfo([4]), LeafInfo([5])],
                    'CCCACCC': [LeafInfo([4]), LeafInfo([5])],
                    'GCCCCCC': [LeafInfo([4])],
                    'GGGGGGG': [LeafInfo([7]), LeafInfo([8])],
                    'GGGAGGG': [LeafInfo([7]), LeafInfo([8])],
                    'AGGGGGG': [LeafInfo([7])],
                    'TTTTTTT': [LeafInfo([10]), LeafInfo([11])],
                    'TTTATTT': [LeafInfo([10]), LeafInfo([11])],
                    'ATTTTTT': [LeafInfo([10])]}
        for query, expected_value in expected.items():
            self.assertCountEqual(r.query(query, 0, 1, False), expected_value)

    def test_query_1_mismatch_with_gu(self):
        r = trie.Node(None)
        r.insert('AAAAAAA', LeafInfo([1]))
        r.insert('AAATAAA', LeafInfo([2]))
        r.insert('AAATTAA', LeafInfo([3]))
        r.insert('CCCCCCC', LeafInfo([4]))
        r.insert('CCCGCCC', LeafInfo([5]))
        r.insert('CCCGGCC', LeafInfo([6]))
        r.insert('GGGGGGG', LeafInfo([7]))
        r.insert('GGGTGGG', LeafInfo([8]))
        r.insert('GGGTTGG', LeafInfo([9]))
        r.insert('TTTTTTT', LeafInfo([10]))
        r.insert('TTTGTTT', LeafInfo([11]))
        r.insert('TTTGGTT', LeafInfo([12]))

        # Test above keys with mismatches and allowing G-U pairing
        expected = {'AAAAAAA': [LeafInfo([1]), LeafInfo([2]), LeafInfo([7]), LeafInfo([8])],
                    'AAAGAAA': [LeafInfo([1]), LeafInfo([2]), LeafInfo([7]), LeafInfo([8])],
                    'CAAAAAA': [LeafInfo([1]), LeafInfo([7])],
                    'ATATATA': [],
                    'CCCCCCC': [LeafInfo([4]), LeafInfo([5]), LeafInfo([10]), LeafInfo([11])],
                    'CCCACCC': [LeafInfo([4]), LeafInfo([5]), LeafInfo([6]), LeafInfo([10]), LeafInfo([11]), LeafInfo([12])],
                    'GCCCCCC': [LeafInfo([4]), LeafInfo([10])],
                    'GGGGGGG': [LeafInfo([7]), LeafInfo([8])],
                    'GGGAGGG': [LeafInfo([7]), LeafInfo([8])],
                    'AGGGGGG': [LeafInfo([7]), LeafInfo([8])],
                    'TTTTTTT': [LeafInfo([10]), LeafInfo([11])],
                    'TTTATTT': [LeafInfo([10]), LeafInfo([11]), LeafInfo([12])],
                    'ATTTTTT': [LeafInfo([10])]}
        for query, expected_value in expected.items():
            self.assertCountEqual(r.query(query, 0, 1, True), expected_value)

    def test_remove_and_reinsert(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))

        # Check if the contents are as expected
        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([3,6]),
                'ACG': LeafInfo([4]),
                'GTG': LeafInfo([5]),
                'TCC': LeafInfo([7])}
        for query in all_3mers():
            if query in true:
                expected = [true[query]]
            else:
                expected = []
            self.assertCountEqual(r.query(query, 0, 0, False), expected)

        # Remove a few strings
        self.assertTrue(r.remove('GGG'))
        self.assertTrue(r.remove('ACC'))
        self.assertTrue(r.remove('GTG'))
        self.assertFalse(r.remove('GTG'))
        self.assertFalse(r.remove('CCC'))

        # Check if the contents are as expected
        true_with_removals = {'AAA': LeafInfo([1]),
                              'ACG': LeafInfo([4]),
                              'TCC': LeafInfo([7])}
        for query in all_3mers():
            if query in true_with_removals:
                expected = [true_with_removals[query]]
            else:
                expected = []
            self.assertCountEqual(r.query(query, 0, 0, False), expected)

        # Re-insert the removed strings
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))

        # Check if the contents are as expected
        for query in all_3mers():
            if query in true:
                expected = [true[query]]
            else:
                expected = []
            self.assertCountEqual(r.query(query, 0, 0, False), expected)

    def test_traverse_and_find(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))
        r.insert('GAA', LeafInfo([2]))
        r.insert('GCC', LeafInfo([5]))
        r.insert('GCC', LeafInfo([7]))

        self.assertCountEqual(r.traverse_and_find(1),
            [('AAA', LeafInfo([1]))])
        self.assertCountEqual(r.traverse_and_find(2),
            [('GGG', LeafInfo([2])), ('GAA', LeafInfo([2]))])
        self.assertCountEqual(r.traverse_and_find(3),
            [('ACC', LeafInfo([3,6]))])
        self.assertCountEqual(r.traverse_and_find(4),
            [('ACG', LeafInfo([4]))])
        self.assertCountEqual(r.traverse_and_find(5),
            [('GTG', LeafInfo([5])), ('GCC', LeafInfo([5,7]))])
        self.assertCountEqual(r.traverse_and_find(6),
            [('ACC', LeafInfo([3,6]))])
        self.assertCountEqual(r.traverse_and_find(7),
            [('TCC', LeafInfo([7])), ('GCC', LeafInfo([5,7]))])

    def test_traverse_and_remove(self):
        r = trie.Node(None)
        r.insert('AAA', LeafInfo([1]))
        r.insert('GGG', LeafInfo([2]))
        r.insert('ACC', LeafInfo([3]))
        r.insert('ACG', LeafInfo([4]))
        r.insert('GTG', LeafInfo([5]))
        r.insert('ACC', LeafInfo([6]))
        r.insert('TCC', LeafInfo([7]))
        r.insert('GAA', LeafInfo([2]))
        r.insert('GCC', LeafInfo([5]))
        r.insert('GCC', LeafInfo([7]))

        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([3,6]),
                'ACG': LeafInfo([4]),
                'GTG': LeafInfo([5]),
                'TCC': LeafInfo([7]),
                'GAA': LeafInfo([2]),
                'GCC': LeafInfo([5,7])}
        def check_contents():
            # Check if the contents are as expected
            for query in all_3mers():
                if query in true:
                    expected = [true[query]]
                else:
                    expected = []
                self.assertCountEqual(r.query(query, 0, 0, False), expected)

        # Start by checking contents
        check_contents()

        # Repeatedly remove and check contents
        r.traverse_and_remove(3)
        true['ACC'] = LeafInfo([6])
        check_contents()

        r.traverse_and_remove(100)
        check_contents()

        r.traverse_and_remove(3)
        check_contents()

        r.traverse_and_remove(1)
        del true['AAA']
        check_contents()

        r.traverse_and_remove(2)
        del true['GGG']
        del true['GAA']
        check_contents()

        r.traverse_and_remove(5)
        true['GCC'] = LeafInfo([7])
        del true['GTG']
        check_contents()

        r.traverse_and_remove(7)
        del true['TCC']
        del true['GCC']
        check_contents()

        r.traverse_and_remove(4)
        del true['ACG']
        check_contents()

        r.traverse_and_remove(3)
        check_contents()

        r.traverse_and_remove(6)
        del true['ACC']
        check_contents()

        r.traverse_and_remove(1)
        check_contents()


class TestTrie(unittest.TestCase):
    """Tests methods in Trie class.

    Many of these operations are wrappers around methods in Node,
    so these unit tests are less extensive than ones in TestNode.
    """

    def test_insert_and_query(self):
        t = trie.Trie()
        t.insert([('AAA', LeafInfo([1])),
                  ('AAA', LeafInfo([1])),
                  ('GGG', LeafInfo([2])),
                  ('ACC', LeafInfo([3])),
                  ('ACG', LeafInfo([4])),
                  ('GTG', LeafInfo([5])),
                  ('ACC', LeafInfo([6])),
                  ('TCC', LeafInfo([7]))])

        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([3,6]),
                'ACG': LeafInfo([4]),
                'GTG': LeafInfo([5]),
                'TCC': LeafInfo([7])}
        for query in all_3mers():
            if query in true:
                expected = [true[query]]
            else:
                expected = []
            self.assertCountEqual(t.query(query, 0, False), expected)

    def test_mask_and_unmask(self):
        t = trie.Trie()

        t.insert([('AAA', LeafInfo([1])),
                  ('GGG', LeafInfo([2])),
                  ('ACC', LeafInfo([3])),
                  ('ACG', LeafInfo([4])),
                  ('GTG', LeafInfo([5])),
                  ('ACC', LeafInfo([6])),
                  ('TCC', LeafInfo([7])),
                  ('GAA', LeafInfo([2])),
                  ('GCC', LeafInfo([5])),
                  ('GCC', LeafInfo([7]))])

        true = {'AAA': LeafInfo([1]),
                'GGG': LeafInfo([2]),
                'ACC': LeafInfo([3,6]),
                'ACG': LeafInfo([4]),
                'GTG': LeafInfo([5]),
                'TCC': LeafInfo([7]),
                'GAA': LeafInfo([2]),
                'GCC': LeafInfo([5,7])}
        true_orig = dict(true)
        def check_contents():
            # Check if the contents are as expected
            for query in all_3mers():
                if query in true:
                    expected = [true[query]]
                else:
                    expected = []
                self.assertCountEqual(t.query(query, 0, False), expected)

        # Start by checking contents
        check_contents()

        # Repeatedly mask and check contents
        t.mask(3)
        true['ACC'] = LeafInfo([6])
        check_contents()

        t.mask(100)
        check_contents()

        t.mask(3)
        check_contents()

        t.mask(1)
        del true['AAA']
        check_contents()

        t.mask(2)
        del true['GGG']
        del true['GAA']
        check_contents()

        t.mask(5)
        true['GCC'] = LeafInfo([7])
        del true['GTG']
        check_contents()

        t.mask(7)
        del true['TCC']
        del true['GCC']
        check_contents()

        t.mask(4)
        del true['ACG']
        check_contents()

        t.mask(3)
        check_contents()

        t.mask(6)
        del true['ACC']
        check_contents()

        t.mask(1)
        check_contents()

        # Unmask everything
        t.unmask_all()

        # Check contents
        true = true_orig
        check_contents()
