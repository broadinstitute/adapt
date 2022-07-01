"""Tests for alignment module.
"""

import random
import unittest
from math import log2

from adapt import alignment
from adapt.utils import lsh

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestAlignment(unittest.TestCase):
    """Tests methods in the Alignment class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        self.a_seqs = ['ATCGAA', 'ATCGAT', 'AYCGAA', 'AYCGAT', 'AGCGAA']
        self.a = alignment.Alignment.from_list_of_seqs(self.a_seqs)

        self.b_seqs = ['ATCGAA', 'ATNNAT', 'ATCGNN', 'ATNNAT', 'ATNNAC']
        self.b = alignment.Alignment.from_list_of_seqs(self.b_seqs)

        self.c_seqs = ['ATCGAA', 'ATC-AA']
        self.c = alignment.Alignment.from_list_of_seqs(self.c_seqs)

        self.d_seqs = ['ATCGAA',
                       'ATCGAA',
                       'GGGCCC',
                       'ATCGAA',
                       'ATCGAA',
                       'ATCGAA',
                       'GGGCCC']
        self.d_seqs_aln = alignment.Alignment.from_list_of_seqs(self.d_seqs)

        self.gc = alignment.SequenceClusterer(lsh.HammingDistanceFamily(4), k=2)

    def test_make_list_of_seqs_a(self):
        self.assertEqual(self.a.make_list_of_seqs(),
                         self.a_seqs)
        self.assertEqual(self.a.make_list_of_seqs([0]),
                         ['ATCGAA'])
        self.assertEqual(self.a.make_list_of_seqs([0,3]),
                         ['ATCGAA', 'AYCGAT'])

    def test_make_list_of_seqs_b(self):
        self.assertEqual(self.b.make_list_of_seqs(),
                         self.b_seqs)

    def test_make_list_of_seqs_c(self):
        self.assertEqual(self.c.make_list_of_seqs(),
                         self.c_seqs)

    def test_make_list_of_seqs_c_with_no_gaps(self):
        self.assertEqual(self.c.make_list_of_seqs(remove_gaps=True),
                         ['ATCGAA', 'ATCAA'])

    def test_determine_consensus_sequence_a(self):
        self.assertEqual(self.a.determine_consensus_sequence(), 'ATCGAA')
        self.assertEqual(self.a.determine_consensus_sequence([0]), 'ATCGAA')
        self.assertIn(self.a.determine_consensus_sequence([2]),
                      ['ATCGAA', 'ACCGAA'])
        self.assertIn(self.a.determine_consensus_sequence([0,1]),
                      ['ATCGAA', 'ATCGAT'])
        self.assertIn(self.a.determine_consensus_sequence([1,2]),
                      ['ATCGAT', 'ATCGAA'])

        # Check weighted consensus
        self.a.seq_norm_weights = [0.1, 0.6, 0.1, 0.1, 0.1]
        self.assertEqual(self.a.determine_consensus_sequence(), 'ATCGAT')

    def test_determine_consensus_sequence_b(self):
        self.assertEqual(self.b.determine_consensus_sequence(), 'ATCGAT')

    def test_determine_consensus_sequence_c(self):
        with self.assertRaises(ValueError):
            # Should fail when determining consensus sequence given an indel
            self.c.determine_consensus_sequence()
        self.assertIn(self.c.determine_consensus_sequence([0]), 'ATCGAA')

    def test_seq_idxs_weighted(self):
        seqs = ['ATCGAA',
                'GGGCCC',
                'ATCGAA']
        seqs_aln = alignment.Alignment.from_list_of_seqs(
            seqs, seq_norm_weights=[3/7, 2/7, 2/7])

        # 'ATCGAA' is most sequences, and let's construct a guide by
        # needing more from the group consisting of these sequences
        self.assertEqual(seqs_aln.seq_idxs_weighted([]), 0)
        self.assertAlmostEqual(seqs_aln.seq_idxs_weighted([1]), 2/7)
        self.assertAlmostEqual(seqs_aln.seq_idxs_weighted([0, 2]), 5/7)
        self.assertAlmostEqual(seqs_aln.seq_idxs_weighted([0, 1, 2]), 1)

    def test_seqs_with_gap(self):
        self.assertCountEqual(self.a.seqs_with_gap(), [])
        self.assertCountEqual(self.b.seqs_with_gap(), [])
        self.assertCountEqual(self.c.seqs_with_gap(), [1])

    def test_seqs_with_required_flanking_none_required(self):
        # No required flanking sequences should yield that all match
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 3, (None, None)),
            {0,1,2,3,4,5,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                0, 3, (None, None)),
            {0,1,2,3,4,5,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, (None, None)),
            {0,1,2,3,4,5,6}
        )

    def test_seqs_with_required_flanking_subset_to_consider(self):
        # Only look over a subset of the input sequences
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 3, (None, None), seqs_to_consider={2,3,6}),
            {2,3,6}
        )

    def test_seqs_with_required_flanking_end5(self):
        # Required flanking on 5' end
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('ATC', None)),
            {0,1,3,4,5}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('C', None)),
            {0,1,3,4,5}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('C', None), seqs_to_consider={0,1,3,4,5}),
            {0,1,3,4,5}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('A', None)),
            {}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('S', None)),
            {0,1,2,3,4,5,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 3, ('NC', None)),
            {0,1,3,4,5}
        )

    def test_seqs_with_required_flanking_end5_too_close(self):
        # Required flanking on 5' end, with guide too close to end
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                1, 3, ('ATC', None)),
            {}
        )

    def test_seqs_with_required_flanking_end3(self):
        # Required flanking on 3' end
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                0, 3, (None, 'CCC')),
            {2,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                0, 3, (None, 'CNN')),
            {2,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                0, 3, (None, 'SNN')),
            {0,1,2,3,4,5,6}
        )

    def test_seqs_with_required_flanking_end3_too_close(self):
        # Required flanking on 3' end, with guide too close to end
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                1, 3, (None, 'CCC')),
            {}
        )

    def test_seqs_with_required_flanking_both_ends(self):
        # Required flanking on both ends
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 2, ('G', 'AA')),
            {}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 2, ('T', 'MN')),
            {0,1,3,4,5}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 2, ('T', 'MN'), seqs_to_consider={1,2,3,4}),
            {1,3,4}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 2, ('K', 'MN')),
            {0,1,2,3,4,5,6}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                2, 2, ('K', 'MN'), seqs_to_consider={4,5}),
            {4,5}
        )

    def test_seqs_with_required_flanking_both_ends_too_close(self):
        # Required flanking on both ends, with guide too close to end
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                3, 2, ('K', 'MN')),
            {}
        )
        self.assertCountEqual(self.d_seqs_aln.seqs_with_required_flanking(
                0, 2, ('K', 'MN')),
            {}
        )

    def test_determine_representative_oligos(self):
        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'GGGGGG',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'TCAAAT',
                'TCAAAT',
                'TCAAAA',
                'T-GGTA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)
        seqs_to_consider = {0: set(range(len(seqs)))}

        # Note that, along with being a cluster consensus, 'CATTTT' is also the
        # overall consensus (according to how the consensus function is
        # defined, which takes the most common allele)
        representatives = aln.determine_representative_oligos(0,
                guide_length, seqs_to_consider, guide_clusterer)
        self.assertSetEqual(representatives,
                {'TCAAAT', 'CCAAAA', 'GGGGGG', 'CATTTT'})

    def test_determine_representative_oligos_with_distinct_consensus(self):
        # Here, unlike above, the overall consensus is not a cluster consensus
        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTA',
                'GGGGGG',
                'CATTTA',
                'CATTTA',
                'CATTTA',
                'TCAAAT',
                'TCAAAT',
                'TCAAAT',
                'T-GGAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)
        seqs_to_consider = {0: set(range(len(seqs)))}

        representatives = aln.determine_representative_oligos(0,
                guide_length, seqs_to_consider, guide_clusterer)
        self.assertSetEqual(representatives,
                {'TCAAAT', 'CCAAAA', 'GGGGGG', 'CATTTA', 'CATTTT'})

    def test_compute_activity(self):
        # Predict guides matching target to have activity 1, and
        # starting with 'A' to have activity 2 (otherwise, 0)
        class PredictorTest:
            def __init__(self):
                self.context_nt = 1
            def compute_activity(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    target_without_context = target[self.context_nt:len(target)-self.context_nt]
                    if guide == target_without_context:
                        if guide[0] == 'A':
                            y += [2]
                        else:
                            y += [1]
                    else:
                        y += [0]
                return y
        predictor = PredictorTest()

        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'GGGGGG',
                'CATTTT',
                'CATTTT',
                'CAT-TT',
                'TCAAAT',
                'TCA-AT',
                'TCAAAA',
                'TCGGAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        activities_AAA = aln.compute_activity(2, 'AAA', predictor)
        self.assertListEqual(list(activities_AAA),
                [2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0])

        activities_TTT = aln.compute_activity(2, 'TTT', predictor)
        self.assertListEqual(list(activities_TTT),
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])

    def test_sequences_bound_by_oligo(self):
        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CCTTGT',
                'TCAAAT',
                'TCAAAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=(None, None)),
                         [2,3,4,5,6,7,8,9])
        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=('A', None)),
                         [2,3,4,5,6,7,8])
        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=('C', None)),
                         [9])
        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'G')),
                         [9])
        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'GN')),
                         [9])
        self.assertEqual(aln.sequences_bound_by_oligo('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'N')),
                         [2,3,4,5,6,7,8,9])

    def test_sequences_bound_by_oligo_with_required_flanking(self):
        self.assertEqual(self.a.sequences_bound_by_oligo('ATCG', 0, 0, False),
                         [0,1,2,3])
        self.assertEqual(self.a.sequences_bound_by_oligo('ATCG', 0, 1, False),
                         [0,1,2,3,4])

    def test_most_common_sequence_simple(self):
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequences(
                            skip_ambiguity=False),
                         ['ATCGAA'])
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequences(
                            skip_ambiguity=True),
                         ['ATCGAA'])
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequences(
                            n=2),
                         ['ATCGAA','GGGCCC'])
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequences(
                            n=3),
                         ['ATCGAA','GGGCCC'])

    def test_most_common_sequence_weighted(self):
        seqs = ['ATCGAA',
                'GGGCCC',
                'ATCGAA']
        seqs_aln = alignment.Alignment.from_list_of_seqs(
            seqs, seq_norm_weights=[1/7, 5/7, 1/7])
        self.assertEqual(seqs_aln.determine_most_common_sequences(
                            skip_ambiguity=False),
                         ['GGGCCC'])
        self.assertEqual(seqs_aln.determine_most_common_sequences(
                            skip_ambiguity=True),
                         ['GGGCCC'])
        self.assertEqual(seqs_aln.determine_most_common_sequences(
                            n=2),
                         ['GGGCCC','ATCGAA'])
        self.assertEqual(seqs_aln.determine_most_common_sequences(
                            n=3),
                         ['GGGCCC','ATCGAA'])

    def test_most_common_sequence_with_ambiguity(self):
        seqs = ['ATCNAA',
                'ATCNAA',
                'GGGCCC',
                'ATCGAA',
                'ATCNAA',
                'ATCNAA',
                'GGGCCC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=False),
                         ['ATCNAA'])
        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=True),
                         ['GGGCCC'])
        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=False, n=2),
                         ['ATCNAA', 'GGGCCC'])
        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=True, n=2),
                         ['GGGCCC', 'ATCGAA'])
        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=False, n=3),
                         ['ATCNAA', 'GGGCCC', 'ATCGAA'])
        self.assertEqual(aln.determine_most_common_sequences(skip_ambiguity=True, n=3),
                         ['GGGCCC', 'ATCGAA'])

    def test_position_entropy_simple(self):
        seqs = ['ACCCC',
                'AAGGC',
                'AAATA',
                'AAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        all_ps = [[1],
                  [0.25, 0.75],
                  [0.25, 0.25, .5],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.5, 0.5]]

        entropy = [sum([-p*log2(p) for p in ps]) for ps in all_ps]
        self.assertEqual(aln.position_entropy(), entropy)

    def test_position_entropy_weighted(self):
        seqs = ['ACCCC',
                'AAGGC',
                'AAATA',
                'AAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs,
            seq_norm_weights = [0, 0.25, 0.5, 0.25])
        all_ps = [[1],
                  [1],
                  [0.25, 0.75],
                  [0.25, 0.5, 0.25],
                  [0.25, 0.75]]

        entropy = [sum([-p*log2(p) for p in ps]) for ps in all_ps]
        self.assertEqual(aln.position_entropy(), entropy)

    def test_position_entropy_with_ambiguity(self):
        seqs = ['MRWSYKVHDBN-']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        all_ps = [[0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5],
                  [1/3.0, 1/3.0, 1/3.0],
                  [1/3.0, 1/3.0, 1/3.0],
                  [1/3.0, 1/3.0, 1/3.0],
                  [1/3.0, 1/3.0, 1/3.0],
                  [0.25, 0.25, 0.25, 0.25],
                  [1]]

        entropy = [sum([-p*log2(p) for p in ps]) for ps in all_ps]
        self.assertEqual(aln.position_entropy(), entropy)

    def test_base_percentages_simple(self):
        seqs = ['ACCCC',
                'AAGGC',
                'AAATA',
                'AAAAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        base_p = {
            'A': 0.6,
            'C': 0.25,
            'G': 0.1,
            'T': 0.05
        }

        self.assertEqual(aln.base_percentages(), base_p)

    def test_base_percentages_weighted(self):
        seqs = ['ACCCC',
                'AAGGC',
                'AAATA',
                'AAAAA']

        aln = alignment.Alignment.from_list_of_seqs(seqs,
            seq_norm_weights = [0, 0.25, 0.5, 0.25])
        base_p = {
            'A': 0.75,
            'C': 0.05,
            'G': 0.1,
            'T': 0.1
        }

        self.assertEqual(aln.base_percentages(), base_p)

    def test_base_percentages_with_ambiguity(self):
        seqs = ['MRWSYKVHDBN-']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        base_p = {
            'A': 0.25,
            'C': 0.25,
            'G': 0.25,
            'T': 0.25
        }

        self.assertEqual(aln.base_percentages(), base_p)

    def test_construct_from_0_seqs(self):
        with self.assertRaises(Exception):
            seqs = []
            alignment.Alignment.from_list_of_seqs(seqs)


class TestSequenceClusterer(unittest.TestCase):
    """Tests the SequenceClusterer class.
    """

    def setUp(self):
        # Set a random seed so hash functions are always the same
        random.seed(0)

        family = lsh.HammingDistanceFamily(10)
        self.sc = alignment.SequenceClusterer(family, k=3)

    def test_cluster(self):
        seqs = [('ATCGAAATAA', 4),
                ('ATCGAAATAA', 5),
                ('ATTGAAATAT', 1),
                ('CTGGTCATAA', 2),
                ('CTCGTCATAA', 3)]

        clusters = self.sc.cluster(seqs)
        self.assertCountEqual(clusters,
            {frozenset({1, 4, 5}), frozenset({2, 3})})

    def test_largest_cluster(self):
        seqs = [('ATCGAAATAA', 4),
                ('ATCGAAATAA', 5),
                ('ATTGAAATAT', 1),
                ('CTGGTCATAA', 2),
                ('CTCGTCATAA', 3)]

        largest_cluster = self.sc.largest_cluster(seqs)
        self.assertCountEqual(largest_cluster, {1, 4, 5})

