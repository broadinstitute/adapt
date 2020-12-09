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

    def test_determine_consensus_sequence_b(self):
        self.assertEqual(self.b.determine_consensus_sequence(), 'ATCGAT')

    def test_determine_consensus_sequence_c(self):
        with self.assertRaises(ValueError):
            # Should fail when determining consensus sequence given an indel
            self.c.determine_consensus_sequence()
        self.assertIn(self.c.determine_consensus_sequence([0]), 'ATCGAA')

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

    def test_construct_guide_a(self):
        self.assertEqual(self.a.construct_guide(0, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('ATCG', {0,1,2,3}))
        self.assertEqual(self.a.construct_guide(0, 4, {0: {0,1,2,3,4}}, 1, False, self.gc),
                         ('ATCG', {0,1,2,3,4}))
        self.assertEqual(self.a.construct_guide(0, 4, {0: {4}}, 1, False, self.gc),
                         ('AGCG', {4}))
        self.assertIn(self.a.construct_guide(0, 4, {0: {2,3}}, 0, False, self.gc),
                      [('ATCG', {2,3}), ('ACCG', {2,3})])
        self.assertEqual(self.a.construct_guide(1, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('TCGA', {0,1,2,3}))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 0, False, self.gc),
                         ('CGAA', {0,2,4}))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 1, False, self.gc),
                         ('CGAA', {0,1,2,3,4}))
        self.assertEqual(self.a.construct_guide(2, 4, {0: {0,1,2,3,4}}, 2, False, self.gc),
                         ('CGAA', {0,1,2,3,4}))
        self.assertIn(self.a.construct_guide(2, 4, {0: {0,1,2,3}}, 0, False, self.gc),
                      [('CGAA', {0,2}), ('CGAT', {1,3})])
        self.assertIn(self.a.construct_guide(2, 4, {0: {0,1,2,3}}, 1, False, self.gc),
                      [('CGAA', {0,1,2,3}), ('CGAT', {0,1,2,3})])

    def test_construct_guide_b(self):
        # self.b has many Ns, which makes it difficult to write test cases
        # when clustering (the clusters tend to consist of guides in
        # which a position only has N); so pass None to guide_clusterer in
        # construct_guide() to skip clustering
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 0, False, None),
                         ('ATCG', {0,2}))
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 1, False, None),
                         ('ATCG', {0,2}))
        self.assertEqual(self.b.construct_guide(0, 4, {0: {0,1,2,3,4}}, 2, False, None),
                         ('ATCG', {0,1,2,3,4}))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 0, False, None),
                         ('CGAA', {0}))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 1, False, None),
                         ('CGAT', {0}))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 2, False, None),
                         ('CGAT', {0,1,2,3}))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {0,1,2,3,4}}, 3, False, None),
                         ('CGAT', {0,1,2,3,4}))
        self.assertEqual(self.b.construct_guide(2, 4, {0: {2,4}}, 2, False, None),
                         ('CGAC', {2,4}))
        self.assertIn(self.b.construct_guide(2, 4, {0: {2,3,4}}, 2, False, None),
                      [('CGAC', {2,4}), ('CGAT', {2,3})])
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when 'N' is all that exists at a position
            self.b.construct_guide(0, 4, {0: {1,3,4}}, 0, False, None)
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when a potential guide (here, 'CGAC') cannot
            # bind to any sequence because they all have 'N' somewhere
            self.b.construct_guide(2, 4, {0: {2,4}}, 1, False, None)

    def test_construct_guide_c(self):
        with self.assertRaises(alignment.CannotConstructGuideError):
            # Should fail when the only sequence given (1) has an indel
            self.c.construct_guide(0, 4, {0: {1}}, 0, False, self.gc)

    def test_construct_guide_with_large_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        seqs_aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 3, 1: 1}
        # 'ATCGAA' is most sequences, and let's construct a guide by
        # needing more from the group consisting of these sequences
        self.assertEqual(seqs_aln.construct_guide(0, 4, seqs_to_consider, 0,
                            False, self.gc, num_needed=num_needed),
                         ('ATCG', {0, 1, 3, 4, 5}))

    def test_construct_guide_with_small_group_needed(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'ATCGAA',
                'ATCGAA',
                'GGGCCC']
        seqs_aln = alignment.Alignment.from_list_of_seqs(seqs)

        seqs_to_consider = {0: {0, 1, 3, 4, 5}, 1: {2, 6}}
        num_needed = {0: 1, 1: 2}
        # 'ATCGAA' is most sequences, but let's construct a guide by
        # needing more from a group consisting of the 'GGGCCC' sequences
        self.assertEqual(seqs_aln.construct_guide(0, 4, seqs_to_consider, 0,
                            False, self.gc, num_needed=num_needed),
                         ('GGGC', {2, 6}))

    def test_construct_guide_with_suitable_fn(self):
        seqs = ['GTATCAAAT',
                'CTACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        seqs_to_consider = {0: {0, 1, 2, 3}}
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)

        # The best guide is 'GTATCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer)
        gd, covered_seqs = p
        self.assertEqual(gd, 'GTATCA')
        self.assertEqual(covered_seqs, {0, 2, 3})

        # Do not allow guides with 'TAT' in them
        def f(guide):
            if 'TAT' in guide:
                return False
            else:
                return True
        # Now the best guide is 'CTACCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
            guide_is_suitable_fn=f)
        gd, covered_seqs = p
        self.assertEqual(gd, 'CTACCA')
        self.assertEqual(covered_seqs, {1})

        # Do not allow guides with 'A' in them
        def f(guide):
            if 'A' in guide:
                return False
            else:
                return True
        # Now there is no suitable guide
        with self.assertRaises(alignment.CannotConstructGuideError):
            aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
                guide_is_suitable_fn=f)

    def test_construct_guide_with_predictor(self):
        seqs = ['GTATCAAAT',
                'ATACCAAAA',
                'GTATCAAAT',
                'GTATCAAAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        seqs_to_consider = {0: {0, 1, 2, 3}}
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)

        # The best guide is 'GTATCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer)
        gd, covered_seqs = p
        self.assertEqual(gd, 'GTATCA')
        self.assertEqual(covered_seqs, {0, 2, 3})

        # Only predict guides starting with 'A' to be active
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'A']
                return y
        predictor = PredictorTest()
        # Now the best guide is 'ATACCA'
        p = aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
            predictor=predictor, stop_early=False)
        gd, covered_seqs = p
        self.assertEqual(gd, 'ATACCA')
        self.assertEqual(covered_seqs, {1})

        # Only predict guides starting with 'A' to be active, and impose an
        # early stopping criterion
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'A']
                return y
        predictor = PredictorTest()
        # With early stopping, it will not find a guide
        with self.assertRaises(alignment.CannotConstructGuideError):
            aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
                predictor=predictor, stop_early=True)

        # Only predictor guides starting with 'C' to be active
        class PredictorTest:
            def __init__(self):
                self.context_nt = 0
            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    y += [guide[0] == 'C']
                return y
        predictor = PredictorTest()
        # Now there is no suitable guide
        with self.assertRaises(alignment.CannotConstructGuideError):
            aln.construct_guide(0, guide_length, seqs_to_consider, 1, False, guide_clusterer,
                predictor=predictor)

    def test_construct_guide_with_required_flanking(self):
        seqs = ['TCAAAT',
                'CCAAAA',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'CATTTT',
                'TCAAAT',
                'TCAAAT']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 2
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=2)
        seqs_to_consider = {0: set(range(len(seqs)))}

        # The best guide at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        p = aln.construct_guide(2, guide_length, seqs_to_consider, 1, False,
            guide_clusterer, required_flanking_seqs=('C', None))
        gd, covered_seqs = p
        self.assertEqual(gd, 'AA')
        self.assertEqual(covered_seqs, {0,1,9,10})

        # The best guide at start=2 is 'TT', but if we require
        # 'C' to flank on the 5' end, the best is 'AA'
        # Now if we require 'M' on the 5' end, 'TT' will be the best guide
        p = aln.construct_guide(2, guide_length, seqs_to_consider, 1, False,
            guide_clusterer, required_flanking_seqs=('M', None))
        gd, covered_seqs = p
        self.assertEqual(gd, 'TT')
        self.assertEqual(covered_seqs, {2,3,4,5,6,7,8})

    def test_determine_representative_guides(self):
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
                'T-GGAA']
        aln = alignment.Alignment.from_list_of_seqs(seqs)
        guide_length = 6
        guide_clusterer = alignment.SequenceClusterer(
            lsh.HammingDistanceFamily(guide_length),
            k=3)
        seqs_to_consider = {0: set(range(len(seqs)))}

        representatives = aln.determine_representative_guides(0,
                guide_length, seqs_to_consider, guide_clusterer)
        self.assertSetEqual(representatives,
                {'TCAAAT', 'CCAAAA', 'GGGGGG', 'CATTTT'})

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

    def test_sequences_bound_by_guide(self):
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

        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=(None, None)),
                         [2,3,4,5,6,7,8,9])
        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=('A', None)),
                         [2,3,4,5,6,7,8])
        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=('C', None)),
                         [9])
        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'G')),
                         [9])
        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'GN')),
                         [9])
        self.assertEqual(aln.sequences_bound_by_guide('TT', 2, 0, False,
                            required_flanking_seqs=(None, 'N')),
                         [2,3,4,5,6,7,8,9])

    def test_sequences_bound_by_guide_with_required_flanking(self):
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 0, False),
                         [0,1,2,3])
        self.assertEqual(self.a.sequences_bound_by_guide('ATCG', 0, 1, False),
                         [0,1,2,3,4])

    def test_most_common_sequence_simple(self):
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequence(
                            skip_ambiguity=False),
                         'ATCGAA')
        self.assertEqual(self.d_seqs_aln.determine_most_common_sequence(
                            skip_ambiguity=True),
                         'ATCGAA')

    def test_most_common_sequence_with_ambiguity(self):
        seqs = ['ATCNAA',
                'ATCNAA',
                'GGGCCC',
                'ATCGAA',
                'ATCNAA',
                'ATCNAA',
                'GGGCCC']
        aln = alignment.Alignment.from_list_of_seqs(seqs)

        self.assertEqual(aln.determine_most_common_sequence(skip_ambiguity=False),
                         'ATCNAA')
        self.assertEqual(aln.determine_most_common_sequence(skip_ambiguity=True),
                         'GGGCCC')

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
        self.assertEqual(aln.position_entropy(),
                         entropy)

    def test_position_entropy_with_ambiguity(self):
        seqs = ['MRWSYKVHDBN']
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
                  [0.25, 0.25, 0.25, 0.25]]

        entropy = [sum([-p*log2(p) for p in ps]) for ps in all_ps]
        self.assertEqual(aln.position_entropy(),
                         entropy)

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

