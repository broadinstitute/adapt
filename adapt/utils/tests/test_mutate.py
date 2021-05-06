"""Tests mutate module.
"""

import unittest

import numpy as np

from adapt import alignment
from adapt.utils import mutate, predict_activity

__author__ = 'Priya Pillai <ppillai@broadinstitute.org>'


class TestGTRSubstitutionMutator(unittest.TestCase):
    """Tests methods in the GTRSubstitutionMutator class.
    """
    def setUp(self):
        self.seqs = ['AC',
                     'GT']
        aln_a = alignment.Alignment.from_list_of_seqs(self.seqs)
        self.mutator_a = mutate.GTRSubstitutionMutator(aln_a,
                                                       1, 1, 1, 1, 1, 1,
                                                       1, 1000, 10)
        self.wild_seq = 'AAAAAAAA'
        self.mutated_seqs = self.mutator_a.mutate(self.wild_seq)

        aln_b = alignment.Alignment.from_list_of_seqs(self.mutated_seqs)
        rates = np.random.rand(6)
        self.mutator_b = mutate.GTRSubstitutionMutator(aln_b,
                                                       *rates,
                                                       1, 1, 5)

    def test_construct_rateQ(self):
        expected_a_Q = [[-1, 1/3, 1/3, 1/3],
                        [1/3, -1, 1/3, 1/3],
                        [1/3, 1/3, -1, 1/3],
                        [1/3, 1/3, 1/3, -1]]
        np.testing.assert_allclose(self.mutator_a.Q, expected_a_Q)

        Q = self.mutator_b.Q
        for row in Q:
            self.assertAlmostEqual(sum(row), 0)

    def test_construct_P(self):
        expected_a_P = [[0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25]]
        np.testing.assert_allclose(self.mutator_a.P, expected_a_P)

        P = self.mutator_b.P
        for row in P:
            self.assertAlmostEqual(sum(row), 1)

    def test_mutate(self):
        self.assertEqual(len(self.mutated_seqs), self.mutator_a.n)
        for mutated_seq in self.mutated_seqs:
            self.assertEqual(len(mutated_seq), len(self.wild_seq))

    def test_compute_sequence_probability(self):
        for mutated_seq in self.mutated_seqs:
            self.assertAlmostEqual(self.mutator_a.compute_sequence_probability(
                self.wild_seq, mutated_seq), (0.25**8))

    def test_compute_mutated_activity(self):
        predictor_a = predict_activity.SimpleBinaryPredictor(8, False)
        activity = self.mutator_a.compute_mutated_activity(predictor_a,
                                                           self.wild_seq,
                                                           self.wild_seq)
        self.assertEqual(activity, 1)

        predictor_a.mismatches = 0
        predictor_a.required_flanking_seqs = ('A', None)
        activity = self.mutator_a.compute_mutated_activity(predictor_a,
                                                           'A'+self.wild_seq,
                                                           self.wild_seq)
        self.assertEqual(activity, 0)
