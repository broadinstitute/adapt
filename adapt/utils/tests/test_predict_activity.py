"""Tests predict_activity module.
"""

import unittest

import numpy as np
import os

from adapt import alignment
from adapt.utils import predict_activity, thermo
from adapt.utils.version import get_project_path, get_latest_model_version

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

FAKE_DNA_DNA_INSIDE ={
    'A': {
        'A': (1, 0),
        'T': (10, 0.05),
        'C': (2, 0),
        'G': (3, 0),
    },
    'T': {
        'A': (10, 0.05),
        'T': (1, 0),
        'C': (2, 0),
        'G': (3, 0),
    },
    'C': {
        'A': (2, 0),
        'T': (2, 0),
        'C': (2, 0),
        'G': (10, 0.05),
    },
    'G': {
        'A': (3, 0),
        'T': (3, 0),
        'C': (10, 0.05),
        'G': (3, 0),
    }
}

FAKE_DNA_DNA_INTERNAL = {
    'A': FAKE_DNA_DNA_INSIDE,
    'T': FAKE_DNA_DNA_INSIDE,
    'C': FAKE_DNA_DNA_INSIDE,
    'G': FAKE_DNA_DNA_INSIDE,
}

FAKE_DNA_DNA_TERMINAL = FAKE_DNA_DNA_INTERNAL

FAKE_DNA_DNA_TERM_GC = (0, 0)

FAKE_DNA_DNA_SYM = (0, 0)

FAKE_DNA_DNA_TERM_AT = (0, 0)

# With n bp matching, delta H is 10n and delta S is 0.05n. With thermodynamic
# conditions set to not interfere, the melting temperature is delta H/delta S,
# which is 200K (Note: this doesn't actually make sense in practice, as Tm
# can't go below 0°C; this is a toy example to make testing easier)
PERFECT_TM = 200

class TestPredictor(unittest.TestCase):
    """Tests methods in the Predictor class.
    """

    def setUp(self):
        # Use the provided models with default thresholds
        dir_path = get_project_path()
        cla_path_all = os.path.join(dir_path, 'models', 'classify', 'cas13a')
        reg_path_all = os.path.join(dir_path, 'models', 'regress', 'cas13a')
        cla_version = get_latest_model_version(cla_path_all)
        reg_version = get_latest_model_version(reg_path_all)
        cla_path = os.path.join(cla_path_all, cla_version)
        reg_path = os.path.join(reg_path_all, reg_version)

        self.predictor = predict_activity.Predictor(cla_path, reg_path)

    def test_model_input_from_nt(self):
        # Make context (both ends) be all 'A'
        context_1 = 'A' * self.predictor.context_nt
        # Make guide and target both be the same, an arbitrary sequence
        guide_1 = 'ATCG'
        target_with_context_1 = context_1 + guide_1 + context_1

        # Make context (both ends) be all 'T'
        context_2 = 'T' * self.predictor.context_nt
        # Make guide and target both be the same, an arbitrary sequence
        guide_2 = 'CCGG'
        target_with_context_2 = context_2 + guide_2 + context_2

        pairs = [(target_with_context_1, guide_1),
                (target_with_context_2, guide_2)]

        expected_onehot_1 = [[1,0,0,0,0,0,0,0]]*self.predictor.context_nt
        expected_onehot_1 += [[1,0,0,0,1,0,0,0], [0,0,0,1,0,0,0,1],
            [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0]]
        expected_onehot_1 += [[1,0,0,0,0,0,0,0]]*self.predictor.context_nt

        expected_onehot_2 = [[0,0,0,1,0,0,0,0]]*self.predictor.context_nt
        expected_onehot_2 += [[0,1,0,0,0,1,0,0], [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0], [0,0,1,0,0,0,1,0]]
        expected_onehot_2 += [[0,0,0,1,0,0,0,0]]*self.predictor.context_nt

        expected = np.array([expected_onehot_1, expected_onehot_2])

        self.assertTrue(
                np.array_equal(self.predictor._model_input_from_nt(pairs),
                    expected))

    def test_classify_and_decide(self):
        # Target context is all A and, at guide, is all G
        # Guide is GC repeated, so has 14 mismatches against the target
        # Should be inactive
        target_with_context_1 = ('A'*self.predictor.context_nt +
                'G'*28 + 'A'*self.predictor.context_nt)
        guide_1 = 'GC'*14

        # Target context is all A and, at guide, is all G
        # Guide is all G, so matches target
        # Should be active
        target_with_context_2 = ('A'*self.predictor.context_nt +
                'G'*28 + 'A'*self.predictor.context_nt)
        guide_2 = 'G'*28

        pairs = [(target_with_context_1, guide_1), (target_with_context_2,
            guide_2)]
        pairs_onehot = self.predictor._model_input_from_nt(pairs)

        self.assertListEqual(self.predictor._classify_and_decide(pairs_onehot),
                [False, True])

    def test_regress(self):
        # Target context is all A (except G PFS) and, at guide, is all G
        # Guide is all G except all A in the seed region against the target
        # Should not be highly active
        target_with_context_1 = ('A'*self.predictor.context_nt +
                'G'*28 + 'G' + 'A'*(self.predictor.context_nt - 1))
        guide_1 = 'G'*9 + 'A'*9 + 'G'*10

        # Target context is all A and, at guide, is all G
        # Guide is all A, so matches target
        # Should (probably) be highly active
        target_with_context_2 = ('A'*self.predictor.context_nt +
                'A'*28 + 'A'*self.predictor.context_nt)
        guide_2 = 'A'*28

        pairs = [(target_with_context_1, guide_1), (target_with_context_2,
            guide_2)]
        pairs_onehot = self.predictor._model_input_from_nt(pairs)

        r = self.predictor._regress(pairs_onehot)
        self.assertLess(r[0], self.predictor.regression_threshold)
        self.assertGreater(r[1], self.predictor.regression_threshold)

    def test_determine_highly_active_computations(self):
        # Target context is all A and, at guide, is all G
        # Guide is GC repeated, so has 14 mismatches against the target
        # Should be inactive
        target_with_context_1 = ('A'*self.predictor.context_nt +
                'G'*28 + 'A'*self.predictor.context_nt)
        guide_1 = 'GC'*14

        # Target context is all A and, at guide, is all G
        # Guide is all A, so matches target
        # Should (probably) be highly active
        target_with_context_2 = ('A'*self.predictor.context_nt +
                'A'*28 + 'A'*self.predictor.context_nt)
        guide_2 = 'A'*28

        # Target context is all A (except G PFS) and, at guide, is all G
        # Guide is all G except all A in the seed region against the target
        # Should be inactive
        target_with_context_3 = ('A'*self.predictor.context_nt +
                'G'*28 + 'G' + 'A'*(self.predictor.context_nt - 1))
        guide_3 = 'G'*9 + 'A'*9 + 'G'*10

        # Target context is all A (except G PFS) and, at guide, is all G
        # Guide is all G except all A in the seed region against the target
        # Should be inactive
        target_with_context_4 = ('A'*self.predictor.context_nt +
                'G'*28 + 'G' + 'A'*(self.predictor.context_nt - 1))
        guide_4 = 'G'*9 + 'A'*9 + 'G'*10

        # Target context is all A and, at guide, is all G
        # Guide is all A, so matches target
        # Should (probably) be highly active
        target_with_context_5 = ('A'*self.predictor.context_nt +
                'A'*28 + 'A'*self.predictor.context_nt)
        guide_5 = 'A'*28

        pairs = [(target_with_context_1, guide_1),
                (target_with_context_2, guide_2),
                (target_with_context_3, guide_3),
                (target_with_context_4, guide_4),
                (target_with_context_5, guide_5)]

        # Use _run_models_and_memoize() to compute results, and use a
        # nonsense start position (-1) (it doesn't matter), and pull
        # output determinations about highly active
        self.predictor._run_models_and_memoize(-1, pairs)
        r = [self.predictor._memoized_evaluations[-1][pair] for pair in pairs]
        highly_active = [x[1] for x in r]

        self.assertListEqual(highly_active, [False, True, False, False, True])

    def test_memoized_evaluations(self):
        # Target context is all A and, at guide, is all G
        # Guide is GC repeated, so has 14 mismatches against the target
        # Should be inactive
        target_with_context_1 = ('A'*self.predictor.context_nt +
                'G'*28 + 'A'*self.predictor.context_nt)
        guide_1 = 'GC'*14

        # Target context is all A and, at guide, is all G
        # Guide is all A, so matches target
        # Should (probably) be highly active
        target_with_context_2 = ('A'*self.predictor.context_nt +
                'A'*28 + 'A'*self.predictor.context_nt)
        guide_2 = 'A'*28

        # Target context is all A (except G PFS) and, at guide, is all G
        # Guide is all G except all A in the seed region against the target
        # Should be inactive
        target_with_context_3 = ('A'*self.predictor.context_nt +
                'G'*28 + 'G' + 'A'*(self.predictor.context_nt - 1))
        guide_3 = 'G'*9 + 'A'*9 + 'G'*10

        pairs_1 = [(target_with_context_1, guide_1)]
        pairs_23 = [(target_with_context_2, guide_2),
                (target_with_context_3, guide_3)]

        self.assertListEqual(self.predictor.determine_highly_active(1, pairs_1),
                [False])
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_23),
                [True, False])
        self.assertEqual(self.predictor.compute_activity(1, pairs_1)[0], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_23)[0], 0)
        self.assertEqual(self.predictor.compute_activity(5, pairs_23)[1], 0)
        self.predictor.cleanup_memoized(1)
        self.assertEqual(self.predictor.compute_activity(1, pairs_1)[0], 0)
        self.predictor.cleanup_memoized(1)

        # Target context is all A (except G PFS) and, at guide, is all G
        # Guide is all G except all A in the seed region against the target
        # Should be inactive
        target_with_context_4 = ('A'*self.predictor.context_nt +
                'G'*28 + 'G' + 'A'*(self.predictor.context_nt - 1))
        guide_4 = 'G'*9 + 'A'*9 + 'G'*10

        # Target context is all A and, at guide, is all G
        # Guide is all A, so matches target
        # Should (probably) be highly active
        target_with_context_5 = ('A'*self.predictor.context_nt +
                'A'*28 + 'A'*self.predictor.context_nt)
        guide_5 = 'A'*28

        pairs_45 = [(target_with_context_4, guide_4),
                (target_with_context_5, guide_5)]
        pairs_54 = [(target_with_context_5, guide_5),
                (target_with_context_4, guide_4)]

        # Should be memoized
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_45),
                [False, True])
        self.assertEqual(self.predictor.compute_activity(5, pairs_45)[0], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_45)[1], 0)

        # Cleanup memoizations and test
        self.predictor.cleanup_memoized(5)
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_45),
                [False, True])
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_54),
                [True, False])
        self.assertEqual(self.predictor.compute_activity(5, pairs_45)[0], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_45)[1], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_54)[0], 0)
        self.assertEqual(self.predictor.compute_activity(5, pairs_54)[1], 0)

        # Do this again, with activity values first
        self.predictor.cleanup_memoized(5)
        self.assertEqual(self.predictor.compute_activity(5, pairs_45)[0], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_45)[1], 0)
        self.assertGreater(self.predictor.compute_activity(5, pairs_54)[0], 0)
        self.assertEqual(self.predictor.compute_activity(5, pairs_54)[1], 0)
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_45),
                [False, True])
        self.assertListEqual(self.predictor.determine_highly_active(5, pairs_54),
                [True, False])


class TestTmPredictor(unittest.TestCase):
    """Tests methods in the TestTmPredictor class.
    """
    def setUp(self):
        # Temporarily set constants to fake values
        self.DNA_DNA_INTERNAL = thermo.DNA_DNA_INTERNAL
        self.DNA_DNA_TERMINAL = thermo.DNA_DNA_TERMINAL
        self.DNA_DNA_TERM_GC = thermo.DNA_DNA_TERM_GC
        self.DNA_DNA_SYM = thermo.DNA_DNA_SYM
        self.DNA_DNA_TERM_AT = thermo.DNA_DNA_TERM_AT

        thermo.DNA_DNA_INTERNAL = FAKE_DNA_DNA_INTERNAL
        thermo.DNA_DNA_TERMINAL = FAKE_DNA_DNA_TERMINAL
        thermo.DNA_DNA_TERM_GC = FAKE_DNA_DNA_TERM_GC
        thermo.DNA_DNA_SYM = FAKE_DNA_DNA_SYM
        thermo.DNA_DNA_TERM_AT = FAKE_DNA_DNA_TERM_AT


    def test_compute_activity(self):
        seqs = ['ATCG',
                'ATCG',
                'GGGC',
                'ATCG',
                'ATCA',
                'GGGC']
        oligo = 'ATCG'
        pairs = [(seq, oligo) for seq in seqs]
        conditions = thermo.Conditions(sodium=1, magnesium=0, dNTP=0,
            oligo_concentration=1)
        shared_memo = {}
        left_predictor = predict_activity.TmPredictor(PERFECT_TM,
            conditions, False, shared_memo=shared_memo)
        right_predictor = predict_activity.TmPredictor(PERFECT_TM,
            conditions, True, shared_memo=shared_memo)
        activities = left_predictor.compute_activity(0, pairs)
        expected = np.array([0, 0, -PERFECT_TM, 0, -20, -PERFECT_TM])
        np.testing.assert_array_almost_equal(activities, expected)
        activities = right_predictor.compute_activity(0, pairs)
        expected = np.array([0, 0, -PERFECT_TM, 0, -30, -PERFECT_TM])
        np.testing.assert_array_almost_equal(activities, expected)

    def tearDown(self):
        # Fix modified constants and functions
        thermo.DNA_DNA_INTERNAL = self.DNA_DNA_INTERNAL
        thermo.DNA_DNA_TERMINAL = self.DNA_DNA_TERMINAL
        thermo.DNA_DNA_TERM_GC = self.DNA_DNA_TERM_GC
        thermo.DNA_DNA_SYM = self.DNA_DNA_SYM
        thermo.DNA_DNA_TERM_AT = self.DNA_DNA_TERM_AT


class TestSimpleBinaryPredictor(unittest.TestCase):
    """Tests methods in the SimpleBinaryPredictor class.
    """

    def test_compute_activity(self):
        seqs = ['ATCGAA',
                'ATCGAA',
                'GGGCCC',
                'ATCGAA',
                'AT-GAA',
                'ATCAAA',
                'GGGCCC']
        seqs_aln = alignment.Alignment.from_list_of_seqs(seqs)

        predictor = predict_activity.SimpleBinaryPredictor(1, False)
        activities = predictor.compute_activity(1, 'TCG', seqs_aln)
        expected = np.array([1, 1, 0, 1, 0, 1, 0])

        np.testing.assert_array_equal(activities, expected)

