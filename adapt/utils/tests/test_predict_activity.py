"""Tests predict_activity module.
"""

import unittest

import numpy as np

from adapt.utils import predict_activity

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestPredictor(unittest.TestCase):
    """Tests methods in the Predictor class.
    """

    def setUp(self):
        # Use the provided models with default thresholds
        classification_model_path = 'models/classify/model-51373185'
        regression_model_path = 'models/regress/model-f8b6fd5d'
        self.predictor = predict_activity.Predictor(
                classification_model_path,
                regression_model_path)

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

    def test_classify(self):
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

        self.assertListEqual(self.predictor._classify(pairs_onehot),
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

        self.assertListEqual(self.predictor._regress(pairs_onehot),
                [False, True])

    def test_determine_highly_active(self):
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

        self.assertListEqual(self.predictor._determine_highly_active(pairs),
                [False, True, False, False, True])

    def test_evaluate(self):
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

        self.assertListEqual(self.predictor.evaluate(1, pairs_1),
                [False])
        self.assertListEqual(self.predictor.evaluate(5, pairs_23),
                [True, False])
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
        self.assertListEqual(self.predictor.evaluate(5, pairs_45),
                [False, True])

        # Cleanup memoizations and test
        self.predictor.cleanup_memoized(5)
        self.assertListEqual(self.predictor.evaluate(5, pairs_45),
                [False, True])
        self.assertListEqual(self.predictor.evaluate(5, pairs_54),
                [True, False])
