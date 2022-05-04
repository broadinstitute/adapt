"""Tests for weight module.
"""

import logging
import unittest
import math

from adapt.utils import weight

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'


class TestWeights(unittest.TestCase):
    """Tests functions for weighting
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

    def test_weight_by_log_group(self):
        test_weights = weight.weight_by_log_group({
            'group_1': ['A', 'B'],
            'group_2': ['C', 'D']
        })
        group_weight = math.log(3)/2
        self.assertAlmostEqual(test_weights['A'], group_weight)
        self.assertAlmostEqual(test_weights['B'], group_weight)
        self.assertAlmostEqual(test_weights['C'], group_weight)
        self.assertAlmostEqual(test_weights['D'], group_weight)

        test_weights = weight.weight_by_log_group({
            'group_1': ['A'],
            'group_2': ['B', 'C', 'D']
        })
        group_weight = math.log(4)/3
        self.assertAlmostEqual(test_weights['A'], math.log(2)/1)
        self.assertAlmostEqual(test_weights['B'], group_weight)
        self.assertAlmostEqual(test_weights['C'], group_weight)
        self.assertAlmostEqual(test_weights['D'], group_weight)


    def test_normalize(self):
        test_norm_weights = weight.normalize({'A': 200, 'B': 200, 'C': 100},
                                             ['A', 'B'])
        self.assertAlmostEqual(test_norm_weights['A'], .5)
        self.assertAlmostEqual(test_norm_weights['B'], .5)
        self.assertNotIn('C', test_norm_weights)

        test_norm_weights = weight.normalize({'A': .2, 'B': .1, 'C': .1},
                                             ['C'])
        self.assertNotIn('A', test_norm_weights)
        self.assertNotIn('B', test_norm_weights)
        self.assertAlmostEqual(test_norm_weights['C'], 1)

        test_norm_weights = weight.normalize({'A': .5, 'B': .25, 'C': .25},
                                             ['A', 'B', 'C'])
        self.assertAlmostEqual(test_norm_weights['A'], .5)
        self.assertAlmostEqual(test_norm_weights['B'], .25)
        self.assertAlmostEqual(test_norm_weights['C'], .25)

        test_weights = weight.weight_by_log_group({'group_1': ['A'],
                                                   'group_2': ['B', 'C', 'D']})
        test_norm_weights = weight.normalize(test_weights,
                                             ['A', 'B', 'C', 'D'])
        self.assertAlmostEqual(test_norm_weights['A'], 1/3)
        self.assertAlmostEqual(test_norm_weights['B'], 2/9)
        self.assertAlmostEqual(test_norm_weights['C'], 2/9)
        self.assertAlmostEqual(test_norm_weights['D'], 2/9)


    def test_percentile(self):
        activities = [2.5, -1, 0, 4, 2.5]

        # Test empty
        test_percentiles = weight.percentile(activities, [],
                                             [0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(test_percentiles, [])
        # Test boundary percentiles
        test_percentiles = weight.percentile(activities,
                                             [0, 20, 40, 60, 80, 100],
                                             [0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(test_percentiles, [-1, -1, 0, 2.5, 2.5, 4])
        # Test in between percentiles
        test_percentiles = weight.percentile(activities,
                                             [10, 30, 50, 70, 90],
                                             [0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(test_percentiles, [-1, -1, 0, 2.5, 2.5])
        # Test normalization
        test_percentiles = weight.percentile(activities,
                                             [0, 10, 20, 30, 40, 50,
                                              60, 70, 80, 90, 100],
                                             [1, 1, 1, 1, 1])
        self.assertEqual(test_percentiles, [-1, -1, -1, -1, 0, 0,
                                            2.5, 2.5, 2.5, 2.5, 4])
        # Test non-uniform weights
        test_percentiles = weight.percentile(activities,
                                             [0, 50, 60, 90, 100],
                                             [0.1, 0.2, 0.4, 0.2, 0.1])
        self.assertEqual(test_percentiles, [-1, -1, 0, 2.5, 4])
        # Test non-uniform weight normalization
        test_percentiles = weight.percentile(activities,
                                             [0, 50, 60, 90, 100],
                                             [1, 2, 4, 2, 1])
        self.assertEqual(test_percentiles, [-1, -1, 0, 2.5, 4])
        # Test out of order percentiles
        test_percentiles = weight.percentile(activities,
                                             [60, 0, 99, 100, 50],
                                             [1, 2, 4, 2, 1])
        self.assertEqual(test_percentiles, [0, -1, 2.5, 4, -1])

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)

