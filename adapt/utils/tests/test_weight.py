"""Tests for weight module.
"""

import unittest
import math

from adapt.utils import weight

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'


class TestIndexCompress(unittest.TestCase):
    """Tests functions for compressing/de-compressing integers.
    """

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
        test_percentiles = weight.percentile([2.5, -1, 0, 4], [0, 60, 50, 100],
                                             [0.2, 0.2, 0.4, 0.2])
        self.assertEqual(test_percentiles, [-1, 0, -1, 4])


