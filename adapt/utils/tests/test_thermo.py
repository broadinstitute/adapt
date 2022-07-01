"""Tests for the thermo module.
"""

import unittest
import logging
import math
import numpy as np

from adapt.utils import thermo

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'

FAKE_DNA_DNA_INSIDE ={
    'A': {
        'A': (1, 0.005),
        'T': (10, 0.05),
        'C': (1, 0.005),
        'G': (1, 0.005),
    },
    'T': {
        'A': (10, 0.05),
        'T': (1, 0.005),
        'C': (1, 0.005),
        'G': (1, 0.005),
    },
    'C': {
        'A': (2, 0.01),
        'T': (2, 0.01),
        'C': (2, 0.01),
        'G': (20, 0.1),
    },
    'G': {
        'A': (2, 0.01),
        'T': (2, 0.01),
        'C': (20, 0.1),
        'G': (2, 0.01),
    }
}

FAKE_DNA_DNA_INTERNAL = {
    'A': FAKE_DNA_DNA_INSIDE,
    'T': FAKE_DNA_DNA_INSIDE,
    'C': FAKE_DNA_DNA_INSIDE,
    'G': FAKE_DNA_DNA_INSIDE,
}

FAKE_DNA_DNA_TERMINAL = FAKE_DNA_DNA_INTERNAL
FAKE_DNA_DNA_TERM_GC = (20, 0.1)
FAKE_DNA_DNA_SYM = (5, 0.05)
FAKE_DNA_DNA_TERM_AT = (5, 0.1)
FAKE_DNA_DNA_SALT = (0, 0.0)
FAKE_R_CONSTANT = 1/np.log(4)


class TestThermo(object):
    """General class for testing analyze_coverage.py

    Defines helper functions for test cases and basic setUp and
    tearDown functions.
    """
    class TestThermoCase(unittest.TestCase):
        def setUp(self):
            # Disable logging
            logging.disable(logging.WARNING)

            self.DNA_DNA_INTERNAL = thermo.DNA_DNA_INTERNAL
            self.DNA_DNA_TERMINAL = thermo.DNA_DNA_TERMINAL
            self.DNA_DNA_TERM_GC = thermo.DNA_DNA_TERM_GC
            self.DNA_DNA_SYM = thermo.DNA_DNA_SYM
            self.DNA_DNA_TERM_AT = thermo.DNA_DNA_TERM_AT
            self.DNA_DNA_SALT = thermo.DNA_DNA_SALT
            self.R_CONSTANT = thermo.R_CONSTANT

            thermo.DNA_DNA_INTERNAL = FAKE_DNA_DNA_INTERNAL
            thermo.DNA_DNA_TERMINAL = FAKE_DNA_DNA_TERMINAL
            thermo.DNA_DNA_TERM_GC = FAKE_DNA_DNA_TERM_GC
            thermo.DNA_DNA_SYM = FAKE_DNA_DNA_SYM
            thermo.DNA_DNA_TERM_AT = FAKE_DNA_DNA_TERM_AT
            thermo.DNA_DNA_SALT = FAKE_DNA_DNA_SALT
            thermo.R_CONSTANT = FAKE_R_CONSTANT

        def tearDown(self):
            # Re-enable logging
            logging.disable(logging.NOTSET)

            thermo.DNA_DNA_INTERNAL = self.DNA_DNA_INTERNAL
            thermo.DNA_DNA_TERMINAL = self.DNA_DNA_TERMINAL
            thermo.DNA_DNA_TERM_GC = self.DNA_DNA_TERM_GC
            thermo.DNA_DNA_SYM = self.DNA_DNA_SYM
            thermo.DNA_DNA_TERM_AT = self.DNA_DNA_TERM_AT
            thermo.DNA_DNA_SALT = self.DNA_DNA_SALT
            thermo.R_CONSTANT = self.R_CONSTANT

class TestThermoCases(TestThermo.TestThermoCase):
    def test_calculate_delta_h_s(self):
        # Note-the nearest neighbor method looks at pairs of base pairs
        # A/Ts
        # h = (3 XA/XT bases * 10) + (2 terminal AT bases * 5) = 40
        # s = (3 XA/XT bases * 0.05) + (2 terminal AT bases * 0.1) = 0.35
        h,s = thermo.calculate_delta_h_s('AAAT','AAAT')
        self.assertEqual(h, 40)
        self.assertAlmostEqual(s, 0.35)
        # C/Gs
        # h = (3 XC/XG bases * 20) + (2 terminal CG bases * 20) = 100
        # s = (3 XC/XG bases * 0.1) + (2 terminal CG bases * 0.1) = 0.5
        h,s = thermo.calculate_delta_h_s('GGGC','GGGC')
        self.assertEqual(h, 100)
        self.assertAlmostEqual(s, 0.5)
        # Mismatches
        # h = (2 XA/XT base * 10) + (1 XC/XA bases * 2)
        #    + (2 terminal AT bases * 5) = 32
        # s = (2 XA/XT base * 0.05) + (1 XC/XA bases * 0.01)
        #    + (2 terminal AT bases * 0.1) = 0.31
        h,s = thermo.calculate_delta_h_s('AAAT','AAAC')
        self.assertEqual(h, 32)
        self.assertAlmostEqual(s, 0.31)
        # Reverse oligo false
        # h = (2 XA/XT base * 10) + (1 XT/XG bases * 1)
        #    + (2 terminal AT bases * 5) = 31
        # s = (2 XA/XT base * 0.05) + (1 XT/XG bases * 0.005)
        #    + (2 terminal AT bases * 0.1) = 0.305
        h,s = thermo.calculate_delta_h_s('AAAT','AAAC', reverse_oligo=False)
        self.assertEqual(h, 31)
        self.assertAlmostEqual(s, 0.305)
        # Symmetric
        # h = (3 XA/XT bases * 10) + (2 terminal AT bases * 5) + (5 for symmetric) = 45
        # s = (3 XA/XT bases * 0.05) + (2 terminal AT bases * 0.1) + (0.05 for symmetric) = 0.4
        h,s = thermo.calculate_delta_h_s('AATT','AATT')
        self.assertEqual(h, 45)
        self.assertAlmostEqual(s, 0.4)
        # Salt
        # The salt correction is multiplied by the number of phosphates (for
        # 4 bases, there are 3 phosphates) times the log of the positive ions,
        # where positive ions is sodium + 120 * sqrt(magnesium^2-dNTP^2)
        # Salt correction was set to 0 to not affect other tests, change that
        # here
        thermo.DNA_DNA_SALT = (1, 0.05)
        # The following parameters set the positive ions to 1
        na_conc = math.e/2
        mg_conc = math.e**2/(120000**2)
        dNTP_conc = 3/4 * math.e**2/(120000**2)
        h,s = thermo.calculate_delta_h_s('GGGC', 'GGGC',
            conditions=thermo.Conditions(sodium=math.e, magnesium=mg_conc,
                dNTP=dNTP_conc))
        # Fix salt before doing any tests
        thermo.DNA_DNA_SALT = FAKE_DNA_DNA_SALT
        self.assertAlmostEqual(h, 103, places=3)
        self.assertAlmostEqual(s, 0.65, places=5)

    def test_calculate_delta_g(self):
        # g = h - t * s = 100 - 100*0.5 = 50
        g = thermo.calculate_delta_g('GGGC', 'GGGC',
            conditions=thermo.Conditions(t=100))
        self.assertAlmostEqual(g, 50)

    def test_calculate_equilibrium_constant(self):
        # g = h - t * s = 100 - 200*0.5 = 0
        # K = e^{-g/(R*t)} = e^0 = 1
        K = thermo.calculate_equilibrium_constant('GGGC', 'GGGC',
            conditions=thermo.Conditions(t=200))
        self.assertAlmostEqual(K, 1)
        # g = h - t * s = 100 - 100*0.4 = 50
        # K = e^{-g/(R*t)} = e^{-50/(R*100)} = 4^{-.5} = 1/2
        K = thermo.calculate_equilibrium_constant('GGGC', 'GGGC',
            conditions=thermo.Conditions(t=100))
        self.assertAlmostEqual(K, 1/2)

    def test_calculate_percent_bound(self):
        # Percent bound solves the roots of the following:
        # K[AB]^2 - (K[A_tot]+K[B_tot]+1)[AB] + K[A_tot][B_tot]
        # K[AB]^2 - (K[A_tot]+K[B_tot]+1)[AB] + K[A_tot][B_tot]
        # which (given k=1, [A_tot]=3, and [B_tot]=4) is:
        # [AB]^2 - 8[AB] + 12
        # the roots of which are 2 and 6. The possible percent bounds are
        # 2/4=50% and 6/4=150%, and only 50% is a valid percent bound
        percent_bound = thermo.calculate_percent_bound('GGGC', 'GGGC',
            conditions=thermo.Conditions(t=200, oligo_concentration=4,
                target_concentration=3))
        self.assertEqual(percent_bound, [.5])

    def test_calculate_calculate_melting_temp(self):
        # Melting temperature should be :
        # h/(s + R * log(oligo concentration-target concentration/2))
        # Since R is set to 1/log(4), this is:
        # h/(s + log_4(oligo concentration-target concentration/2))
        # Setting oligo concentration to 24 and target to 16 makes this:
        # h/(s+2) = 100/(.5+2) = 100/2.5 = 40
        melting_temp = thermo.calculate_melting_temp('GGGC', 'GGGC',
            conditions=thermo.Conditions(oligo_concentration=24,
                target_concentration=16))
        self.assertAlmostEqual(melting_temp, 40)

    def test_binds(self):
        # Melting temperature is 40, so within range
        bind = thermo.binds('GGGC', 'GGGC', 40, 5,
            conditions=thermo.Conditions(oligo_concentration=24,
                target_concentration=16))
        self.assertTrue(bind)
        # Melting temperature is 40, so outside range
        bind = thermo.binds('GGGC', 'GGGC', 30, 5,
            conditions=thermo.Conditions(oligo_concentration=24,
                target_concentration=16))
        self.assertFalse(bind)
        # Melting temperature is 40, so on edge of range (counts as true)
        bind = thermo.binds('GGGC', 'GGGC', 35, 5,
            conditions=thermo.Conditions(oligo_concentration=24,
                target_concentration=16))
        self.assertTrue(bind)

    def test_calculate_i_x(self):
        # No mismatches
        i_x = thermo.calculate_i_x('CATGCA', 'CATGCA')
        self.assertEqual(i_x, 0)
        # Mismatch at first base
        i_x = thermo.calculate_i_x('GATGCA', 'CATGCA')
        self.assertEqual(i_x, 6)
        # Mismatch at sixth base
        i_x = thermo.calculate_i_x('CATGCT', 'CATGCA')
        self.assertEqual(i_x, 1)
        # Mismatch at second and sixth base
        i_x = thermo.calculate_i_x('CTTGCT', 'CATGCA')
        self.assertEqual(i_x, 5)
        # <6bps
        i_x = thermo.calculate_i_x('CATC', 'CATG')
        self.assertEqual(i_x, 3)
        # Reverse oligo false
        # Mismatch at sixth base
        i_x = thermo.calculate_i_x('CATGCT', 'CATGCA', reverse_oligo=False)
        self.assertEqual(i_x, 6)

