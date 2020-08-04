"""Tests for seq_io module.
"""

import logging
import tempfile
from os import unlink
import unittest

from adapt.utils import year_cover

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestReadYears(unittest.TestCase):
    """Tests reading years from a file.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

    def test_read_success(self):
        # Write the temporary years file
        fn = tempfile.NamedTemporaryFile(mode='w', delete=False)
        fn.write("genome_1\t2016\n")
        fn.write("genome_2\t2016\n")
        fn.write("genome_3\t2018\n")
        fn.write("genome_4\t2016\n")
        fn.write("\n")
        fn.write("\n")
        # Closes the file so that it can be reopened on Windows
        fn.close()

        expected = {2016: {"genome_1", "genome_2", "genome_4"},
                    2018: {"genome_3"}}

        years = year_cover.read_years(fn.name)
        self.assertEqual(years, expected)        

        # Delete temporary file
        unlink(fn.name)

    def test_read_fail_nonyear(self):
        fn = tempfile.NamedTemporaryFile(mode='w', delete=False)
        fn.write("genome_1\t2016\n")
        fn.write("genome_2\t16\n")
        fn.write("genome_3\t2018\n")
        fn.write("genome_4\t2016\n")
        fn.write("\n")
        fn.write("\n")
        # Closes the file so that it can be reopened on Windows
        fn.close()
        with self.assertRaises(ValueError):
            years = year_cover.read_years(fn.name)
        # Delete temporary file
        unlink(fn.name)

    def test_read_fail_duplicate_sequence(self):
        fn = tempfile.NamedTemporaryFile(mode='w', delete=False)
        fn.write("genome_1\t2016\n")
        fn.write("genome_2\t2016\n")
        fn.write("genome_3\t2018\n")
        fn.write("genome_1\t2018\n")
        fn.write("\n")
        fn.write("\n")
        # Closes the file so that it can be reopened on Windows
        fn.close()
        with self.assertRaises(ValueError):
            years = year_cover.read_years(fn.name)
        # Delete temporary file
        unlink(fn.name)

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)


class TestConstructPartialCovers(unittest.TestCase):
    """Tests constructing partial cover for each year.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.INFO)

    def test_construct_partial_covers_full_outside_range(self):
        cover_frac = year_cover.construct_partial_covers(
            {2013, 2014, 2015}, 2018, 1.0, 0.9)
        self.assertEqual(cover_frac,
            {2013: 0.9**5, 2014: 0.9**4, 2015:0.9**3})

    def test_construct_partial_covers_full_in_range(self):
        cover_frac = year_cover.construct_partial_covers(
            {2013, 2014, 2015, 2016, 2017}, 2015, 1.0, 0.9)
        self.assertEqual(cover_frac,
            {2013: 0.9**2, 2014: 0.9, 2015: 1.0, 2016: 1.0, 2017: 1.0})

    def test_construct_partial_covers_skip_years(self):
        cover_frac = year_cover.construct_partial_covers(
            {2013, 2014, 2017, 2018}, 2018, 0.5, 0.9)
        self.assertEqual(cover_frac,
            {2013: 0.5 * 0.9**5, 2014: 0.5 * 0.9**4, 2017: 0.5 * 0.9, 2018: 0.5})

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)

