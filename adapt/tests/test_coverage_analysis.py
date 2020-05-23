"""Tests for coverage_analysis module.
"""

import logging
import unittest

from adapt import coverage_analysis

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class TestCoverageAnalysisWithMismatchModel(unittest.TestCase):
    """Tests methods in the CoverageAnalysis class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        self.seqs = {
                'seq1': 'ATCGAATTCGATCGAA',
                'seq2': 'GGATCGAATTCGAG',
                'seq3': 'TGCAACCGATCCGT'
        }
        self.design1 = coverage_analysis.Design({'TTCGA'})
        self.design2 = coverage_analysis.Design({'TTCGA'},
                ({'TCGA'}, {'CGAT'}))
        self.design3 = coverage_analysis.Design({'TTCGA', 'CCGA', 'GGGGG'},
                ({'TCGA', 'GCAA', 'AAAA'}, {'CGAT', 'TCCG', 'CCCC'}))
        designs = {
                'design1': self.design1,
                'design2': self.design2,
                'design3': self.design3
        }
        self.ca = coverage_analysis.CoverageAnalyzerWithMismatchModel(self.seqs,
                designs, 1, 1, allow_gu_pairs=False)

        # Index sequences with a k-mer length of k=2 to ensure k-mers
        # will be found
        self.ca._index_seqs(k=2, stride_by_k=False)

    def test_find_binding_pos(self):
        bind_fn = self.ca.guide_bind_fn
        for allow_not_fully_sensitive in [False, True]:
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'TTCGAT', bind_fn,
                        allow_not_fully_sensitive=allow_not_fully_sensitive),
                    {6}
            )
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'TTCC', bind_fn,
                        allow_not_fully_sensitive=allow_not_fully_sensitive),
                    {6}
            )
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'AAAA', bind_fn,
                        allow_not_fully_sensitive=allow_not_fully_sensitive),
                    set()
            )
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'AT', bind_fn,
                        allow_not_fully_sensitive=allow_not_fully_sensitive),
                    {0,5,10}
            )

    def test_seqs_where_guide_binds(self):
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'TTCGA'}),
                {'seq1', 'seq2'}
        )
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'TTCGA', 'CCGA'}),
                {'seq1', 'seq2', 'seq3'}
        )
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'GGGCC'}),
                set()
        )

    def test_seqs_where_targets_bind(self):
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGA'},
                    {'TCGA'}, {'CGAT'}),
                {'seq1'}
        )

    def test_seqs_bound_by_design(self):
        self.assertEqual(self.ca.seqs_bound_by_design(self.design1),
                {'seq1', 'seq2'})
        self.assertEqual(self.ca.seqs_bound_by_design(self.design2),
                {'seq1'})
        self.assertEqual(self.ca.seqs_bound_by_design(self.design3),
                {'seq1', 'seq3'})
        
    def test_frac_of_seqs_bound(self):
        self.assertEqual(self.ca.frac_of_seqs_bound(),
                {'design1': 2.0/3.0, 'design2': 1.0/3.0, 'design3': 2.0/3.0})

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)
