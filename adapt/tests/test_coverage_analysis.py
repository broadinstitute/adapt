"""Tests for coverage_analysis module.
"""

import logging
import unittest
import math

from adapt import coverage_analysis

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya Pillai <ppillai@broadinstitute.org>'


class TestCoverageAnalysisWithMismatchModel(unittest.TestCase):
    """Tests methods in the CoverageAnalysisWithMismatchModel class.
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
                designs, 1, primer_mismatches=1, allow_gu_pairs=False)

        # Index sequences with a k-mer length of k=2 to ensure k-mers
        # will be found
        self.ca._index_seqs(k=2, stride_by_k=False)

    def test_find_binding_pos(self):
        bind_fn = self.ca.guide_bind_fn
        for fully_sensitive in [False, True]:
            self.ca.fully_sensitive = fully_sensitive
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'TTCGAT', bind_fn),
                    {6}
            )
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'TTCC', bind_fn),
                    {6}
            )
            self.assertEqual(
                    self.ca.find_binding_pos('seq1', 'AAAA', bind_fn),
                    set()
            )
            if fully_sensitive is True:
                self.assertEqual(
                        self.ca.find_binding_pos('seq1', 'AT', bind_fn),
                        {0,4,5,6,10,14}
                )
        self.ca.fully_sensitive = False

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
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGA'},
                    {'TCGT'}, {'CGAT'}),
                {'seq1'}
        )
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGA'},
                    {'TCGA'}, {'GGAA'}),
                {'seq1'}
        )
        self.ca.primer_terminal_mismatches = 0
        self.ca.bases_from_terminal = 1
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGA'},
                    {'TCGA'}, {'CGAT'}),
                {'seq1'}
        )
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGA'},
                    {'TCGT'}, {'CGAT'}),
                set()
        )
        self.assertEqual(
                self.ca.seqs_where_targets_bind({'TTCGT'},
                    {'TCGA'}, {'GGAA'}),
                set()
        )

    def test_seqs_where_primers_bind(self):
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAT'}, {'CGAT'}),
                {'seq1', 'seq2'}
        )
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAG'}, {'CGAT'}),
                {'seq1', 'seq2'}
        )
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAT'}, {'TGAT'}),
                {'seq1'}
        )
        self.ca.primer_terminal_mismatches = 0
        self.ca.bases_from_terminal = 1
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAT'}, {'CGAT'}),
                {'seq1', 'seq2'}
        )
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAG'}, {'CGAT'}),
                set()
        )
        self.assertEqual(
                self.ca.seqs_where_primers_bind({'ATCGAAT'}, {'TGAT'}),
                set()
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

    def test_per_seq_guide(self):
        self.ca.designs = {
            'design1': coverage_analysis.Design({'GGGCC'}),
            'design2': coverage_analysis.Design({'TTCGA', 'CCGA'}),
            'design3': coverage_analysis.Design({'ATTCGAT'}),
            'design4': coverage_analysis.Design({'ATTCGAG'}),
            'design5': coverage_analysis.Design({'ATTCGAT', 'ATTCGAG'}),
        }
        per_seq_guide_scores = self.ca.per_seq_guide()
        self.assertIn('design1', per_seq_guide_scores)
        self.assertDictEqual(
                per_seq_guide_scores['design1'],
                {'seq1': [[math.inf, ], None, None],
                 'seq2': [[math.inf, ], None, None],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertIn('design2', per_seq_guide_scores)
        self.assertDictEqual(
                per_seq_guide_scores['design2'],
                {'seq1': [[0, ], 'TTCGA', 6],
                 'seq2': [[0, ], 'TTCGA', 8],
                 'seq3': [[0, ], 'CCGA', 5]}
        )
        self.assertIn('design3', per_seq_guide_scores)
        self.assertDictEqual(
                per_seq_guide_scores['design3'],
                {'seq1': [[0, ], 'ATTCGAT', 5],
                 'seq2': [[1, ], 'ATTCGAT', 7],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertIn('design4', per_seq_guide_scores)
        self.assertDictEqual(
                per_seq_guide_scores['design4'],
                {'seq1': [[1, ], 'ATTCGAG', 5],
                 'seq2': [[0, ], 'ATTCGAG', 7],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertIn('design5', per_seq_guide_scores)
        self.assertDictEqual(
                per_seq_guide_scores['design5'],
                {'seq1': [[0, ], 'ATTCGAT', 5],
                 'seq2': [[0, ], 'ATTCGAG', 7],
                 'seq3': [[math.inf, ], None, None]}
        )

    def test_per_seq_primers(self):
        self.ca.designs = {
            'design1': coverage_analysis.Design(None, ({'GGGCC'},
                                                       {'TTCGA', 'CCGA'})),
            'design2': coverage_analysis.Design(None, ({'TATC'}, {'TATC'})),
            'design3': coverage_analysis.Design(None, ({'ATTCGAT'},
                                                       {'ATTCGAT'})),
        }
        per_seq_primers = self.ca.per_seq_primers()
        self.assertIn('design1', per_seq_primers[0])
        self.assertIn('design1', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design1'],
                {'seq1': [[math.inf, ], None, None],
                 'seq2': [[math.inf, ], None, None],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design1'],
                {'seq1': [[0, ], 'TTCGA', 6],
                 'seq2': [[0, ], 'TTCGA', 8],
                 'seq3': [[0, ], 'CCGA', 5]}
        )
        self.assertIn('design2', per_seq_primers[0])
        self.assertIn('design2', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design2'],
                {'seq1': [[1, ], 'TATC', 9],
                 'seq2': [[1, ], 'TATC', 1],
                 'seq3': [[1, ], 'TATC', 7]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design2'],
                {'seq1': [[1, ], 'TATC', 9],
                 'seq2': [[1, ], 'TATC', 1],
                 'seq3': [[1, ], 'TATC', 7]}
        )
        self.assertIn('design3', per_seq_primers[0])
        self.assertIn('design3', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design3'],
                {'seq1': [[0, ], 'ATTCGAT', 5],
                 'seq2': [[1, ], 'ATTCGAT', 7],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design3'],
                {'seq1': [[0, ], 'ATTCGAT', 5],
                 'seq2': [[1, ], 'ATTCGAT', 7],
                 'seq3': [[math.inf, ], None, None]}
        )

        self.ca.primer_terminal_mismatches = 0
        self.ca.bases_from_terminal = 1

        per_seq_primers = self.ca.per_seq_primers()
        self.assertIn('design1', per_seq_primers[0])
        self.assertIn('design1', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design1'],
                {'seq1': [[math.inf, ], None, None],
                 'seq2': [[math.inf, ], None, None],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design1'],
                {'seq1': [[0, 0], 'TTCGA', 6],
                 'seq2': [[0, 0], 'TTCGA', 8],
                 'seq3': [[0, 0], 'CCGA', 5]}
        )
        # With terminal mismatches, only the left primer should be valid
        self.assertIn('design2', per_seq_primers[0])
        self.assertIn('design2', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design2'],
                {'seq1': [[1, 0], 'TATC', 9],
                 'seq2': [[1, 0], 'TATC', 1],
                 'seq3': [[1, 0], 'TATC', 7]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design2'],
                {'seq1': [[math.inf, ], None, None],
                 'seq2': [[math.inf, ], None, None],
                 'seq3': [[math.inf, ], None, None]}
        )
        # With terminal mismatches, only the right primer should be valid for
        # the second sequence
        self.assertIn('design3', per_seq_primers[0])
        self.assertIn('design3', per_seq_primers[1])
        self.assertDictEqual(
                per_seq_primers[0]['design3'],
                {'seq1': [[0, 0], 'ATTCGAT', 5],
                 'seq2': [[math.inf, ], None, None],
                 'seq3': [[math.inf, ], None, None]}
        )
        self.assertDictEqual(
                per_seq_primers[1]['design3'],
                {'seq1': [[0, 0], 'ATTCGAT', 5],
                 'seq2': [[1, 0], 'ATTCGAT', 7],
                 'seq3': [[math.inf, ], None, None]}
        )

    def tearDown(self):
        # Re-enable logging
        logging.disable(logging.NOTSET)


class TestCoverageAnalysisWithPredictedActivity(unittest.TestCase):
    """Tests methods in the CoverageAnalysisWithPredictedActivity class.
    """

    def setUp(self):
        # Disable logging
        logging.disable(logging.WARNING)

        # Predict guides matching target to have activity 1, and
        # starting with 'A' to have activity 2 (otherwise, 0)
        # Predict guides to be highly active iff they start with A and
        # and match target
        class PredictorTest:
            def __init__(self):
                self.context_nt = 1

            def determine_highly_active(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    target_without_context = target[
                        self.context_nt:len(target)-self.context_nt]
                    if guide == target_without_context:
                        if guide[0] == 'A':
                            y += [True]
                        else:
                            y += [False]
                    else:
                        y += [False]
                return y

            def compute_activity(self, start_pos, pairs):
                y = []
                for target, guide in pairs:
                    target_without_context = target[
                        self.context_nt:len(target)-self.context_nt]
                    if guide == target_without_context:
                        if guide[0] == 'A':
                            y += [2]
                        else:
                            y += [1]
                    else:
                        y += [0]
                return y

            def cleanup_memoized(self, pos):
                pass

        predictor = PredictorTest()

        self.seqs = {
                'seq1': 'ATCGAATTCGATCGAA',
                'seq2': 'GGATCGAATTCGAG',
                'seq3': 'TGCAACCGATCCGT'
        }
        self.ca = coverage_analysis.CoverageAnalyzerWithPredictedActivity(
                self.seqs, {}, predictor, primer_mismatches=1,
                highly_active=False)

        # Index sequences with a k-mer length of k=2 to ensure k-mers
        # will be found
        self.ca._index_seqs(k=2, stride_by_k=False)

    def test_scores_where_guide_binds(self):
        self.assertDictEqual(
                self.ca.scores_where_guide_binds({'GGGCC'}),
                {'seq1': [[0, ], None, None],
                 'seq2': [[0, ], None, None],
                 'seq3': [[0, ], None, None]}
        )
        self.assertDictEqual(
                self.ca.scores_where_guide_binds({'ATTCGAT'}),
                {'seq1': [[2, ], 'ATTCGAT', 5],
                 'seq2': [[0, ], None, None],
                 'seq3': [[0, ], None, None]}
        )
        self.assertDictEqual(
                self.ca.scores_where_guide_binds({'TTCGA'}),
                {'seq1': [[1, ], 'TTCGA', 6],
                 'seq2': [[1, ], 'TTCGA', 8],
                 'seq3': [[0, ], None, None]}
        )
        self.assertDictEqual(
                self.ca.scores_where_guide_binds({'TTCGA', 'ATTCGAT'}),
                {'seq1': [[2, ], 'ATTCGAT', 5],
                 'seq2': [[1, ], 'TTCGA', 8],
                 'seq3': [[0, ], None, None]}
        )
        self.assertDictEqual(
                self.ca.scores_where_guide_binds({'TTCGA', 'CCGA'}),
                {'seq1': [[1, ], 'TTCGA', 6],
                 'seq2': [[1, ], 'TTCGA', 8],
                 'seq3': [[1, ], 'CCGA', 5]}
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
        self.ca.highly_active = True
        # Highly active is true, so nothing binds unless guide starts with A
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'TTCGA'}),
                set()
        )
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'TTCGA', 'CCGA'}),
                set()
        )
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'GGGCC'}),
                set()
        )
        self.assertEqual(
                self.ca.seqs_where_guides_bind({'AATT'}),
                {'seq1', 'seq2'}
        )
