"""Methods for searching for optimal guides to use for a diagnostic.
"""

from collections import defaultdict
import logging
import math
import random

import numpy as np

from adapt import alignment
from adapt.utils import oligo
from adapt.utils import search
from adapt.utils import index_compress
from adapt.utils import lsh
from adapt.utils import predict_activity
from adapt.utils import weight

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


class GuideSearcher(search.OligoSearcher):
    """Methods to search for guides to use for a diagnostic.

    This is a base class, with subclasses defining methods depending on the
    objective. It should not be used without subclassing, as it does not define
    all the positional arguments necessary for search.OligoSearcher.
    """
    def __init__(self, guide_length, **kwargs):
        """
        Args:
            guide_length: integer length of the guide. Sets guide length to
                min_oligo_length and max_oligo_length.
            kwargs: see search.OligoSearcher.__init__()
        """
        if 'min_oligo_length' in kwargs or 'max_oligo_length' in kwargs:
            raise ValueError("Variable oligo lengths are not yet implemented "
                "for guides; you cannot use min_oligo_length or "
                "max_oligo_length.")
        super().__init__(min_oligo_length=guide_length,
            max_oligo_length=guide_length, **kwargs)


class GuideSearcherMinimizeGuides(search.OligoSearcherMinimizeNumber,
        GuideSearcher):
    """Methods to minimize the number of guides.
    """

    def __init__(self, aln, guide_length, mismatches, cover_frac,
            missing_data_params, **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            mismatches: threshold on number of mismatches for determining whether
                a guide would hybridize to a target sequence
            cover_frac: minimum weighted fraction in (0, 1] of sequences that
                must be 'captured' by a guide set; see seq_groups. The
                weighted fraction is the sum of the normalized weights of the
                sequences that are 'captured'.
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            seq_groups: dict that maps group ID to collection of sequences in
                that group. If set, cover_frac must also be a dict that maps
                group ID to the weighted fraction of sequences in that group
                that must be 'captured' by a guide. If None, then do not divide
                the sequences into groups.
            kwargs: see OligoSearcherMinimizeNumber.init() and
                GuideSearcher.__init__()
        """
        super().__init__(aln=aln, guide_length=guide_length,
            mismatches=mismatches, missing_data_params=missing_data_params,
            cover_frac=cover_frac, **kwargs)

    def _compress_result(self, p):
        """Compress the information to be stored in self._memo

        Args:
            p: result of calling construct_oligo()

        Returns:
            compressed version of p
        """
        gd, covered_seqs, score = p

        # covered_seqs may contain mostly contiguous indices
        covered_seqs_compressed = index_compress.compress_mostly_contiguous(covered_seqs)

        return (gd, covered_seqs_compressed, score)

    def _decompress_result(self, p_compressed):
        """"Decompress the information stored in self._memo

        Args:
            p_compressed: output of _compress_result()

        Returns:
            decompressed version of p_compressed
        """
        gd, covered_seqs_compressed, score = p_compressed

        # Decompress covered_seqs
        covered_seqs = index_compress.decompress_ranges(covered_seqs_compressed)

        return (gd, covered_seqs, score)

    def find_guides_with_sliding_window(self, window_size, out_fn,
            window_step=1, sort=False, print_analysis=True):
        """Find the smallest collection of guides that cover sequences, across
        all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            window_size: length of the window to use when sliding across
                alignment
            out_fn: output TSV file to write guide sequences by window
            window_step: amount by which to increase the window start for
                every search
            sort: if set, sort output TSV by number of guides (ascending)
                then by score (descending); when not set, default is to
                sort by window position
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the smallest number of guides and highest
                score
        """
        guide_collections = list(self._find_oligos_for_each_window(
            window_size, window_step=window_step))

        if sort:
            # Sort by number of guides ascending (len(x[2])), then by
            # score of guides descending
            guide_collections.sort(key=lambda x: (len(x[2]),
                -self._score_collection(x[2])))

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'score', 'total-frac-bound', 'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, guide_seqs = guides_in_window
                score = self._score_collection(guide_seqs)
                frac_bound = self.total_frac_bound(guide_seqs)
                count = len(guide_seqs)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_positions[gd_seq]
                             for gd_seq in guide_seqs_sorted]
                positions_str = ' '.join(str(p) for p in positions)
                line = [start, end, count, score, frac_bound, guide_seqs_str,
                        positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')

        if print_analysis:
            num_windows_scanned = len(
                range(0, self.aln.seq_length - window_size + 1))
            num_windows_with_guides = len(guide_collections)

            if num_windows_with_guides == 0:
                stat_display = [
                    ("Number of windows scanned", num_windows_scanned),
                    ("Number of windows with guides", num_windows_with_guides)
                ]
            else:
                min_count = min(len(x[2]) for x in guide_collections)
                num_with_min_count = sum(1 for x in guide_collections
                    if len(x[2]) == min_count)

                min_count_str = (str(min_count) + " guide" +
                                 ("s" if min_count > 1 else ""))

                stat_display = [
                    ("Number of windows scanned", num_windows_scanned),
                    ("Number of windows with guides", num_windows_with_guides),
                    ("Minimum number of guides required in a window", min_count),
                    ("Number of windows with " + min_count_str,
                        num_with_min_count),
                ]

            # Print the above statistics, with padding on the left
            # so that the statistic names are right-justified in a
            # column and the values line up, left-justified, in a column
            max_stat_name_len = max(len(name) for name, val in stat_display)
            for name, val in stat_display:
                pad_spaces = max_stat_name_len - len(name)
                name_padded = " "*pad_spaces + name + ":"
                print(name_padded, str(val))


class GuideSearcherMaximizeActivity(search.OligoSearcherMaximizeActivity,
        GuideSearcher):
    """Methods to maximize expected activity of the guide set.
    """

    def __init__(self, aln, guide_length, soft_guide_constraint,
            hard_guide_constraint, penalty_strength,
            missing_data_params, **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            guide_length: length of the guide to construct
            soft_guide_constraint: number of guides for the soft constraint
            hard_guide_constraint: number of guides for the hard constraint
            penalty_strength: coefficient in front of the soft penalty term
                (i.e., its importance relative to expected activity)
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            kwargs: see OligoSearcherMaximizeActivity.init() and
                GuideSearcher.__init__()
        """
        super().__init__(aln=aln, guide_length=guide_length,
            soft_constraint=soft_guide_constraint,
            hard_constraint=hard_guide_constraint,
            penalty_strength=penalty_strength,
            missing_data_params=missing_data_params, **kwargs)

    def find_guides_with_sliding_window(self, window_size, out_fn,
            window_step=1, sort=False, print_analysis=True):
        """Find a collection of guides that maximizes expected activity,
        across all windows.

        This writes a table of the guides to a file, in which each row
        corresponds to a window in the genome. It also optionally prints
        an analysis to stdout.

        Args:
            window_size: length of the window to use when sliding across
                alignment
            out_fn: output TSV file to write guide sequences by window
            window_step: amount by which to increase the window start for
                every search
            sort: if set, sort output TSV by objective value
            print_analysis: print to stdout the best window(s) -- i.e.,
                the one(s) with the highest objective value
        """
        guide_collections = list(self._find_oligos_for_each_window(
            window_size, window_step=window_step))

        if sort:
            # Sort by objective value descending
            guide_collections.sort(key=lambda x: self.obj_value(x[2]),
                    reverse=True)

        with open(out_fn, 'w') as outf:
            # Write a header
            outf.write('\t'.join(['window-start', 'window-end',
                'count', 'objective-value', 'total-frac-bound',
                'guide-set-expected-activity',
                'guide-set-median-activity', 'guide-set-5th-pctile-activity',
                'guide-expected-activities',
                'target-sequences',
                'target-sequence-positions']) + '\n')

            for guides_in_window in guide_collections:
                start, end, guide_seqs = guides_in_window
                count = len(guide_seqs)
                activities = self.oligo_set_activities(start, end, guide_seqs)
                obj = self.obj_value(start, end, guide_seqs,
                        activities=activities)
                frac_bound = self.total_frac_bound(start, end,
                        guide_seqs, activities=activities)
                guides_activity_expected = self.oligo_set_activities_expected_value(
                        start, end, guide_seqs, activities=activities)
                guides_activity_median, guides_activity_5thpctile = \
                        self.oligo_set_activities_percentile(start, end,
                                guide_seqs, [50, 5], activities=activities)

                guide_seqs_sorted = sorted(list(guide_seqs))
                guide_seqs_str = ' '.join(guide_seqs_sorted)
                positions = [self._selected_positions[gd_seq]
                             for gd_seq in guide_seqs_sorted]
                positions_str = ' '.join(str(p) for p in positions)
                expected_activities_per_guide_dict = \
                        self.oligo_set_activities_expected_value_per_oligo(
                            start, end, guide_seqs)
                expected_activities_per_guide = \
                        [expected_activities_per_guide_dict[gd_seq]
                         for gd_seq in guide_seqs_sorted]
                expected_activities_per_guide_str = ' '.join(
                        str(a) for a in expected_activities_per_guide)
                line = [start, end, count, obj, frac_bound,
                        guides_activity_expected,
                        guides_activity_median, guides_activity_5thpctile,
                        expected_activities_per_guide_str,
                        guide_seqs_str,
                        positions_str]

                outf.write('\t'.join([str(x) for x in line]) + '\n')

        if print_analysis:
            num_windows_scanned = len(
                range(0, self.aln.seq_length - window_size + 1))
            num_windows_with_guides = len(guide_collections)

            stat_display = [
                ("Number of windows scanned", num_windows_scanned),
                ("Number of windows with guides", num_windows_with_guides)
            ]

            # Print the above statistics, with padding on the left
            # so that the statistic names are right-justified in a
            # column and the values line up, left-justified, in a column
            max_stat_name_len = max(len(name) for name, val in stat_display)
            for name, val in stat_display:
                pad_spaces = max_stat_name_len - len(name)
                name_padded = " "*pad_spaces + name + ":"
                print(name_padded, str(val))
