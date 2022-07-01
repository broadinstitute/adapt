"""Methods for searching for optimal primers through a genome.
"""

import logging

from adapt.utils import search
from adapt.utils import oligo
from adapt import alignment

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


class PrimerResult:
    """Store results of a primer set at a site."""

    def __init__(self, start, num_primers, primer_length, frac_bound,
            primers_in_set, obj_value):
        """
        Args:
            start: start position of the primer
            num_primers: number of primers designed at the site
            primer_length: length of each primer
            frac_bound: total fraction of all sequences bound by the primers
            primers_in_set: set of primers that achieve the best obj_value at
                a site
            obj_value: value to use to compare primer sets to each other
        """
        self.start = start
        self.num_primers = num_primers
        self.primer_length = primer_length
        self.frac_bound = frac_bound
        self.primers_in_set = primers_in_set
        self.obj_value = obj_value

    def overlaps(self, other, expand=0):
        """Determine if self overlaps other.

        Args:
            other: PrimerResult object
            expand: tests overlap within +/- EXPAND nt of other

        Returns:
            True iff self overlaps other
        """
        return ((self.start - expand <= other.start < (self.start +
                    self.primer_length + expand)) or
                (other.start - expand <= self.start < (other.start +
                    other.primer_length + expand)))

    def overlaps_range(self, start, end):
        """Determine if self overlaps a range.

        Args:
            start: start of range (inclusive)
            end: end of range (exclusive)

        Returns:
            True iff self overlaps (start, end)
        """
        return ((self.start < end) and (start < self.start + self.primer_length))

    def __str__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_set))

    def __repr__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_set))

    def __eq__(self, other):
        """Determine equality of self and other.

        Args:
            other: an object of PrimerResult

        Returns:
            True iff self is identical to other
        """
        return (self.start == other.start and
                self.num_primers == other.num_primers and
                self.primer_length == other.primer_length and
                self.frac_bound == other.frac_bound and
                self.primers_in_set == other.primers_in_set)


class PrimerSearcher(search.OligoSearcher):
    """Methods to search for primers over a genome.

    This looks for oligos (here, primers) within each window of size w
    where w is the maximum length of a primer.

    This is a base class, with subclasses defining methods depending on the
    oligo. It should not be used without subclassing, as it does not define
    all the positional arguments necessary for search.OligoSearcher.
    """

    def __init__(self, primer_gc_content_bounds=None, pre_filter_fns=None,
            **kwargs):
        """
        Args:
            primer_gc_content_bounds: a tuple (lo, hi) such that this only
                yields sites where all primers have a GC content fraction in
                [lo, hi]; or None for no bounds
            pre_filter_fns: if set, the value of this argument is a list
                of functions f(x) such that this will only construct a primer x
                for which each f(x) is True
        """
        pre_filter_fns = pre_filter_fns if pre_filter_fns is not None else []
        if primer_gc_content_bounds:
            lo, hi = primer_gc_content_bounds
            assert lo <= hi
            def check_gc_content(primer_seq):
                """Determine whether primer meets bounds on GC content.

                Args:
                    primer_seq: primer sequence

                Returns:
                    True/False indicating whether all primers meet the bounds on
                    GC content
                """
                gc_frac = oligo.gc_frac(primer_seq)
                return (gc_frac >= lo and gc_frac <= hi)
            pre_filter_fns = pre_filter_fns + [check_gc_content]

        super().__init__(pre_filter_fns=pre_filter_fns, **kwargs)


    def find_primers(self, max_at_site=None):
        """Find primers across the alignment.

        Args:
            max_at_site: only yield sites that have <= MAX_AT_SITE
                primers at a site; or None for no limit

        Yields:
            tuple at each site in the alignment, consisting of the
            following values, in order:
              1) start position of the primer
              2) number of primers designed at the site
              3) total fraction of all sequences bound by the primers
              4) set of primers that achieve the desired coverage
                 and is minimal at the site
            They are given in sorted order, by position in the
            alignment
        """
        window_size = self.max_oligo_length
        for primer_set in self._find_oligos_for_each_window(
                window_size, hide_warnings=True):
            start, end, primers_in_set = primer_set
            num_primers = len(primers_in_set)
            if self.obj_type == 'min':
                frac_bound = self.total_frac_bound(primers_in_set)
                obj_value = self.obj_value(primers_in_set)
            else:
                frac_bound = self.total_frac_bound(start, end, primers_in_set)
                obj_value = self.obj_value(start, end, primers_in_set)

            # Check constraints
            if max_at_site is not None and num_primers > max_at_site:
                continue

            yield PrimerResult(
                start, num_primers, window_size,
                frac_bound, primers_in_set, obj_value)


class PrimerSearcherMaximizeActivity(search.OligoSearcherMaximizeActivity,
        PrimerSearcher):
    """Methods to search for primers over a genome using activity models

    'Activity' here is defined loosely by whatever the predictor's output is.
    (For example, using predict_activity.TmPredictor, the activity would be
    defined as the negative of the difference between the calculated and ideal
    melting temperature)
    """

    def __init__(self, aln, min_primer_length, max_primer_length,
            soft_constraint, hard_constraint, penalty_strength,
            missing_data_params, **kwargs):
        """
        Args:
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design oligos overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
            soft_constraint: number of oligos for the soft constraint
            hard_constraint: number of oligos for the hard constraint
            penalty_strength: coefficient in front of the soft penalty term
                (i.e., its importance relative to expected activity)
        """
        super().__init__(aln=aln, min_oligo_length=min_primer_length,
            max_oligo_length=max_primer_length, soft_constraint=soft_constraint,
            hard_constraint=hard_constraint, penalty_strength=penalty_strength,
            missing_data_params=missing_data_params, **kwargs)


class PrimerSearcherMinimizePrimers(search.OligoSearcherMinimizeNumber,
        PrimerSearcher):
    """Methods to search for primers over a genome.

    This looks for oligos (here, primers) within each window of size w
    where w is the length of a primer.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable primers.
    """

    def __init__(self, aln, primer_length, mismatches, cover_frac,
            missing_data_params, primer_gc_content_bounds=None, **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            primer_length: length of the primer to construct
            mismatches: threshold on number of mismatches for determining whether
                a primer would hybridize to a target sequence
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a primer; see seq_groups
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m)), where m is
                the median fraction of sequences with missing data over the
                alignment
            primer_gc_content_bounds: a tuple (lo, hi) such that this only
                yields sites where all primers have a GC content fraction in
                [lo, hi]; or None for no bounds
        """

        super().__init__(aln=aln, min_oligo_length=primer_length,
            max_oligo_length=primer_length, cover_frac=cover_frac,
            mismatches=mismatches, missing_data_params=missing_data_params,
            primer_gc_content_bounds=primer_gc_content_bounds, **kwargs)
