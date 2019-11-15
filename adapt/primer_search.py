"""Methods for searching for optimal primers through a genome.

This makes heavy use of the guide_search module.
"""

import logging

from adapt import guide_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class PrimerResult:
    """Store results of a primer cover at a site."""

    def __init__(self, start, num_primers, primer_length,
                 frac_bound, primers_in_cover):
        """
        Args:
            start: start position of the primer
            num_primers: number of primers designed at the site
            primer_length: length of each primer
            frac_bound: total fraction of all sequences bound by the primers
            primers_in_cover: set of primers that achieve the desired coverage
                 and is minimal at the site
        """
        self.start = start
        self.num_primers = num_primers
        self.primer_length = primer_length
        self.frac_bound = frac_bound
        self.primers_in_cover = primers_in_cover

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

    def __str__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_cover))

    def __repr__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_cover))

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
                self.primers_in_cover == other.primers_in_cover)


class PrimerSearcher(guide_search.GuideSearcher):
    """Methods to search for primers over a genome.

    This is a special case of guide_search.GuideSearcher; thus, it
    is a subclass of guide_search.GuideSearcher. This effectively
    looks for guides (here, primers) within each window of size w
    where w is the length of a primer.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable primers.
    """

    def __init__(self, aln, primer_length, mismatches, cover_frac,
                 missing_data_params, seq_groups=None):
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
            seq_groups: dict that maps group ID to collection of sequences in
                that group. If set, cover_frac must also be a dict that maps
                group ID to the fraction of sequences in that group that
                must be 'captured' by a primer. If None, then do not divide
                the sequences into groups.
        """
        super().__init__(aln, primer_length, mismatches,
                         cover_frac, missing_data_params,
                         seq_groups=seq_groups,
                         allow_gu_pairs=False,
                         do_not_memoize_guides=True)

    def seqs_bound_by_primers(self, primers):
        """Determine the sequences in the alignment bound by the primers.

        Args:
            primers: collection of str representing primer sequences

        Returns:
            set of sequence identifiers (index in alignment) bound by
            a primer
        """
        return super()._seqs_bound_by_guides(primers)

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
        window_size = self.guide_length # primer length
        for cover in self._find_guides_that_cover_for_each_window(
                window_size, hide_warnings=True):
            start, end, num_primers, score, frac_bound, primers_in_cover = cover
            if max_at_site is None or num_primers <= max_at_site:
                yield PrimerResult(
                    start, num_primers, window_size,
                    frac_bound, primers_in_cover)