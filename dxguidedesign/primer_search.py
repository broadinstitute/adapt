"""Methods for searching for optimal primers through a genome.

This makes heavy use of the guide_search module.
"""

import logging

from dxguidedesign import guide_search

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


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
                 missing_data_params):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            primer_length: length of the primer to construct
            mismatches: threshold on number of mismatches for determining whether
                a primer would hybridize to a target sequence
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a primer
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m), where m is
                the median fraction of sequences with missing data over the
                alignment
        """
        super().__init__(aln, primer_length, mismatches,
                         cover_frac, missing_data_params)

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
              3) total fraction of all sequences bound by the primer
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
                yield (start, num_primers, frac_bound, primers_in_cover) 
