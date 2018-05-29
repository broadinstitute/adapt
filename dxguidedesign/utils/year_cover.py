"""Functions for constructing partial cover input set by year.
"""

from collections import defaultdict
import logging
import re

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def read_years(fn):
    """Read years for a list of sequences.

    Args:
        fn: path to file with two columns; col 1 gives name of a sequence
            (fasta header, without '>') and col 2 gives a 4-digit year of the
            sequence

    Returns:
        dict mapping each year to set of sequence names with that year
    """
    year_pattern = re.compile('^(\d{4})$')
    d = defaultdict(set)

    seqs_seen = set()
    with open(fn) as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            ls = line.split('\t')
            assert len(ls) == 2
            seq_name, year = ls

            if not year_pattern.match(year):
                raise ValueError(("All values in 2nd column must be 4-digit "
                    "years"))
            if seq_name in seqs_seen:
                raise ValueError(("Sequence '%s' appears more than once" %
                    seq_name))

            year = int(year)
            seqs_seen.add(seq_name)
            d[year].add(seq_name)
    return d


def construct_partial_covers(years, year_with_highest_cover,
                             highest_cover_frac, decay):
    """Construct desired partial cover fractions for each year.

    This specifies achieving a coverage of highest_cover_frac
    for all years >= year_with_full_cover, and every preceding year gets
    a desired coverage that decreases according to decay.

    Args:
        years: collection of years for which to compute partial covers
        year_with_highest_cover: give the highest coverage (cover most number
            of sequences) for all years >= YEAR_WITH_HIGHEST_COVER
        highest_cover_frac: float in (0,1]; the desired partial cover to
            assign to all years >= YEAR_WITH_HIGHEST_COVER
        decay: float in (0, 1); year n is assigned a desired partial cover of
            DECAY*(partial cover of year n+1)

    Returns:
        dict mapping each year in years to a desired partial cover
    """
    if highest_cover_frac <= 0 or highest_cover_frac > 1:
        raise ValueError("highest_cover_frac must be in (0,1]")
    if decay <= 0 or decay >= 1:
        raise ValueError("decay must be in (0,1)")

    d = {}
    for year in set(years):
        if year >= year_with_highest_cover:
            d[year] = highest_cover_frac
        else:
            total_decay = decay**(year_with_highest_cover - year)
            d[year] = highest_cover_frac * total_decay
    return d

