"""Functions for compressing collections of sequence indices.

Many other methods work on collections of sequence indices (represented by
integers) that are mostly contiguous -- i.e., span the full set of sequences
with some breaks throughout. This module contains some functions to help
represent these more efficiently.
"""

import itertools
import logging

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


def compress_mostly_contiguous(idx):
    """Collapse mostly contiguous list of integers into ranges.

    Args:
        idx: collection of indices (integers)

    Returns:
        set of tuples t=(t_i, t_j) such that all indices in the range
        [t_i, t_j) (i.e., exclusive) are in idx, and all indices in
        idx are represented by the tuples
    """
    idx_sorted = sorted(idx)

    # Use itertools.groupby() to obtain a group for each collection of
    # contiguous indices; if we key (group) on k = (value - position in the
    # list), then each distinct group of contiguous indices will have
    # the same group
    idx_ranges = []
    idx_grouped = itertools.groupby(
            idx_sorted,
            lambda i, c=itertools.count(): i - next(c))
    for _, idx_in_group in idx_grouped:
        idx_in_group = list(idx_in_group)
        first_idx, last_idx = idx_in_group[0], idx_in_group[-1]
        r = (first_idx, last_idx + 1)
        idx_ranges += [r]
    return set(idx_ranges)


def decompress_ranges(idx_ranges):
    """Un-collapse collection of ranges into a collection of all indices.

    Args:
        idx_ranges: collection of index ranges (as output by
            compress_mostly_contiguous())

    Returns:
        collection c of indices such that all indices in the range
        [t_i, t_j), for each tuple (t_i, t_j) in idx_ranges, are
        in c
    """
    idx = set()
    for r in idx_ranges:
        for i in range(r[0], r[1]):
            idx.add(i)
    return idx
