"""Functions for working with guides.
"""

import logging

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


FASTA_CODES = {'A': set(('A')),
               'T': set(('T')),
               'C': set(('C')),
               'G': set(('G')),
               'K': set(('G', 'T')),
               'M': set(('A', 'C')),
               'R': set(('A', 'G')),
               'Y': set(('C', 'T')),
               'S': set(('C', 'G')),
               'W': set(('A', 'T')),
               'B': set(('C', 'G', 'T')),
               'V': set(('A', 'C', 'G')),
               'H': set(('A', 'C', 'T')),
               'D': set(('A', 'G', 'T')),
               'N': set(('A', 'T', 'C', 'G'))}


def seq_mismatches(seq_a, seq_b):
    """Count number of mismatches between two sequences, tolerating ambiguity.

    Note that because 'N' usually signals missing data (and not necessarily
    ambiguity about what the genome actually is), this considers 'N' with
    any other base to be a mismatch.

    Args:
        seq_a: str of a sequence
        seq_b: str of a sequence, same length as seq_a

    Returns:
        number of mismatches between seq_a and seq_b, tolerating ambiguity
        between them (e.g., 'Y' and 'T' is not a mismatch) but considering
        'N' to be a mismatch (e.g., 'N' and 'T' is a mismatch)
    """
    assert len(seq_a) == len(seq_b)

    return sum(1 for i in range(len(seq_a))
               if (seq_a[i] == 'N' or seq_b[i] == 'N' or
                   not (FASTA_CODES[seq_a[i]] & FASTA_CODES[seq_b[i]])))


def guide_binds(guide_seq, target_seq, mismatches=0):
    """Determine whether a guide binds to a target sequence.

    This tolerates ambiguity and decides whether a guide binds based on
    whether its number of mismatches with the target sequence is within
    a threshold.

    Args:
        guide_seq: str of a guide sequence
        target_seq: str of a target sequence, same length as guide_seq
        mismatches: int giving threshold on number of mismatches for binding

    Returns:
        True iff the number of mismatches between guide_seq and target_seq
        is <= mismatches
    """
    return seq_mismatches(guide_seq, target_seq) <= mismatches
