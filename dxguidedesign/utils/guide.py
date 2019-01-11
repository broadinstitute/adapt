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


def seq_mismatches_with_gu_pairs(guide_seq, target_seq):
    """Count number of mismatches between a guide and target, allowing G-U
    pairs, and tolerating ambiguity.

    As in seq_mismatches(..), this considers 'N' with any other base to
    be a mismatch.

    This also tolerates G-U base pairing:
    An RNA guide with U can bind to target RNA with G, and vice-versa.
    Note that a guide sequence that ends up being synthesized to RNA
    is the reverse-complement of the guide sequence constructed here, so
    that it will hybridize to the target (here, the guide sequences
    are designed to match the target). If the RNA guide were to have U and
    the RNA target were to have G, then the guide sequence here would be
    A and the target would be G. If the RNA guide were to have G
    and the RNA target were to have U, then the guide sequence here were
    would be C and the target would be T. Thus, to allow G-U pairing,
    we count a base X in the guide sequence as matching a base Y in
    the target if either of the following is true:
      - X == 'A' and Y == 'G'
      - X == 'C' and Y == 'T'

    Unlike in seq_mismatches(..), it is important to know which sequence
    here represents the guide and which represents the target, so we
    cannot just refer to them interchangeably as seq_a and seq_b.

    Args:
        guide_seq: str of a guide sequence
        target_seq: str of target sequence, same length as guide_seq

    Returns:
        number of mismatches between guide and target sequence, counting
        G-U pairs as matching, and tolerating ambiguity between them
    """
    assert len(guide_seq) == len(target_seq)

    # Since this function is low-level and called so often, it
    # may be better to ensure it is efficient by keeping all
    # conditionals inside the sum(..), even if it is messy, rather
    # than defining and using subfunctions
    return sum(1 for i in range(len(guide_seq)) if (
                   # mismatch if either guide or target is 'N'
                   (guide_seq[i] == 'N' or target_seq[i] == 'N') or
                   # mismatch if not a match
                   not (
                       # both bases match
                       (FASTA_CODES[guide_seq[i]] & FASTA_CODES[target_seq[i]]) or
                       # guide is 'A' and target is 'G'
                       ('A' in FASTA_CODES[guide_seq[i]] and
                        'G' in FASTA_CODES[target_seq[i]]) or
                       # guide is 'C' and target is 'T'
                       ('C' in FASTA_CODES[guide_seq[i]] and
                        'T' in FASTA_CODES[target_seq[i]])
                   )
               )
           )


def guide_binds(guide_seq, target_seq, mismatches, allow_gu_pairs):
    """Determine whether a guide binds to a target sequence.

    This tolerates ambiguity and decides whether a guide binds based on
    whether its number of mismatches with the target sequence is within
    a threshold.

    If the target sequence contains a gap (and the guide sequence does
    not, as it should not), this decides that the guide does not bind.

    Args:
        guide_seq: str of a guide sequence
        target_seq: str of a target sequence, same length as guide_seq
        mismatches: int giving threshold on number of mismatches for binding
        allow_gu_pairs: if True, tolerate G-U base pairs when
            counting mismatches between guide_seq and target_seq

    Returns:
        True iff the number of mismatches between guide_seq and target_seq
        is <= mismatches
    """
    if '-' in target_seq:
      assert '-' not in guide_seq
      return False

    if allow_gu_pairs:
        m = seq_mismatches_with_gu_pairs(guide_seq, target_seq)
    else:
        m = seq_mismatches(guide_seq, target_seq)
    return m <= mismatches
