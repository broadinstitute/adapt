"""Functions for working with oligos.
"""

import logging

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


# Store the unambiguous bases that make up each
# ambiguous base in the IUPAC notation
FASTA_CODES = {
    'A': set(('A')),
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
    'N': set(('A', 'T', 'C', 'G'))
}

COMPLEMENTS = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
    'K': 'M',
    'M': 'K',
    'R': 'Y',
    'Y': 'R',
    'S': 'S',
    'W': 'W',
    'B': 'V',
    'V': 'B',
    'H': 'D',
    'D': 'H',
    'N': 'N'
}

BINDS_EARLY_TERMINATE_SUBSTR_LEN = 5


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


def seq_mismatches_with_gu_pairs(oligo_seq, target_seq):
    """Count number of mismatches between an oligo and target, allowing G-U
    pairs, and tolerating ambiguity.

    As in seq_mismatches(..), this considers 'N' with any other base to
    be a mismatch.

    This also tolerates G-U base pairing:
    An RNA oligo with U can bind to target RNA with G, and vice-versa.
    Note that a oligo sequence that ends up being synthesized to RNA
    is the reverse-complement of the oligo sequence constructed here, so
    that it will hybridize to the target (here, the oligo sequences
    are designed to match the target). If the RNA oligo were to have U and
    the RNA target were to have G, then the oligo sequence here would be
    A and the target would be G. If the RNA oligo were to have G
    and the RNA target were to have U, then the oligo sequence here were
    would be C and the target would be T. Thus, to allow G-U pairing,
    we count a base X in the oligo sequence as matching a base Y in
    the target if either of the following is true:
      - X == 'A' and Y == 'G'
      - X == 'C' and Y == 'T'

    Unlike in seq_mismatches(..), it is important to know which sequence
    here represents the oligo and which represents the target, so we
    cannot just refer to them interchangeably as seq_a and seq_b.

    Args:
        oligo_seq: str of a oligo sequence
        target_seq: str of target sequence, same length as oligo_seq

    Returns:
        number of mismatches between oligo and target sequence, counting
        G-U pairs as matching, and tolerating ambiguity between them
    """
    assert len(oligo_seq) == len(target_seq)

    # Since this function is low-level and called so often, it
    # may be better to ensure it is efficient by keeping all
    # conditionals inside the sum(..), even if it is messy, rather
    # than defining and using subfunctions
    return sum(1 for i in range(len(oligo_seq)) if (
                   # mismatch if either guide or target is 'N'
                   (oligo_seq[i] == 'N' or target_seq[i] == 'N') or
                   # mismatch if not a match
                   not (
                       # both bases match
                       (FASTA_CODES[oligo_seq[i]] & FASTA_CODES[target_seq[i]]) or
                       # guide is 'A' and target is 'G'
                       ('A' in FASTA_CODES[oligo_seq[i]] and
                        'G' in FASTA_CODES[target_seq[i]]) or
                       # guide is 'C' and target is 'T'
                       ('C' in FASTA_CODES[oligo_seq[i]] and
                        'T' in FASTA_CODES[target_seq[i]])
                   )
               )
           )

def query_target_eq(query_seq, target_seq):
    """Determine if a query sequence equals a target.

    This tolerates 'N' in the query sequence but not in the target
    sequence. As a result, it is different than checking if
    seq_mismatches(query_seq, target_seq) == 0. For example,
    a query of 'AN' equals a target of 'AT' (but a query of 'AT'
    does not equal a target of 'AN'). The reason is the same
    as in seq_mismatches() -- 'N' in a target usually signals
    missing data rather than ambiguity.

    Args:
        query_seq: str of a query sequence
        target_seq: str of a target sequence, same length as
            query_seq

    Returns:
        True if query_seq equals target_seq, tolerating all
        non-N ambiguity codes in the query and target, and
        tolerating 'N' in the query; otherwise, False
    """
    assert len(query_seq) == len(target_seq)

    return sum(1 for i in range(len(query_seq))
               if target_seq[i] == 'N' or
               not (FASTA_CODES[query_seq[i]] & FASTA_CODES[target_seq[i]])) == 0


def binds(oligo_seq, target_seq, mismatches, allow_gu_pairs):
    """Determine whether a guide binds to a target sequence.

    This tolerates ambiguity and decides whether a guide binds based on
    whether its number of mismatches with the target sequence is within
    a threshold.

    If the target sequence contains a gap (and the guide sequence does
    not, as it should not), this decides that the guide does not bind.

    The calls to seq_mismatches() or seq_mismatches_with_gu_pairs() may
    be slow because they sum over all bases in oligo_seq and target_seq;
    in particular, the lookups in FASTA_CODES may take a while for each
    term in the sum. In many cases, the output of those functions (i.e.,
    the number of mismatches between oligo_seq and target_seq) will be
    far higher than the given threshold on mismatches. The guide will not
    bind, but we can determine this without having to sum over all bases.
    To speed up these cases (i.e., when oligo_seq and target_seq are very
    different), this function calls seq_mismatches() (or
    seq_mismatches_with_gu_pairs()) on just a substring of oligo_seq
    and target_seq. The substring is arbitrary: here, it is taken from
    the start of those sequences. If the number of mismatches between
    those substrings is already too high, we can terminate early and
    say that the guide does not bind.

    Args:
        oligo_seq: str of a guide sequence
        target_seq: str of a target sequence, same length as oligo_seq
        mismatches: int giving threshold on number of mismatches for binding
        allow_gu_pairs: if True, tolerate G-U base pairs when
            counting mismatches between oligo_seq and target_seq

    Returns:
        True iff the number of mismatches between oligo_seq and target_seq
        is <= mismatches
    """
    if '-' in target_seq:
      assert '-' not in oligo_seq
      return False

    if allow_gu_pairs:
        mismatches_fn = seq_mismatches_with_gu_pairs
    else:
        mismatches_fn = seq_mismatches

    # Determine whether to terminate early, using just a substring of
    # the two sequences
    if (mismatches < BINDS_EARLY_TERMINATE_SUBSTR_LEN and
            len(oligo_seq) > BINDS_EARLY_TERMINATE_SUBSTR_LEN and
            len(target_seq) > BINDS_EARLY_TERMINATE_SUBSTR_LEN):
        guide_subseq = oligo_seq[:BINDS_EARLY_TERMINATE_SUBSTR_LEN]
        target_subseq = target_seq[:BINDS_EARLY_TERMINATE_SUBSTR_LEN]
        m_substr = mismatches_fn(guide_subseq, target_subseq)
        if m_substr > mismatches:
            # Even in just the substring, there are too many mismatches for
            # the guide to bind
            return False

    # Compute the number of mismatches between oligo_seq and target_seq,
    # and compare against the given threshold
    m = mismatches_fn(oligo_seq, target_seq)
    return m <= mismatches


def gc_frac(oligo_seq):
    """Compute fraction of guide that is GC.

    Args:
        oligo_seq: string of guide sequence; must be all uppercase

    Returns:
        fraction of guide sequence that is G or C
    """
    gc = oligo_seq.count('G') + oligo_seq.count('C')
    return float(gc) / len(oligo_seq)


def overlap_in_seq(oligo_seqs, target_seq, mismatches, allow_gu_pairs):
    """Use a simple sliding strategy to find where guides bind in a sequence.

    This computes binding according to guide_binds().

    This uses a naive sliding strategy over target_seq, computing binding at
    every position. Since this is slow, this assumes target_seq is short (~100s
    of nt). Similarly, this returns a set of indices -- rather than ranges --
    which are easier to work with downstream even though they are less
    efficient.

    Args:
        oligo_seqs: list of str of guide sequences
        target_seq: str of a target sequence, at least the length of oligo_seq
        mismatches: int giving threshold on number of mismatches for binding
        allow_gu_pairs: if True, tolerate G-U base pairs when
            counting mismatches between oligo_seq and target_seq

    Returns:
        set of indices in target_seq to which oligo_seq binds
    """
    for oligo_seq in oligo_seqs:
        assert len(target_seq) >= len(oligo_seq)

    indices_bound = set()

    for i in range(0, len(target_seq) - len(oligo_seq) + 1):
        target_seq_at_i = target_seq[i:(i + len(oligo_seq))]
        for oligo_seq in oligo_seqs:
            binds = guide_binds(oligo_seq, target_seq_at_i, mismatches,
                    allow_gu_pairs)
            if binds:
                for j in range(0, len(oligo_seq)):
                    indices_bound.add(i + j)

    return indices_bound




# class Oligo:
#     """
#     """
#     def __init__(self, seq):
#         self.seq = seq

#     def equals_target(self, target_seq):
#         """Determine if an oligo equals a target.

#         This tolerates 'N' in the query sequence but not in the target
#         sequence. As a result, it is different than checking if
#         seq_mismatches(query_seq, target_seq) == 0. For example,
#         a query of 'AN' equals a target of 'AT' (but a query of 'AT'
#         does not equal a target of 'AN'). The reason is the same
#         as in seq_mismatches() -- 'N' in a target usually signals
#         missing data rather than ambiguity.

#         Args:
#             target_seq: str of a target sequence, same length as
#                 query_seq

#         Returns:
#             True if self.seq equals target_seq, tolerating all
#             non-N ambiguity codes in the query and target, and
#             tolerating 'N' in the query; otherwise, False
#         """
#         assert len(self.seq) == len(target_seq)

#         if 'N' in target_seq:
#             return False

#         return sum(1 for i in range(len(self.seq))
#                    if not (FASTA_CODES[self.seq[i]] &
#                            FASTA_CODES[target_seq[i]])) == 0

#     def seq_mismatches(self, other):
#         """Count number of mismatches between two sequences, tolerating ambiguity.

#         Note that because 'N' usually signals missing data (and not necessarily
#         ambiguity about what the genome actually is), this considers 'N' with
#         any other base to be a mismatch.

#         Args:
#             other: Oligo, where other.seq is the same length as self.seq

#         Returns:
#             number of mismatches between self and other, tolerating ambiguity
#             between them (e.g., 'Y' and 'T' is not a mismatch) but considering
#             'N' to be a mismatch (e.g., 'N' and 'T' is a mismatch)
#         """
#         assert len(self.seq) == len(other.seq)

#         return sum(1 for i in range(len(self.seq))
#                    if (self.seq[i] == 'N' or other.seq[i] == 'N' or
#                        not (FASTA_CODES[self.seq[i]] &
#                             FASTA_CODES[other.seq[i]])))

#     def seq_mismatches_with_gu_pairs(self, target_seq):
#         """Count number of mismatches between an oligo and target, allowing G-U
#         pairs, and tolerating ambiguity.

#         As in seq_mismatches(..), this considers 'N' with any other base to
#         be a mismatch.

#         This also tolerates G-U base pairing:
#         An RNA oligo with U can bind to target RNA with G, and vice-versa.
#         Note that a oligo sequence that ends up being synthesized to RNA
#         is the reverse-complement of the oligo sequence constructed here, so
#         that it will hybridize to the target (here, the oligo sequences
#         are designed to match the target). If the RNA oligo were to have U and
#         the RNA target were to have G, then the oligo sequence here would be
#         A and the target would be G. If the RNA oligo were to have G
#         and the RNA target were to have U, then the oligo sequence here were
#         would be C and the target would be T. Thus, to allow G-U pairing,
#         we count a base X in the oligo sequence as matching a base Y in
#         the target if either of the following is true:
#           - X == 'A' and Y == 'G'
#           - X == 'C' and Y == 'T'

#         Unlike in seq_mismatches(..), it is important to know which sequence
#         here represents the oligo and which represents the target, so we
#         cannot just refer to them interchangeably as seq_a and seq_b.

#         Args:
#             target_seq: str of target sequence, same length as self.seq

#         Returns:
#             number of mismatches between oligo and target sequence, counting
#             G-U pairs as matching, and tolerating ambiguity between them
#         """
#         assert len(self.seq) == len(target_seq)

#         # Since this function is low-level and called so often, it
#         # may be better to ensure it is efficient by keeping all
#         # conditionals inside the sum(..), even if it is messy, rather
#         # than defining and using subfunctions
#         return sum(1 for i in range(len(self.seq)) if (
#                        # mismatch if either oligo or target is 'N'
#                        (self.seq[i] == 'N' or target_seq[i] == 'N') or
#                        # mismatch if not a match
#                        not (
#                            # both bases match
#                            (FASTA_CODES[self.seq[i]] & FASTA_CODES[target_seq[i]]) or
#                            # oligo is 'A' and target is 'G'
#                            ('A' in FASTA_CODES[self.seq[i]] and
#                             'G' in FASTA_CODES[target_seq[i]]) or
#                            # oligo is 'C' and target is 'T'
#                            ('C' in FASTA_CODES[self.seq[i]] and
#                             'T' in FASTA_CODES[target_seq[i]])
#                        )
#                    )
#                )

#     def binds(self, target_seq, mismatches, allow_gu_pairs):
#         """Determine whether a oligo binds to a target sequence.

#         Args:
#             self.seq: str of a oligo sequence
#             target_seq: str of a target sequence, same length as self.seq
#             mismatches: int giving threshold on number of mismatches for binding
#             allow_gu_pairs: if True, tolerate G-U base pairs when
#                 counting mismatches between self.seq and target_seq

#         Returns:
#             True iff the number of mismatches between self.seq and target_seq
#             is <= mismatches
#         """
#         raise NotImplementedError("Subclasses of Oligo must implement binds()")

#     def gc_frac(self):
#         """Compute fraction of oligo that is GC.

#         Returns:
#             fraction of oligo sequence that is G or C
#         """
#         gc = self.seq.count('G') + self.seq.count('C')
#         return float(gc) / len(self.seq)

#     @staticmethod
#     def overlap_in_seq(oligo_seqs, target_seq, mismatches, allow_gu_pairs):
#         """Use a simple sliding strategy to find where oligos bind in a sequence.

#         This computes binding according to binds().

#         This uses a naive sliding strategy over target_seq, computing binding at
#         every position. Since this is slow, this assumes target_seq is short (~100s
#         of nt). Similarly, this returns a set of indices -- rather than ranges --
#         which are easier to work with downstream even though they are less
#         efficient.

#         Args:
#             oligo_seqs: list of Oligo objects
#             target_seq: str of a target sequence, at least the length of oligo_seq
#             mismatches: int giving threshold on number of mismatches for binding
#             allow_gu_pairs: if True, tolerate G-U base pairs when
#                 counting mismatches between oligo_seq and target_seq

#         Returns:
#             set of indices in target_seq to which oligo_seq binds
#         """
#         for oligo_seq in oligo_seqs:
#             assert len(target_seq) >= len(oligo_seq)

#         indices_bound = set()

#         for i in range(0, len(target_seq) - len(oligo_seq) + 1):
#             target_seq_at_i = target_seq[i:(i + len(oligo_seq))]
#             for oligo_seq in oligo_seqs:
#                 binds = oligo_seq.binds(oligo_seq, target_seq_at_i, mismatches,
#                         allow_gu_pairs)
#                 if binds:
#                     for j in range(0, len(oligo_seq)):
#                         indices_bound.add(i + j)

#         return indices_bound
