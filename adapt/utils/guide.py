"""Functions for working with guides.
"""

import logging

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


# Store the unambiguous bases that make up each
# ambiguous base in the IUPAC notation
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

# Specify the substring length to use for checking whether
# to terminate early in guide_binds()
GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN = 5


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
    return num_mismatches == 0


def guide_binds(guide_seq, target_seq, mismatches, allow_gu_pairs):
    """Determine whether a guide binds to a target sequence.

    This tolerates ambiguity and decides whether a guide binds based on
    whether its number of mismatches with the target sequence is within
    a threshold.

    If the target sequence contains a gap (and the guide sequence does
    not, as it should not), this decides that the guide does not bind.

    The calls to seq_mismatches() or seq_mismatches_with_gu_pairs() may
    be slow because they sum over all bases in guide_seq and target_seq;
    in particular, the lookups in FASTA_CODES may take a while for each
    term in the sum. In many cases, the output of those functions (i.e.,
    the number of mismatches between guide_seq and target_seq) will be
    far higher than the given threshold on mismatches. The guide will not
    bind, but we can determine this without having to sum over all bases.
    To speed up these cases (i.e., when guide_seq and target_seq are very
    different), this function calls seq_mismatches() (or
    seq_mismatches_with_gu_pairs()) on just a substring of guide_seq
    and target_seq. The substring is arbitrary: here, it is taken from
    the start of those sequences. If the number of mismatches between
    those substrings is already too high, we can terminate early and
    say that the guide does not bind.

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
        mismatches_fn = seq_mismatches_with_gu_pairs
    else:
        mismatches_fn = seq_mismatches

    # Determine whether to terminate early, using just a substring of
    # the two sequences
    if (mismatches < GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN and
            len(guide_seq) > GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN and
            len(target_seq) > GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN):
        guide_subseq = guide_seq[:GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN]
        target_subseq = target_seq[:GUIDE_BINDS_EARLY_TERMINATE_SUBSTR_LEN]
        m_substr = mismatches_fn(guide_subseq, target_subseq)
        if m_substr > mismatches:
            # Even in just the substring, there are too many mismatches for
            # the guide to bind
            return False

    # Compute the number of mismatches between guide_seq and target_seq,
    # and compare against the given threshold
    m = mismatches_fn(guide_seq, target_seq)
    return m <= mismatches


def gc_frac(guide_seq):
    """Compute fraction of guide that is GC.

    Args:
        guide_seq: string of guide sequence; must be all uppercase

    Returns:
        fraction of guide sequence that is G or C
    """
    gc = guide_seq.count('G') + guide_seq.count('C')
    return float(gc) / len(guide_seq)


def guide_overlap_in_seq(guide_seqs, target_seq, mismatches, allow_gu_pairs):
    """Use a simple sliding strategy to find where guides bind in a sequence.

    This computes binding according to guide_binds().

    This uses a naive sliding strategy over target_seq, computing binding at
    every position. Since this is slow, this assumes target_seq is short (~100s
    of nt). Similarly, this returns a set of indices -- rather than ranges --
    which are easier to work with downstream even though they are less
    efficient.

    Args:
        guide_seqs: list of str of guide sequences
        target_seq: str of a target sequence, at least the length of guide_seq
        mismatches: int giving threshold on number of mismatches for binding
        allow_gu_pairs: if True, tolerate G-U base pairs when
            counting mismatches between guide_seq and target_seq
    
    Returns:
        set of indices in target_seq to which guide_seq binds
    """
    for guide_seq in guide_seqs:
        assert len(target_seq) >= len(guide_seq)

    indices_bound = set()

    for i in range(0, len(target_seq) - len(guide_seq) + 1):
        target_seq_at_i = target_seq[i:(i + len(guide_seq))]
        for guide_seq in guide_seqs:
            binds = guide_binds(guide_seq, target_seq_at_i, mismatches,
                    allow_gu_pairs)
            if binds:
                for j in range(0, len(guide_seq)):
                    indices_bound.add(i + j)

    return indices_bound
