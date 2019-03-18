"""Methods for analyzing coverage of a design against a collection of sequences.
"""

import logging

from dxguidedesign.utils import guide

__author__ = 'Hayden Metsky <hayden@mit.edu>'

logger = logging.getLogger(__name__)


class Design:
    """Immutable object representing a design.

    This can represent just a collection of guide sequences, or a combination
    of primers and guides between those.
    """

    def __init__(self, guides, primers=None):
        """
        Args:
            guides: collection of guide sequences
            primers: if set, a tuple (l, r) where l is a collection of primers
                on the 5' end and r is a collection of primers on the 3' end
        """
        self.guides = guides
        self.primers = primers

        self.is_complete_target = primers is not None


class CoverageAnalyzer:
    """Methods to analyze coverage of a design.
    """

    def __init__(self, seqs, designs, guide_mismatches, primer_mismatches=None,
            allow_gu_pairs=True):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            guide_mismatches: number of mismatches to tolerate when determining
                whether a guide hybridizes to a target sequence
            primer_mismatches: number of mismatches to tolerate when determining
                whether a primer hybridizes to a target sequence (if not
                set, the designs are assumed to not contain primers)
            allow_gu_pairs: if True, tolerate G-U base pairs between a guide
                and target when computing whether a guide binds
        """
        self.seqs = seqs
        self.designs = designs
        self.guide_mismatches = guide_mismatches
        self.primer_mismatches = primer_mismatches
        self.allow_gu_pairs = allow_gu_pairs

    def find_binding_pos(self, target_seq, seq, mismatches, allow_gu_pairs):
        """Find if and where a sequence binds to in a target sequence.

        This takes a naive (but completely sensitive) approach: it simply
        slides across the target sequence, checking for binding at
        each position.

        Args:
            target_seq: target sequence against which to check for binding
            seq: guide or primer sequence to lookup
            mismatches: number of mismatches to tolerate when determining
                whether seq binds to target_seq at a position
            allow_gu_pairs: if True, tolerate G-U base pairs between
                seq and a subsequence of target_seq

        Returns:
            collection of start positions in target_seq to which seq binds;
            empty collection if seq does not bind anywhere in target_seq
        """
        bind_pos = set()
        for i in range(len(target_seq) - len(seq) + 1):
            target_subseq = target_seq[i:(i + len(seq))]
            if guide.guide_binds(seq, target_subseq, mismatches,
                    allow_gu_pairs):
                bind_pos.add(i)
        return bind_pos

    def seqs_where_guides_bind(self, guide_seqs):
        """Determine sequences to which guide binds.

        Args:
            guide_seqs: guide sequences to lookup

        Returns:
            collection of names of sequences in self.seqs to which a guide
            in guide_seqs binds
        """
        seqs_bound = set()
        for seq_name, target_seq in self.seqs.items():
            for guide_seq in guide_seqs:
                bind_pos = self.find_binding_pos(target_seq, guide_seq,
                        self.guide_mismatches, self.allow_gu_pairs)
                if len(bind_pos) > 0:
                    # guide_seq binds somewhere in target_seq
                    seqs_bound.add(seq_name)
                    break
        return seqs_bound

    def seqs_where_targets_bind(self, guide_seqs, primer_left_seqs,
           primer_right_seqs):
        """Determine sequences to which a target binds.

        Args:
            guide_seqs: guide sequences in the target
            primer_left_seqs: primers on the 5' end of the target
            primer_right_seqs: primers on the 3' end of the target

        Returns:
            collection of names of sequences in self.seqs to which
            some pair of primers and guide between them binds
        """
        seqs_bound = set()
        for seq_name, target_seq in self.seqs.items():
            guide_bind_pos = set()
            min_guide_len_at_pos = {}
            for guide_seq in guide_seqs:
                bind_pos = self.find_binding_pos(target_seq, guide_seq,
                        self.guide_mismatches, self.allow_gu_pairs)
                guide_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_guide_len_at_pos:
                        min_guide_len_at_pos[pos] = len(guide_seq)
                    else:
                        min_guide_len_at_pos[pos] = min(
                                min_guide_len_at_pos[pos], len(guide_seq))
            primer_left_bind_pos = set()
            min_primer_len_at_pos = {}
            for primer_seq in primer_left_seqs:
                bind_pos = self.find_binding_pos(target_seq, primer_seq,
                        self.primer_mismatches, False)
                primer_left_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_primer_len_at_pos:
                        min_primer_len_at_pos[pos] = len(primer_seq)
                    else:
                        min_primer_len_at_pos[pos] = min(
                                min_primer_len_at_pos[pos], len(primer_seq))
            primer_right_bind_pos = set()
            for primer_seq in primer_right_seqs:
                bind_pos = self.find_binding_pos(target_seq, primer_seq,
                        self.primer_mismatches, False)
                primer_right_bind_pos.update(bind_pos)

            # Determine if there exists some combination of (5' primer)/guide/
            # (3' primer) that binds to target_seq
            added_seq = False
            for pl_pos in primer_left_bind_pos:
                # Consider all primers that fall *after* (3' of) pl
                primer_len = min_primer_len_at_pos[pl_pos]
                for pr_pos in primer_right_bind_pos:
                    if pr_pos > pl_pos:
                        # Find if any guide falls in between pl and pr
                        for g_pos in guide_bind_pos:
                            guide_len = min_guide_len_at_pos[g_pos]
                            if (g_pos >= pl_pos + primer_len and
                                    pr_pos >= g_pos + guide_len):
                                seqs_bound.add(seq_name)
                                added_seq = True
                                break
                    if added_seq:
                        break
                if added_seq:
                    break

        return seqs_bound

    def seqs_bound_by_design(self, design):
        """Determine the sequences bound by a Design.

        Args:
            design: Design object

        Returns:
            collection of sequences that are bound by the guides (and
            possibly primers) in design
        """
        if design.is_complete_target:
            primers_left, primers_right = design.primers
            return self.seqs_where_targets_bind(design.guides,
                    primers_left, primers_right)
        else:
            return self.seqs_where_guides_bind(design.guides)

    def frac_of_seqs_bound(self):
        """Determine the fraction of sequences bound by each design.

        Returns:
            dict mapping design identifier (self.designs.keys()) to the
            fraction of all sequences that are bound by the design it
            represents
        """
        frac_bound = {}
        for design_id, design in self.designs.items():
            logger.info(("Computing fraction of sequences bound by design "
                "'%s'"), str(design_id))
            seqs_bound = self.seqs_bound_by_design(design)
            frac_bound[design_id] = float(len(seqs_bound)) / len(self.seqs)
        return frac_bound
