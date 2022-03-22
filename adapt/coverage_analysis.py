"""Methods for analyzing coverage of a design against a collection of sequences.
"""

from collections import defaultdict
import logging
import math

import numpy as np

from adapt.utils import guide

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya Pillai <ppillai@broadinstitute.org>'

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

    def __init__(self, seqs, designs, primer_mismatches=None,
            primer_terminal_mismatches=None, bases_from_terminal=5,
            fully_sensitive=False, max_target_length=None):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            primer_mismatches: number of mismatches to tolerate when determining
                whether a primer hybridizes to a target sequence (if not
                set, the designs are assumed to not contain primers)
            primer_terminal_mismatches: number of mismatches to tolerate in
                the bases_from_terminal bases from the 3' end when determining
                whether a primer hybridizes to a target sequence (if not
                set, don't consider terminal mismatches)
            bases_from_terminal: number of bases from 3' end in which to check
                for primer_terminal_mismatches (defaults to 5, unused if
                primer_terminal_mismatches is not set)
            fully_sensitive: if True, use a fully sensitive lookup of
                primers and guides (slower)
            max_target_length: the maximum length a target can be for it to
                be amplified; if None, no maximum length (i.e. targets of
                any length will be considered amplifiable). Does nothing if
                only a guide is being considered.
        """
        self.seqs = seqs
        self.designs = designs
        self.primer_mismatches = primer_mismatches
        self.primer_terminal_mismatches = primer_terminal_mismatches
        self.bases_from_terminal = bases_from_terminal
        self.seqs_are_indexed = False
        self.seqs_index_k = None
        self.fully_sensitive = fully_sensitive
        self.max_target_length = max_target_length

    def _index_seqs(self, k=6, stride_by_k=False, index_only=None):
        """Construct index of seqs for faster lookups of binding positions.

        This constructs a dictionary d. For key 'name' in self.seqs,
        d[name][kmer] provides a collection of positions in self.seqs[name]
        that contain kmer.

        Args:
            k: length of k-mer to index
            stride_by_k: if False, index the position of every single k-mer
                in each sequence; if True, only index k-mers while striding
                by k (i.e., index every k'th k-mer). Enabling this can
                save memory but may make it less likely to find a hit
            index_only: if set, index only this target sequence
        """
        if index_only is None:
            logger.info(("Indexing all sequences"))
        self.seqs_index_k = k
        self.seqs_index = {}
        if index_only is None:
            seqs_to_index = self.seqs.keys()
        else:
            seqs_to_index = [index_only]
        for name in seqs_to_index:
            seq = self.seqs[name]
            self.seqs_index[name] = defaultdict(set)
            i = 0
            while i < len(seq) - k + 1:
                kmer = seq[i:(i+k)]
                self.seqs_index[name][kmer].add(i)
                if stride_by_k:
                    i += k
                else:
                    i += 1
        self.seqs_are_indexed = True
        if index_only is None:
            logger.info(("Done indexing all sequences"))

    def _clear_index(self):
        """Clear the index of sequences.
        """
        self.seqs_index.clear()
        self.seqs_are_indexed = False

    def evaluate_pos_by_terminal_mismatches_and_total_mismatches(self, seq,
            target_seq, pos, mismatches, terminal_mismatches,
            bases_from_terminal, left, allow_gu_pairs=False, save=None):
        """Evaluate binding with a mismatch model.

        Args:
            seq: guide or primer sequence to lookup
            target_seq: target sequence against which to check
            pos: list of positions indicating subsequence in target_seq
                to check for binding of seq
            mismatches: number of mismatches to tolerate when determining
                whether seq binds to target_seq at a position
            terminal_mismatches: number of mismatches to tolerate in the
                bases_from_terminal bases when determining whether seq binds to
                target_seq at a position
            bases_from_terminal: number of bases from 3' end in which to check
                for mismatches
            left: if True, consider the 3' end as the right end (and so check
                the last bases_from_terminal bases for mismatches); else,
                consider the 3' end as the left end (and so check the first
                bases_from_terminal bases for mismatches)
            allow_gu_pairs: if True, tolerate G-U base pairs between
                seq and a subsequence of target_seq (defaults to False)
            save: if set, save computed mismatches and terminal mismatches in
                this dictionary

        Returns:
            set of positions in pos where seq binds to target_seq
        """
        if save is not None:
            if allow_gu_pairs:
                mismatches_fn = guide.seq_mismatches_with_gu_pairs
            else:
                mismatches_fn = guide.seq_mismatches

        start_shift = len(seq)-bases_from_terminal if left else 0
        end_shift = len(seq) if left else bases_from_terminal
        termseq = seq[start_shift:end_shift]

        bind_pos = set()
        for i in pos:
            target_termseq = target_seq[(i+start_shift):(i+end_shift)]
            if guide.guide_binds(termseq, target_termseq, terminal_mismatches,
                    allow_gu_pairs):
                target_subseq = target_seq[i:(i + len(seq))]
                if guide.guide_binds(seq, target_subseq, mismatches,
                        allow_gu_pairs):
                    bind_pos.add(i)
                    if save is not None:
                        save[i] = (mismatches_fn(seq, target_subseq),
                                   mismatches_fn(termseq, target_termseq))
        return bind_pos

    def evaluate_pos_by_mismatches(self, seq, target_seq, pos,
            mismatches, allow_gu_pairs, save=None):
        """Evaluate binding with a mismatch model.

        Args:
            seq: guide or primer sequence to lookup
            target_seq: target sequence against which to check
            pos: list of positions indicating subsequence in target_seq
                to check for binding of seq
            mismatches: number of mismatches to tolerate when determining
                whether seq binds to target_seq at a position
            allow_gu_pairs: if True, tolerate G-U base pairs between
                seq and a subsequence of target_seq
            save: if set, save computed mismatches in this dictionary

        Returns:
            set of positions in pos where seq binds to target_seq
        """
        if save is not None:
            if allow_gu_pairs:
                mismatches_fn = guide.seq_mismatches_with_gu_pairs
            else:
                mismatches_fn = guide.seq_mismatches

        bind_pos = set()
        for i in pos:
            target_subseq = target_seq[i:(i + len(seq))]
            if guide.guide_binds(seq, target_subseq, mismatches,
                    allow_gu_pairs):
                bind_pos.add(i)
                if save is not None:
                    save[i] = (mismatches_fn(seq, target_subseq), )
        return bind_pos

    def guide_bind_fn(self, seq, target_seq, pos):
        """Determine whether guide detects target at positions.

        This should be implemented in a subclass.
        """
        raise NotImplementedError()

    def find_binding_pos(self, target_seq_name, seq, bind_fn,
            save=None):
        """Find if and where a sequence binds to in a target sequence.

        This takes a naive (but fully sensitive) approach: it simply
        slides across the target sequence, checking for binding at
        each position.

        This can also use a precomputed index of the target sequences to
        speed up finding binding positions. If there are hits from this
        index, it may only report these (depending on allow_not_all_positions).
        But if there are no hits from the index, this will also fall
        back on the naive sliding approach.

        Args:
            target_seq_name: name of target sequence against which to check
                (target sequence is self.seqs[target_seq_name])
            seq: guide or primer sequence to lookup
            bind_fn: function accepting (seq, target_seq, pos) and outputs
                positions in pos to which seq binds to target_seq
            save: if set, pass along to guide_fn to save computed
                activities

        Returns:
            collection of start positions in target_seq to which seq binds;
            empty collection if seq does not bind anywhere in target_seq
        """
        if self.fully_sensitive:
            # Use a naive, slow sliding approach
            allow_not_all_positions = False
            allow_not_fully_sensitive = False
        else:
            # Allow the returned collection of positions to not include *all*
            # positions that seq binds to
            allow_not_all_positions = True
            # Allow returning no positions based only on k-mer lookups in the
            # index (i.e., without performing a slow sliding approach)
            allow_not_fully_sensitive = True

        # Only keep target_seq_name in the index
        if not self.seqs_are_indexed:
            self._index_seqs(index_only=target_seq_name)
        else:
            if target_seq_name not in self.seqs_index:
                self._clear_index()
                self._index_seqs(index_only=target_seq_name)

        def bind_fn_with_save(seq, target_seq, pos):
            if save is None:
                return bind_fn(seq, target_seq, pos)
            else:
                return bind_fn(seq, target_seq, pos, save=save)

        target_seq = self.seqs[target_seq_name]

        bind_pos = set()
        if allow_not_all_positions:
            # Use the index of the target sequences to find potential hits,
            # and only report these, but looking up every k-mer in seq
            k = self.seqs_index_k
            index = self.seqs_index[target_seq_name]
            for j in range(len(seq) - k + 1):
                kmer = seq[j:(j+k)]
                for kmer_start in index[kmer]:
                    if kmer_start - j in bind_pos:
                        # This is already a binding positon; don't
                        # bother checking it again
                        continue
                    # The k-mer starts at kmer_start, so we have to look
                    # backwards to find the potential start of seq within
                    # target_seq
                    if (kmer_start - j >= 0 and
                            kmer_start - j + len(seq) <= len(target_seq)):
                        pos_to_evaluate = [kmer_start - j]
                        bind_pos.update(bind_fn_with_save(seq, target_seq,
                            pos_to_evaluate))
            if len(bind_pos) > 0:
                # We have found at least 1 binding position; this may
                # not be all but we do not need to find all
                return bind_pos
            if allow_not_fully_sensitive:
                # Return bind_pos (an empty set) even without performing
                # the fully sensitive approach
                return bind_pos
            # If len(bind_pos) == 0, then do the fully sensitive naive
            # sliding approach

        pos_to_evaluate = list(range(len(target_seq) - len(seq) + 1))
        bind_pos = bind_fn_with_save(seq, target_seq, pos_to_evaluate)
        return bind_pos

    def scores_where_oligo_binds(self, oligo_seqs, bind_fn, obj_type='min'):
        """Determine binding score for oligo per target sequence.
        'Binding score' is either the number of mismatches with the target or
        a predicted activity

        Args:
            oligo_seqs: oligo sequences
            bind_fn: function that calculates a 'binding score';
                takes in seq, target_seq, pos, save
            obj_type: 'min' (default) if the binding score should be minimized;
                'max' if the binding score should be maximized

        Returns:
            dict {target seq name: [tuple of binding scores,
                                    best oligo sequence,
                                    start position]}
        """
        target_scores = {}
        if obj_type == 'min':
            compare = lambda a,b: a<b
            default_oligo = [(math.inf, ), None, None]
        elif obj_type == 'max':
            compare = lambda a,b: a>b
            default_oligo = [(0, ), None, None]
        else:
            raise ValueError("Objective type must be either 'min' or 'max'")
        for seq_name in self.seqs:
            target_scores[seq_name] = default_oligo
            for oligo_seq in oligo_seqs:
                scores = {}
                bind_pos = self.find_binding_pos(seq_name, oligo_seq, bind_fn,
                    save=scores)
                if len(bind_pos) > 0:
                    # oligo_seq binds somewhere in target_seq
                    # As its best binding site, take the minimum over multiple
                    # positions (if there is more than one where it may bind)
                    for start_pos in bind_pos:
                        # If this score is better than the previous, store it
                        # If there's multiple scores saved, compare by first score
                        if compare(scores[start_pos][0], target_scores[seq_name][0][0]):
                            target_scores[seq_name] = [scores[start_pos],
                                                       oligo_seq,
                                                       start_pos]
        return target_scores

    def scores_where_guide_binds(self, guide_seqs):
        """Determine scores across the target sequences.

        Args:
            guide_seqs: guide sequences to lookup (treat as a guide set)

        Returns:
            (dict {target seq in self.seqs: (score,
                                             best guide sequence,
                                             start position)})
        """
        raise NotImplementedError()

    def seqs_where_guides_bind(self, guide_seqs):
        """Determine sequences to which guide binds.

        Args:
            guide_seqs: collection of guide sequences to lookup

        Returns:
            collection of names of sequences in self.seqs to which a guide
            in guide_seqs binds
        """
        seqs_bound = set()
        for seq_name, target_seq in self.seqs.items():
            for guide_seq in guide_seqs:
                bind_pos = self.find_binding_pos(seq_name, guide_seq,
                        self.guide_bind_fn)
                if len(bind_pos) > 0:
                    # guide_seq binds somewhere in target_seq
                    seqs_bound.add(seq_name)
                    break
        return seqs_bound

    def seqs_where_primers_bind(self, primer_left_seqs, primer_right_seqs):
        """Determine sequences for which a primer pair can amplify.

        Args:
            primer_left_seqs: collection of primers on the 5' end of the target
            primer_right_seqs: collection of primers on the 3' end of the target

        Returns:
            collection of names of sequences in self.seqs to which
            some pair of primers can amplify
        """
        if self.primer_terminal_mismatches is not None:
            def primer_left_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                    seq, target_seq, pos, self.primer_mismatches,
                    self.primer_terminal_mismatches, self.bases_from_terminal,
                    True)
            def primer_right_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                    seq, target_seq, pos, self.primer_mismatches,
                    self.primer_terminal_mismatches, self.bases_from_terminal,
                    False)
        else:
            def primer_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_mismatches(seq, target_seq, pos,
                    self.primer_mismatches, False)
            primer_left_bind_fn = primer_bind_fn
            primer_right_bind_fn = primer_bind_fn

        seqs_bound = set()
        seqs_bound_left = set()
        seqs_bound_right = set()
        primer_left_bind_pos_across_seq = set()
        primer_right_bind_pos_across_seq = set()
        for seq_name, target_seq in self.seqs.items():
            primer_left_bind_pos = set()
            min_primer_len_at_pos = {}
            for primer_seq in primer_left_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_left_bind_fn)
                primer_left_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_primer_len_at_pos:
                        min_primer_len_at_pos[pos] = len(primer_seq)
                    else:
                        min_primer_len_at_pos[pos] = min(
                                min_primer_len_at_pos[pos], len(primer_seq))
            if len(primer_left_bind_pos) > 0:
                seqs_bound_left.add(seq_name)
            primer_right_bind_pos = set()
            for primer_seq in primer_right_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_right_bind_fn)
                bind_pos_with_end = {(pos, pos+len(primer_seq))
                                     for pos in bind_pos}
                primer_right_bind_pos.update(bind_pos_with_end)
            if len(primer_right_bind_pos) > 0:
                seqs_bound_right.add(seq_name)

            # Determine if there exists some combination of (5' primer)/
            # (3' primer) that binds to target_seq
            added_seq = False
            for pl_pos in primer_left_bind_pos:
                # Consider all primers that fall *after* (3' of) pl
                primer_len = min_primer_len_at_pos[pl_pos]
                for (pr_pos, pr_end_pos) in primer_right_bind_pos:
                    if pr_pos > pl_pos and (self.max_target_length is None or
                            self.max_target_length >= pr_end_pos-pl_pos):
                        seqs_bound.add(seq_name)
                        added_seq = True
                        primer_left_bind_pos_across_seq.add(pl_pos)
                        primer_right_bind_pos_across_seq.add(pr_pos)
                        break
                if added_seq:
                    break
        logging.info("Number of seqs bound by left primer: %i" %
                      len(seqs_bound_left))
        logging.info("Number of seqs bound by right primer: %i" %
                      len(seqs_bound_right))
        logging.debug("Left primer binding positions: %s" %
                      primer_left_bind_pos_across_seq)
        logging.debug("Right primer binding positions: %s" %
                      primer_right_bind_pos_across_seq)

        return seqs_bound

    def seqs_where_targets_bind(self, guide_seqs, primer_left_seqs,
           primer_right_seqs):
        """Determine sequences to which a target binds.

        Args:
            guide_seqs: collection of guide sequences in the target
            primer_left_seqs: collection of primers on the 5' end of the target
            primer_right_seqs: collection of primers on the 3' end of the target

        Returns:
            collection of names of sequences in self.seqs to which
            some pair of primers and guide between them binds
        """
        if self.primer_terminal_mismatches is not None:
            def primer_left_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                    seq, target_seq, pos, self.primer_mismatches,
                    self.primer_terminal_mismatches, self.bases_from_terminal,
                    True)
            def primer_right_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                    seq, target_seq, pos, self.primer_mismatches,
                    self.primer_terminal_mismatches, self.bases_from_terminal,
                    False)
        else:
            def primer_bind_fn(seq, target_seq, pos):
                return self.evaluate_pos_by_mismatches(seq, target_seq, pos,
                    self.primer_mismatches, False)
            primer_left_bind_fn = primer_bind_fn
            primer_right_bind_fn = primer_bind_fn

        seqs_bound = set()
        seqs_bound_left = set()
        seqs_bound_right = set()
        seqs_bound_guide = set()
        primer_left_bind_pos_across_seq = set()
        primer_right_bind_pos_across_seq = set()
        guide_bind_pos_across_seq = set()
        for seq_name, target_seq in self.seqs.items():
            guide_bind_pos = set()
            min_guide_len_at_pos = {}
            for guide_seq in guide_seqs:
                bind_pos = self.find_binding_pos(seq_name, guide_seq,
                        self.guide_bind_fn)
                guide_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_guide_len_at_pos:
                        min_guide_len_at_pos[pos] = len(guide_seq)
                    else:
                        min_guide_len_at_pos[pos] = min(
                                min_guide_len_at_pos[pos], len(guide_seq))
            if len(guide_bind_pos) > 0:
                seqs_bound_guide.add(seq_name)
            primer_left_bind_pos = set()
            min_primer_len_at_pos = {}
            for primer_seq in primer_left_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_left_bind_fn)
                primer_left_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_primer_len_at_pos:
                        min_primer_len_at_pos[pos] = len(primer_seq)
                    else:
                        min_primer_len_at_pos[pos] = min(
                                min_primer_len_at_pos[pos], len(primer_seq))
            if len(primer_left_bind_pos) > 0:
                seqs_bound_left.add(seq_name)
            primer_right_bind_pos = set()
            for primer_seq in primer_right_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_right_bind_fn)
                bind_pos_with_end = {(pos, pos+len(primer_seq))
                                     for pos in bind_pos}
                primer_right_bind_pos.update(bind_pos_with_end)
            if len(primer_right_bind_pos) > 0:
                seqs_bound_right.add(seq_name)

            # Determine if there exists some combination of (5' primer)/guide/
            # (3' primer) that binds to target_seq
            added_seq = False
            for pl_pos in primer_left_bind_pos:
                # Consider all primers that fall *after* (3' of) pl
                primer_len = min_primer_len_at_pos[pl_pos]
                for (pr_pos, pr_end_pos) in primer_right_bind_pos:
                    if pr_pos > pl_pos:
                        # Find if any guide falls in between pl and pr
                        for g_pos in guide_bind_pos:
                            guide_len = min_guide_len_at_pos[g_pos]
                            if ((g_pos >= pl_pos + primer_len and
                                    pr_pos >= g_pos + guide_len) and
                                    (self.max_target_length is None or
                                    self.max_target_length >= pr_end_pos-pl_pos)):
                                seqs_bound.add(seq_name)
                                added_seq = True
                                primer_left_bind_pos_across_seq.add(pl_pos)
                                primer_right_bind_pos_across_seq.add(pr_pos)
                                guide_bind_pos_across_seq.add(g_pos)
                                break
                    if added_seq:
                        break
                if added_seq:
                    break
        logging.info("Number of seqs bound by guide: %i" %
                      len(seqs_bound_guide))
        logging.info("Number of seqs bound by left primer: %i" %
                      len(seqs_bound_left))
        logging.info("Number of seqs bound by right primer: %i" %
                      len(seqs_bound_right))
        logging.debug("Guide binding positions: %s" %
                      guide_bind_pos)
        logging.debug("Left primer binding positions: %s" %
                      primer_left_bind_pos)
        logging.debug("Right primer binding positions: %s" %
                      primer_right_bind_pos)

        return seqs_bound

    def seqs_bound_by_design(self, design):
        """Determine the sequences bound by a Design.

        Args:
            design: Design object

        Returns:
            collection of sequences that are bound by the guides and/or
            primers in design
        """
        if len(design.guides) == 0:
            return self.seqs_where_primers_bind(*design.primers)
        if design.is_complete_target:
            primers_left, primers_right = design.primers
            return self.seqs_where_targets_bind(design.guides,
                    primers_left, primers_right)
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

    def per_seq_guide(self):
        """Determine the per sequence score of guides from each design.

        Returns:
            dict mapping design identifier (self.designs.keys()) to a tuple of
            the best guide score per target sequence of its guide set
            and the best guide per target sequence of its guide set
        """
        scores = {}
        for design_id, design in self.designs.items():
            logger.info(("Computing per sequence scores of guides in design "
                "'%s'"), str(design_id))
            scores[design_id] = self.scores_where_guide_binds(design.guides)
        return scores

    def per_seq_primer_mismatches(self):
        """Determine the per sequence mismatches of primers from each design.

        Returns:
            tuple of two dicts (first for the left, second for the right)
            mapping design identifier (self.designs.keys()) to a
            dict {target seq name: ((primer mismatches,
                                     (optional) primer terminal mismatchs),
                                     best primer sequence,
                                     start position)}
        """
        left_mismatches = {}
        right_mismatches = {}
        for design_id, design in self.designs.items():
            logger.info(("Computing per sequence mismatches of primers in design "
                "'%s'"), str(design_id))
            if self.primer_terminal_mismatches is None:
                def bind_fn(seq, target_seq, pos, save=None):
                    return self.evaluate_pos_by_mismatches(seq, target_seq, pos,
                        self.primer_mismatches, False, save=save)
                left_bind_fn = bind_fn
                right_bind_fn = bind_fn
            else:
                def left_bind_fn(seq, target_seq, pos, save=None):
                    return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                        seq, target_seq, pos, self.primer_mismatches,
                        self.primer_terminal_mismatches,
                        self.bases_from_terminal, True, allow_gu_pairs=False,
                        save=save)
                def right_bind_fn(seq, target_seq, pos, save=None):
                    return self.evaluate_pos_by_terminal_mismatches_and_total_mismatches(
                        seq, target_seq, pos, self.primer_mismatches,
                        self.primer_terminal_mismatches,
                        self.bases_from_terminal, False, allow_gu_pairs=False,
                        save=save)
            left_mismatches[design_id] = self.scores_where_oligo_binds(
                design.primers[0], left_bind_fn)
            right_mismatches[design_id] = self.scores_where_oligo_binds(
                design.primers[1], right_bind_fn)
        return (left_mismatches, right_mismatches)


class CoverageAnalyzerWithMismatchModel(CoverageAnalyzer):
    """Methods to analyze coverage of a design using model with fixed number
    of mismatches for guide-target binding.
    """

    def __init__(self, seqs, designs, guide_mismatches, allow_gu_pairs=True,
            **kwargs):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            guide_mismatches: number of mismatches to tolerate when determining
                whether a guide hybridizes to a target sequence
            allow_gu_pairs: if True, tolerate G-U base pairs between a guide
                and target when computing whether a guide binds
            kwargs: keyword arguments from CoverageAnalyzer
        """
        super().__init__(seqs, designs, **kwargs)
        self.guide_mismatches = guide_mismatches
        self.allow_gu_pairs = allow_gu_pairs

    def scores_where_guide_binds(self, guide_seqs):
        """Determine mismatch scores across the target sequences.

        Args:
            guide_seqs: guide sequences to lookup (treat as a guide set)

        Returns:
            dict {target seq name: ((number of mismatches, ),
                                    best guide sequence,
                                    start position)}
        """
        return self.scores_where_oligo_binds(guide_seqs, self.guide_bind_fn,
                                             obj_type='min')

    def guide_bind_fn(self, seq, target_seq, pos, save=None):
        """Evaluate binding with a mismatch model.

        Args:
            seq: guide or primer sequence to lookup
            target_seq: target sequence against which to check
            pos: list of positions indicating subsequence in target_seq
                to check for binding of seq

        Returns:
            set of positions in pos where seq binds to target_seq
        """
        return self.evaluate_pos_by_mismatches(seq, target_seq,
                pos, self.guide_mismatches, self.allow_gu_pairs, save=save)

class CoverageAnalyzerWithPredictedActivity(CoverageAnalyzer):
    """Methods to analyze coverage of a design using model that determines
    guide-target binding based on whether it is predicted to be active or
    highly active.
    """

    def __init__(self, seqs, designs, predictor, highly_active=False,
            **kwargs):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            predictor: adapt.utils.predict_activity.Predictor object, used to
                determine whether a guide-target pair is predicted to be
                active
            highly_active: if True, determine a guide-target pair to bind if
                 it is predicted to be highly active (not just active)
            kwargs: keyword arguments from CoverageAnalyzer
        """
        super().__init__(seqs, designs, **kwargs)
        self.predictor = predictor
        self.highly_active = highly_active

    def scores_where_guide_binds(self, guide_seqs):
        """Determine activities across the target sequences.

        Args:
            guide_seqs: guide sequences to lookup (treat as a guide set)

        Returns:
            dict {target seq name: ((predicted activity, ),
                                    best guide sequence,
                                    start position)}
        """
        return self.scores_where_oligo_binds(guide_seqs, self.guide_bind_fn,
                                             obj_type='max')

    def mean_activity_of_guides(self):
        """Determine the mean activity of guides from each design.

        Returns:
            dict mapping design identifier (self.designs.keys()) to the
            mean activity, across the target sequences, of its guide set
        """
        mean_activities = {}
        for design_id, design in self.designs.items():
            logger.info(("Computing mean activities of guides in design "
                "'%s'"), str(design_id))
            activities_across_targets = self.scores_where_guide_binds(
                design.guides)
            mean_activities[design_id] = np.mean(
                [guide_stats[0] for guide_stats in
                 activities_across_targets.values()])
        return mean_activities

    def guide_bind_fn(self, seq, target_seq, pos, save=None):
        """Evaluate binding with a predictor -- i.e., based on what is
        predicted to be highly active.

        Args:
            seq: guide or primer sequence to lookup
            target_seq: target sequence against which to check
            pos: list of positions indicating subsequence in target_seq
                to check for binding of seq
            save: if set, this saves computed activities as
                save[p]=activity for each p in pos for which
                this can compute activity; self.highly_active must be False

        Returns:
            set of positions in pos where seq binds to target_seq
        """
        pairs_to_evaluate = []
        pos_evaluating = []
        for i in pos:
            # Extract a subsequence with context
            extract_start = i - self.predictor.context_nt
            extract_end = i + len(seq) + self.predictor.context_nt
            if (extract_start < 0 or extract_end > len(target_seq)):
                # The context needed for the target to predict activity
                # falls outside the range of target_seq; do not allow binding
                # at this position
                continue
            target_subseq_with_context = target_seq[extract_start:extract_end]
            pairs_to_evaluate += [(target_subseq_with_context, seq)]
            pos_evaluating += [i]

        # Do not bother memoizing predictions because target_seq are not
        # aligned; a position for one target_seq may not correspond to the
        # same position in another
        if self.highly_active:
            predictions = self.predictor.determine_highly_active(-1,
                    pairs_to_evaluate)
            if save is not None:
                raise Exception(("Cannot use save when using "
                    "'highly active' as a criterion"))
        else:
            predictions = self.predictor.compute_activity(-1,
                    pairs_to_evaluate)
            if save is not None:
                for i, p in zip(pos_evaluating, predictions):
                    save[i] = (p, )
            predictions = [bool(p > 0) for p in predictions]
        bind_pos = set()
        for i, p in zip(pos_evaluating, predictions):
            if p is True:
                # subsequence at i is active or highly active
                bind_pos.add(i)
        return bind_pos
