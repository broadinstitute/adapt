"""Methods for analyzing coverage of a design against a collection of sequences.
"""

from collections import defaultdict
import logging

import numpy as np

from adapt.utils import guide

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

    def __init__(self, seqs, designs, primer_mismatches=None,
            fully_sensitive=False):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            primer_mismatches: number of mismatches to tolerate when determining
                whether a primer hybridizes to a target sequence (if not
                set, the designs are assumed to not contain primers)
            fully_sensitive: if True, use a fully sensitive lookup of
                primers and guides (slower)
        """
        self.seqs = seqs
        self.designs = designs
        self.primer_mismatches = primer_mismatches
        self.seqs_are_indexed = False
        self.seqs_index_k = None
        self.fully_sensitive = fully_sensitive

    def _index_seqs(self, k=6, stride_by_k=False):
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
        """
        logger.info(("Indexing sequences"))
        self.seqs_index_k = k
        self.seqs_index = {}
        for name, seq in self.seqs.items():
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
        logger.info(("Done indexing sequences"))

    def evaluate_pos_by_mismatches(self, seq, target_seq, pos,
            mismatches, allow_gu_pairs):
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

        Returns:
            set of positions in pos where seq binds to target_seq
        """
        bind_pos = set()
        for i in pos:
            target_subseq = target_seq[i:(i + len(seq))]
            if guide.guide_binds(seq, target_subseq, mismatches,
                    allow_gu_pairs):
                bind_pos.add(i)
        return bind_pos

    def guide_bind_fn(self, seq, target_seq, pos):
        """Determine whether guide detects target at positions.

        This should be implemented in a subclass.
        """
        raise NotImplementedError()

    def find_binding_pos(self, target_seq_name, seq, bind_fn,
            save_activities=None):
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
            save_activities: if set, pass along to guide_fn to save computed
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

        if not self.seqs_are_indexed:
            self._index_seqs()

        def bind_fn_with_save(seq, target_seq, pos):
            if save_activities is None:
                return bind_fn(seq, target_seq, pos)
            else:
                return bind_fn(seq, target_seq, pos,
                        save_activities=save_activities)

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
                bind_pos = self.find_binding_pos(seq_name, guide_seq,
                        self.guide_bind_fn)
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
        def primer_bind_fn(seq, target_seq, pos):
            return self.evaluate_pos_by_mismatches(seq, target_seq, pos,
                self.primer_mismatches, False)

        seqs_bound = set()
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
            primer_left_bind_pos = set()
            min_primer_len_at_pos = {}
            for primer_seq in primer_left_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_bind_fn)
                primer_left_bind_pos.update(bind_pos)
                for pos in bind_pos:
                    if pos not in min_primer_len_at_pos:
                        min_primer_len_at_pos[pos] = len(primer_seq)
                    else:
                        min_primer_len_at_pos[pos] = min(
                                min_primer_len_at_pos[pos], len(primer_seq))
            primer_right_bind_pos = set()
            for primer_seq in primer_right_seqs:
                bind_pos = self.find_binding_pos(seq_name, primer_seq,
                        primer_bind_fn)
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
            activities_across_targets = self.activities_where_guide_binds(design.guides)
            mean_activities[design_id] = np.mean(list(activities_across_targets.values()))
        return mean_activities


class CoverageAnalyzerWithMismatchModel(CoverageAnalyzer):
    """Methods to analyze coverage of a design using model with fixed number
    of mismatches for guide-target binding.
    """

    def __init__(self, seqs, designs, guide_mismatches, primer_mismatches=None,
            allow_gu_pairs=True, fully_sensitive=False):
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
            fully_sensitive: if True, use a fully sensitive lookup of
                primers and guides (slower)
        """
        super().__init__(seqs, designs, primer_mismatches=primer_mismatches,
                fully_sensitive=fully_sensitive)
        self.guide_mismatches = guide_mismatches
        self.allow_gu_pairs = allow_gu_pairs

    def activities_where_guide_binds(self, guide_seqs):
        raise NotImplementedError()

    def guide_bind_fn(self, seq, target_seq, pos):
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
                pos, self.guide_mismatches, self.allow_gu_pairs)


class CoverageAnalyzerWithPredictedActivity(CoverageAnalyzer):
    """Methods to analyze coverage of a design using model that determines
    guide-target binding based on whether it is predicted to be active or
    highly active.
    """

    def __init__(self, seqs, designs, predictor, primer_mismatches=None,
            highly_active=False, fully_sensitive=False):
        """
        Args:
            seqs: dict mapping sequence name to sequence; checks coverage of the
                designs against these sequences
            designs: dict mapping an identifier for a design to Design object
            predictor: adapt.utils.predict_activity.Predictor object, used to
                determine whether a guide-target pair is predicted to be
                active
            primer_mismatches: number of mismatches to tolerate when determining
                whether a primer hybridizes to a target sequence (if not
                set, the designs are assumed to not contain primers)
            highly_active: if True, determine a guide-target pair to bind if
                 it is predicted to be highly active (not just active)
            fully_sensitive: if True, use a fully sensitive lookup of
                primers and guides (slower)
        """
        super().__init__(seqs, designs, primer_mismatches=primer_mismatches,
                fully_sensitive=fully_sensitive)
        self.predictor = predictor
        self.highly_active = highly_active

    def activities_where_guide_binds(self, guide_seqs):
        """Determine activities across the target sequences.

        Args:
            guide_seqs: guide sequences to lookup (treat as a guide set)

        Returns:
            dict {target seq in self.seqs: predicted activity}
        """
        target_act = {}
        for seq_name, target_seq in self.seqs.items():
            activities_for_target = []
            for guide_seq in guide_seqs:
                activities = {}
                bind_pos = self.find_binding_pos(seq_name, guide_seq,
                        self.guide_bind_fn, save_activities=activities)
                if len(bind_pos) > 0:
                    # guide_seq binds somewhere in target_seq
                    # As its activity, take the maximum over multiple
                    # positions (if there is more than one where it may bind)
                    activities_for_target += [max(activities.values())]
                else:
                    activities_for_target += [0]
            # If there are multiple guides, take the max across them
            target_act[target_seq] = max(activities_for_target)
        return target_act

    def guide_bind_fn(self, seq, target_seq, pos, save_activities=None):
        """Evaluate binding with a predictor -- i.e., based on what is
        predicted to be highly active.

        Args:
            seq: guide or primer sequence to lookup
            target_seq: target sequence against which to check
            pos: list of positions indicating subsequence in target_seq
                to check for binding of seq
            save_activities: if set, this saves computed activities as
                save_activites[p]=activity for each p in pos for which
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
            if save_activities is not None:
                raise Exception(("Cannot use save_activities when using "
                    "'highly active' as a criterion"))
        else:
            predictions = self.predictor.compute_activity(-1,
                    pairs_to_evaluate)
            if save_activities is not None:
                for i, p in zip(pos_evaluating, predictions):
                    save_activities[i] = p
            predictions = [bool(p > 0) for p in predictions]
        bind_pos = set()
        for i, p in zip(pos_evaluating, predictions):
            if p is True:
                # subsequence at i is active or highly active
                bind_pos.add(i)
        return bind_pos
