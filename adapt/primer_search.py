"""Methods for searching for optimal primers through a genome.

This makes heavy use of the guide_search module.
"""

import logging

from adapt.utils import search
from adapt.utils import oligo
from adapt import alignment

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


class PrimerResult:
    """Store results of a primer cover at a site."""

    def __init__(self, start, num_primers, primer_length,
        frac_bound, primers_in_cover):
        """
        Args:
            start: start position of the primer
            num_primers: number of primers designed at the site
            primer_length: length of each primer
            frac_bound: total fraction of all sequences bound by the primers
            primers_in_cover: set of primers that achieve the desired coverage
                 and is minimal at the site
        """
        self.start = start
        self.num_primers = num_primers
        self.primer_length = primer_length
        self.frac_bound = frac_bound
        self.primers_in_cover = primers_in_cover

    def overlaps(self, other, expand=0):
        """Determine if self overlaps other.

        Args:
            other: PrimerResult object
            expand: tests overlap within +/- EXPAND nt of other

        Returns:
            True iff self overlaps other
        """
        return ((self.start - expand <= other.start < (self.start +
                    self.primer_length + expand)) or
                (other.start - expand <= self.start < (other.start +
                    other.primer_length + expand)))

    def overlaps_range(self, start, end):
        """Determine if self overlaps a range.

        Args:
            start: start of range (inclusive)
            end: end of range (exclusive)

        Returns:
            True iff self overlaps (start, end)
        """
        return ((self.start < end) and (start < self.start + self.primer_length))

    def __str__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_cover))

    def __repr__(self):
        return str((self.start, self.num_primers, self.primer_length,
            self.frac_bound, self.primers_in_cover))

    def __eq__(self, other):
        """Determine equality of self and other.

        Args:
            other: an object of PrimerResult

        Returns:
            True iff self is identical to other
        """
        return (self.start == other.start and
                self.num_primers == other.num_primers and
                self.primer_length == other.primer_length and
                self.frac_bound == other.frac_bound and
                self.primers_in_cover == other.primers_in_cover)


class PrimerSearcher(search.OligoSearcherMinimizeNumber):
    """Methods to search for primers over a genome.

    This is a special case of guide_search.GuideSearcherMinimizeGuides; thus, it
    is a subclass of guide_search.GuideSearcherMinimizeGuides. This effectively
    looks for guides (here, primers) within each window of size w
    where w is the length of a primer.

    The input is an alignment of sequences over which to search, as well as
    parameters defining the space of acceptable primers.
    """

    def __init__(self, aln, primer_length, mismatches, cover_frac,
        missing_data_params, primer_gc_content_bounds=None,
        is_suitable_fns=[], **kwargs):
        """
        Args:
            aln: alignment.Alignment representing an alignment of sequences
            primer_length: length of the primer to construct
            mismatches: threshold on number of mismatches for determining whether
                a primer would hybridize to a target sequence
            cover_frac: fraction in (0, 1] of sequences that must be 'captured' by
                 a primer; see seq_groups
            missing_data_params: tuple (a, b, c) specifying to not attempt to
                design guides overlapping sites where the fraction of
                sequences with missing data is > min(a, max(b, c*m)), where m is
                the median fraction of sequences with missing data over the
                alignment
            primer_gc_content_bounds: a tuple (lo, hi) such that this only
                yields sites where all primers have a GC content fraction in
                [lo, hi]; or None for no bounds
        """
        if primer_gc_content_bounds:
            lo, hi = primer_gc_content_bounds
            assert lo <= hi
            def check_gc_content(primer_seq):
                """Determine whether primer meets bounds on GC content.

                Args:
                    primer_seq: primer sequence

                Returns:
                    True/False indicating whether all primers meet the bounds on
                    GC content
                """
                gc_frac = oligo.gc_frac(primer_seq)
                if gc_frac < lo or gc_frac > hi:
                    return False
                return True
            is_suitable_fns.append(check_gc_content)

        self.primer_gc_content_bounds = primer_gc_content_bounds

        super().__init__(aln=aln, min_oligo_length=primer_length,
            max_oligo_length=primer_length, cover_frac=cover_frac,
            mismatches=mismatches, missing_data_params=missing_data_params,
            is_suitable_fns=is_suitable_fns, **kwargs)

    def construct_oligo(self, start, oligo_length, seqs_to_consider,
        num_needed=None, stop_early=True):
        """Construct a single guide to target a set of sequences in the alignment.

        This constructs a guide to target sequence within the range [start,
        start+guide_length]. It only considers the sequences with indices given in
        seqs_to_consider.

        Args:
            start: start position in alignment at which to target
            seqs_to_consider: dict mapping universe group ID to collection of
                indices to use when constructing the guide
            num_needed: dict mapping universe group ID to the number of sequences
                from the group that are left to cover in order to achieve
                a desired coverage; these are used to help construct a
                guide
            stop_early: if True, impose early stopping criteria while iterating
                over clusters to improve runtime

        Returns:
            tuple (x, y) where:
                x is the sequence of the constructed guide
                y is a set of indices of sequences (a subset of
                    values in seqs_to_consider) to which the guide x will
                    hybridize
            (Note that it is possible that x binds to no sequences and that
            y will be empty.)
        """
        # TODO: There are several optimizations that can be made to
        # this function that take advantage of G-U pairing in order
        # to lower the number of guides that need to be designed.
        # Two are:
        #  1) The function SequenceClusterer.cluster(..), which is
        #     used here, can cluster accounting for G-U pairing (e.g.,
        #     such that 'A' hashes to 'G' and 'C' hashes to 'T',
        #     so that similar guide sequences hash to the same
        #     value tolerating G-U similarity).
        #  2) Instead of taking a consensus sequence across a
        #     cluster, this can take a pseudo-consensus that
        #     accounts for G-U pairing (e.g., in which 'A' is
        #     treated as 'G' and 'C' is treated as 'T' for
        #     the purposes of generating a consensus sequence).

        assert start + oligo_length <= self.aln.seq_length
        assert len(seqs_to_consider) > 0

        for pos in range(start, start + oligo_length):
            if self.aln.frac_missing_at_pos(pos) > self.missing_threshold:
                raise alignment.CannotConstructOligoError(("Too much missing "
                    "data at a position in the target range"))

        aln_for_guide = self.aln.extract_range(start, start + oligo_length)

        if self.predictor is not None:
            # Extract the target sequences, including context to use with
            # prediction
            if (start - self.predictor.context_nt < 0 or
                    start + oligo_length + self.predictor.context_nt >
                    self.aln.seq_length):
                raise alignment.CannotConstructOligoError(("The context needed "
                    "for the target to predict activity falls outside the "
                    "range of the alignment at this position"))
            aln_for_guide_with_context = self.aln.extract_range(
                    start - self.predictor.context_nt,
                    start + oligo_length + self.predictor.context_nt)

        # Before modifying seqs_to_consider, make a copy of it
        seqs_to_consider_cp = {}
        for group_id in seqs_to_consider.keys():
            seqs_to_consider_cp[group_id] = set(seqs_to_consider[group_id])
        seqs_to_consider = seqs_to_consider_cp

        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Ignore any sequences in the alignment that have a gap in
        # this region
        if self.predictor is not None:
            seqs_with_gap = set(aln_for_guide_with_context.seqs_with_gap(
                all_seqs_to_consider))
        else:
            seqs_with_gap = set(aln_for_guide.seqs_with_gap(
                all_seqs_to_consider))
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].difference_update(seqs_with_gap)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # Only consider sequences in the alignment that have the
        # required flanking sequence(s)
        seqs_with_flanking = self.aln.seqs_with_required_flanking(
            start, oligo_length, self.required_flanking_seqs,
            seqs_to_consider=all_seqs_to_consider)
        for group_id in seqs_to_consider.keys():
            seqs_to_consider[group_id].intersection_update(seqs_with_flanking)
        all_seqs_to_consider = set.union(*seqs_to_consider.values())

        # If every sequence in this region has a gap or does not contain
        # required flanking sequence, then there are none left to consider
        if len(all_seqs_to_consider) == 0:
            raise alignment.CannotConstructOligoError(("All sequences in region "
                "have a gap and/or do not contain required flanking sequences"))

        seq_rows = aln_for_guide.make_list_of_seqs(all_seqs_to_consider,
            include_idx=True)
        if self.predictor is not None:
            seq_rows_with_context = aln_for_guide_with_context.make_list_of_seqs(
                    all_seqs_to_consider, include_idx=True)

        if self.predictor is not None:
            # Memoize activity evaluations
            pair_eval = {}
        def determine_binding_and_active_seqs(gd_sequence):
            binding_seqs = set()
            num_bound = 0
            if self.predictor is not None:
                num_passed_predict_active = 0
                # Determine what calls to make to
                # self.predictor.determine_highly_active(); it
                # is best to batch these
                pairs_to_eval = []
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        pairs_to_eval += [pair]
                # Evaluate activity
                evals = self.predictor.determine_highly_active(start, pairs_to_eval)
                for pair, y in zip(pairs_to_eval, evals):
                    pair_eval[pair] = y
                # Fill in binding_seqs
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        num_bound += 1
                        seq_with_context, _ = seq_rows_with_context[i]
                        pair = (seq_with_context, gd_sequence)
                        if pair_eval[pair]:
                            num_passed_predict_active += 1
                            binding_seqs.add(seq_idx)
            else:
                num_passed_predict_active = None
                for i, (seq, seq_idx) in enumerate(seq_rows):
                    if oligo.binds(gd_sequence, seq, self.mismatches,
                            self.allow_gu_pairs):
                        num_bound += 1
                        binding_seqs.add(seq_idx)
            return binding_seqs, num_bound, num_passed_predict_active

        # Define a score function (higher is better) for a collection of
        # sequences covered by a guide
        if num_needed is not None:
            # This is the number of sequences it contains that are needed to
            # achieve the partial cover; we can compute this by summing over
            # the number of needed sequences it contains, taken across the
            # groups in the universe
            # Memoize the scores because this computation might be expensive
            seq_idxs_scores = {}
            def seq_idxs_score(seq_idxs):
                seq_idxs = set(seq_idxs)
                tc = tuple(seq_idxs)
                if tc in seq_idxs_scores:
                    return seq_idxs_scores[tc]
                score = 0
                for group_id, needed in num_needed.items():
                    contained_in_seq_idxs = seq_idxs & seqs_to_consider[group_id]
                    score += min(needed, len(contained_in_seq_idxs))
                seq_idxs_scores[tc] = score
                return score
        else:
            # Score by the number of sequences it contains
            def seq_idxs_score(seq_idxs):
                return len(seq_idxs)

        # First construct the optimal guide to cover the sequences. This would be
        # a string x that maximizes the number of sequences s_i such that x and
        # s_i are equal to within 'mismatches' mismatches; it's called the "max
        # close string" or "close to most strings" problem. For simplicity, let's
        # do the following: cluster the sequences (just the portion with
        # potential guides) with LSH, choose a guide for each cluster to be
        # the consensus of that cluster, and choose the one (across clusters)
        # that has the highest score
        if self.clusterer is None:
            # Don't cluster; instead, draw the consensus from all the
            # sequences. Effectively, treat it as there being just one
            # cluster, which consists of all sequences to consider
            clusters_ordered = [all_seqs_to_consider]
        else:
            # Cluster the sequences
            clusters = self.clusterer.cluster(seq_rows)

            # Sort the clusters by score, from highest to lowest
            # Here, the score is determined by the sequences in the cluster
            clusters_ordered = sorted(clusters, key=seq_idxs_score, reverse=True)

        best_gd = None
        best_gd_binding_seqs = None
        best_gd_score = 0
        stopped_early = False
        for cluster_idxs in clusters_ordered:
            if stop_early and best_gd_score > seq_idxs_score(cluster_idxs):
                # The guide from this cluster is unlikely to exceed the current
                # score; stop early
                stopped_early = True
                break

            gd = aln_for_guide.determine_consensus_sequence(
                cluster_idxs)
            if 'N' in gd:
                # Skip this; all sequences at a position in this cluster
                # are 'N'
                continue
            skip_cluster = False
            for is_suitable_fn in self.is_suitable_fns:
                if is_suitable_fn(gd) is False:
                    # Skip this cluster
                    skip_cluster = True
                    break
            if skip_cluster:
                continue
            # Determine the sequences that are bound by this guide (and
            # where it is 'active', if self.predictor is set)
            binding_seqs, num_bound, num_passed_predict_active = \
                    determine_binding_and_active_seqs(gd)
            score = seq_idxs_score(binding_seqs)
            if score > best_gd_score:
                best_gd = gd
                best_gd_binding_seqs = binding_seqs
                best_gd_score = score

            # Impose an early stopping criterion if self.predictor is
            # used, because using it is slow
            if self.predictor is not None and stop_early:
                if (num_bound >= 0.5*len(cluster_idxs) and
                        num_passed_predict_active < 0.1*len(cluster_idxs)):
                    # gd binds (according to guide.guide_binds()) many
                    # sequences, but is not predicted to be active against
                    # many; it is likely that this region is poor according to
                    # the self.predictor (e.g., due to sequence composition).
                    # Rather than trying other clusters at this site, just skip
                    # it. Note that this is just a heuristic; it can help
                    # runtime when the self.predictor is used, but is not needed
                    # and may hurt the optimality of the solution
                    stopped_early = True
                    break
        gd = best_gd
        binding_seqs = best_gd_binding_seqs
        score = best_gd_score

        # It's possible that the consensus sequence (guide) of no cluster
        # binds to any of the sequences. In this case, simply go through all
        # sequences and find the first that has no ambiguity and is suitable
        # and active, and make this the guide
        if gd is None and not stopped_early:
            for i, (s, idx) in enumerate(seq_rows):
                if not set(s).issubset(set(['A', 'C', 'G', 'T'])):
                    # s has ambiguity; skip it
                    continue
                skip_cluster = False
                for is_suitable_fn in self.is_suitable_fns:
                    if is_suitable_fn(s) is False:
                        # Skip this cluster
                        skip_cluster = True
                        break
                if skip_cluster:
                    continue
                if self.predictor is not None:
                    s_with_context, _ = seq_rows_with_context[i]
                    if not self.predictor.determine_highly_active(start,
                            [(s_with_context, s)])[0]:
                        # s is not active against itself; skip it
                        continue
                # s has no ambiguity and is a suitable guide; use it
                gd = s
                binding_seqs, _, _ = determine_binding_and_active_seqs(gd)
                break

        if gd is None:
            raise alignment.CannotConstructOligoError(("No guides are suitable "
                "or active"))

        return (gd, binding_seqs, score)

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
              3) total fraction of all sequences bound by the primers
              4) set of primers that achieve the desired coverage
                 and is minimal at the site
            They are given in sorted order, by position in the
            alignment
        """
        window_size = self.max_oligo_length
        for cover in self._find_oligos_for_each_window(
                window_size, hide_warnings=True):
            start, end, primers_in_cover = cover
            num_primers = len(primers_in_cover)
            frac_bound = self.total_frac_bound(primers_in_cover)

            # Check constraints
            if max_at_site is not None and num_primers > max_at_site:
                continue

            yield PrimerResult(
                start, num_primers, window_size,
                frac_bound, primers_in_cover)
