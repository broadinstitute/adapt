"""Use GTR to model distribution of future viral sequences
"""

import numpy as np
from adapt import alignment
from adapt.utils import predict_activity
from scipy.linalg import expm

__author__ = 'David K. Yang <yangd5153@gmail.com>, Priya P. Pillai <ppillai@broadinstitute.org>'


class GTRSubstitutionMutator:
    """Use the GTR Substitution model to mutate viral sequences
    """
    def __init__(self, aln,
                 rAC, rAG, rAT, rCG, rCT, rGT,
                 mu, t, n):
        """
        Args:
            aln: An alignment of sequences to use to determine base percentages
            rAC: Relative rate of conversion between A and C
            rAG: Relative rate of conversion between A and G
            rAT: Relative rate of conversion between A and T
            rCG: Relative rate of conversion between C and G
            rCT: Relative rate of conversion between C and T
            rGT: Relative rate of conversion between G and T
            mu: Overall rate of substitutions per site per year
            t: Years to simulate substitutions over
            n: Number of sequences to simulate mutations over
        """
        base_percentages = aln.base_percentages()
        self.piA = base_percentages['A']
        self.piC = base_percentages['C']
        self.piG = base_percentages['G']
        self.piT = base_percentages['T']
        self.rAC = rAC
        self.rAG = rAG
        self.rAT = rAT
        self.rCG = rCG
        self.rCT = rCT
        self.rGT = rGT
        self.mu = mu
        self.t = t
        self.n = n

        self.Q = self._construct_rateQ()
        self.P = self._construct_P()

    def _construct_rateQ(self):
        """Compute transition rate matrix

        Computes transition rate matrix Q, given base frequencies and
        transition rates under GTR model (Tavaré 1986).

        The transition rate matrix defines the rate each base is mutated to
        each other base. The rows indicate the starting base; the columns
        indicate the final base. Bases are ordered A, C, G, T. Diagonal
        elements are set such that the row sums to 0.
        """

        beta = 1 / (2 * (
            self.piA * (self.rAC*self.piC +
                        self.rAG*self.piG +
                        self.rAT*self.piT) +
            self.piC * (self.rCG*self.piG +
                        self.rCT*self.piT) +
            self.piG * (self.rGT*self.piT)
        ))
        Q = np.array([
                [-(self.rAC*self.piC + self.rAG*self.piG + self.rAT*self.piT),
                 self.rAC*self.piC,
                 self.rAG*self.piG,
                 self.rAT*self.piT],
                [self.rAC*self.piA,
                 -(self.rAC*self.piA + self.rCG*self.piG + self.rCT*self.piT),
                 self.rCG*self.piG,
                 self.rCT*self.piT],
                [self.rAG*self.piA,
                 self.rCG*self.piC,
                 -(self.rAG*self.piA + self.rCG*self.piC + self.rGT*self.piT),
                 self.rGT*self.piT],
                [self.rAT*self.piA,
                 self.rCT*self.piC,
                 self.rGT*self.piG,
                 -(self.rAT*self.piA + self.rCT*self.piC + self.rGT*self.piG)]
            ])

        return beta * Q

    def _construct_P(self):
        """Compute transition probability matrix

        Computes transition probability matrix P from rate matrix Q,
        substitution rate m, and time t

        The transition probablility matrix defines the likelihood each base is
        mutated to each other base. The rows indicate the starting base; the
        columns indicate the final base. Bases are ordered A, C, G, T.
        """

        P = expm(self.Q * self.mu * self.t)
        # Normalize so each row sums to 1.
        # Matrix exponentiation should already do this in principle
        row_sums = P.sum(axis=1)
        normalized_matrix = P / row_sums[:, np.newaxis]
        return normalized_matrix

    def _seq_to_encoding(self, seq):
        """Encode string sequence into a nucleotide index list

        Map 'A' to 0, 'C' to 1, 'G' to 2, and 'T' to 3
        (e.g. "ACGT" returns [0, 1, 2, 3])

        Args:
            str: nucleotide sequence

        Returns:
            list(int): list of nucleotide indexs.
        """

        base_key = "ACGT"
        seq_as_list = list(seq)
        return [base_key.index(base) for base in seq_as_list]

    def compute_sequence_probability(self, wild_seq, mut_seq):
        """Compute probability of wild_seq mutating into mut_seq

        Under transition probability matrix P, finds probability of wild_seq
        mutating into mut_seq

        Args:
            wild_seq: reference sequence
            mut_seq: mutated sequence to compute probability for

        Returns:
            float: probability of mutated sequence
        """

        wild_seq_encoded = self._seq_to_encoding(wild_seq)
        mut_seq_encoded = self._seq_to_encoding(mut_seq)

        prob = 1.0
        for a, b in zip(wild_seq_encoded, mut_seq_encoded):
            prob = prob * self.P[a][b]

        return prob

    def mutate(self, wild_seq):
        """Sample future sequences under transition probability matrix P

        Args:
            wild_seq: reference sequence

        Returns:
            list(str): list of samples from the distribution of future
                sequences
        """

        wild_seq_encoded = self._seq_to_encoding(wild_seq)
        sampled_seq_matrix = []

        for res in wild_seq_encoded:
            samples = np.random.choice(["A", "C", "G", "T"], self.n,
                                       p=self.P[res])
            sampled_seq_matrix.append(samples)

        sampled_seqs_list = np.array(sampled_seq_matrix).transpose()
        return ["".join(seq) for seq in sampled_seqs_list]

    def compute_mutated_activity(self, predictor, target_seq,
                                 guide_seq, start=0):
        """Calculate the activity of the guide after mutating the target

        Args:
            predictor: an adapt.utils.predict_activity Predictor object
            target_seq: string of what sequence the guide targets. Includes
                    context if predictor requires it
            guide_seq: string of what the guide is
            start: int, start position for the guide sequence in the alignment.
                    Required to use predictor memoization, if it exists

        Returns:
            The 5th percentile of activity values from simulated mutations
        """
        mutated_target_seqs = self.mutate(target_seq)
        if isinstance(predictor, predict_activity.SimpleBinaryPredictor):
            mutated_aln = alignment.Alignment.from_list_of_seqs(
                mutated_target_seqs)
            left_context = 0
            if predictor.required_flanking_seqs[0]:
                left_context = len(predictor.required_flanking_seqs[0])
            _, activity = predictor.compute_activity(left_context, guide_seq,
                                                     mutated_aln, percentiles=5)
        else:
            pairs_to_eval = []
            for mutated_target_seq in mutated_target_seqs:
                pair = (mutated_target_seq, guide_seq)
                pairs_to_eval.append(pair)
            _, activity = predictor.compute_activity(start, pairs_to_eval,
                                                     percentiles=5)
        return activity
