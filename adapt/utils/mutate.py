"""Use GTR to model distribution of future viral sequences
"""

import numpy as np
from scipy.linalg import expm

__author__ = 'David K. Yang <yangd5153@gmail.com>'


class GTRSubstitutionMutator:
    """GTR Substitution model to model distribution of future viral sequences
    """
    def __init__(self, piA, piC, piG, piT,
                 rAC, rAG, rAT, rCG, rCT, rGT,
                 mu, t, guide_length):
        self.piA = piA
        self.piC = piC
        self.piG = piG
        self.piT = piT
        self.rAC = rAC
        self.rAG = rAG
        self.rAT = rAT
        self.rCG = rCG
        self.rCT = rCT
        self.rGT = rGT
        self.mu = mu
        self.t = t

        self.Q = self._construct_rateQ()
        self.P = self._construct_P()

    def _construct_rateQ(self):
        """Computes transition rate matrix Q, given base frequencies and transition rates under GTR model (Tavaré 1986).
        """

        beta = 1 / (2*(self.piA * (self.rAC*self.piC + self.rAG*self.piG + self.rAT*self.piT) + self.piC * (self.rCG*self.piG + self.rCT*self.piT) + self.piG*self.piT))
        Q = np.array([
                [-(self.rAC*self.piC + self.rAG*self.piG + self.rAT*self.piT), self.rAC*self.piC, self.rAG*self.piG, self.rAT*self.piT],
                [self.rAC*self.piA, -(self.rAC*self.piA + self.rCG*self.piG + self.rCT*self.piT), self.rCG*self.piG, self.rCT*self.piT],
                [self.rAG*self.piA, self.rCG*self.piC, -(self.rAG*self.piA + self.rCG*self.piC + self.rGT*self.piT), self.rGT*self.piT],
                [self.rAT*self.piA, self.rCT*self.piC, self.rGT*self.piG, -(self.rAT*self.piA + self.rCT*self.piC + self.rGT*self.piG)]
            ])

        return beta * Q

    def _construct_P(self):
        """Computes transition probability matrix P from rate matrix Q, substitution rate m, and time t
        """

        P = expm(self.Q * self.mu * self.t)
        row_sums = P.sum(axis=1)
        normalized_matrix = P / row_sums[:, np.newaxis] # normalize so each row sums to 1. Matrix exponentiation should already do this in principle
        return normalized_matrix

    def _seq_to_encoding(self, seq):
        """Encodes string sequence into an AA index list. e.g. "ACGT" returns [0, 1, 2, 3]

        Args:
            str: NT sequence

        Returns:
            list(int): list of AA idxs.
        """

        base_key = "ACGT"
        seq_as_list = list(seq)
        return [base_key.index(base) for base in seq_as_list]

    def compute_sequence_probability(self, wild_seq, mut_seq):
        """Computes probability of wild_seq mutating into mut_seq, under transition probability matrix P

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

    def mutate(self, wild_seq, n):
        """Sample future sequences under transition probability matrix P

        Args:
            wild_seq: reference sequence
            n: number of samples (int)

        Returns:
            list(str): list of samples from the distribution of future sequences
        """

        wild_seq_encoded = self._seq_to_encoding(wild_seq)
        sampled_seq_matrix = []

        for res in wild_seq_encoded:
            samples = np.random.choice(["A", "C", "G", "T"], n, p=self.P[res])
            sampled_seq_matrix.append(samples)

        sampled_seqs_list = np.array(sampled_seq_matrix).transpose()
        return ["".join(seq) for seq in sampled_seqs_list]
