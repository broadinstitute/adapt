"""Utilities involving weighting sequences.
"""

import logging
import math
import numpy as np

from adapt.prepare import ncbi_neighbors

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'


def normalize(sequence_weights, accessions):
    """Normalize by total of the weights across accessions
    Make total of weights of all accessions specified sum to 1

    Args:
        sequence_weights: unnormalized sequences
        accessions: accessions to include in normalization

    Returns:
        dictionary {accession: normalized weight}
    """
    normalization_factor = sum(sequence_weights[accession]
                               for accession in accessions)
    return {accession: (sequence_weights[accession] / normalization_factor)
            for accession in accessions}


def percentile(activities, q, seq_norm_weights):
    """Take a weighted percentile of activities, using 'lower' interpolation

    Args:
        activities: list of activities to take the percentile of
        q: list of percentiles, in range [0, 100]
        seq_norm_weights: list of weights for each activity, in same order as
            activities

    Returns:
        list of activity percentiles, matching q's ordering
    """
    sorted_activities_idxs = np.argsort(activities)
    i = 1
    curr_p = seq_norm_weights[0]
    unset_q = {q_i/100: i for i, q_i in enumerate(q)}
    p_idxs = [-1 for _ in q]
    while len(unset_q) > 0 and i < len(sorted_activities_idxs):
        curr_p = curr_p + seq_norm_weights[sorted_activities_idxs[i]]
        set_qs = set()
        for q_i in unset_q:
            if curr_p < q_i:
                continue
            if curr_p > q_i:
                p_idxs[unset_q[q_i]] = i-1
            else:
                p_idxs[unset_q[q_i]] = i
            set_qs.add(q_i)
        for set_q in set_qs:
            unset_q.pop(set_q)
        i += 1

    return [activities[sorted_activities_idxs[p_idx]] for p_idx in p_idxs]
