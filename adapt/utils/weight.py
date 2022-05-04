"""Utilities involving weighting sequences.
"""

import logging
import math
import numpy as np

from adapt.prepare import ncbi_neighbors

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'

logger = logging.getLogger(__name__)


def weight_by_log_group(groups):
    """Weight sequences by the log of the number of sequences in the group

    If G is a set of sequences in a group, the unnormalized weight of each
    sequence in G will be set to log(|G| + 1)/|G|

    Args:
        groups: dictionary {group name: collection of sequence names in group}.
            Sequence names must be unique across all groups.

    Returns:
        dictionary {sequence name: unnormalized weight}
    """
    weights = {}
    for group in groups:
        num_in_group = len(groups[group])
        # Each group's weight should total to the log of the number of
        # sequences in that group, so the weight per sequence should be that
        # divided by the number in the group
        # Add one to the numerator to make sure weights aren't 0
        if num_in_group != 0:
            weight_per_seq = math.log(num_in_group+1) / num_in_group
            for seq_name in groups[group]:
                if seq_name in weights:
                    raise ValueError("%s is in multiple weighting groups, and "
                        "so its weight cannot be determined." %seq_name)
                weights[seq_name] = weight_per_seq

    return weights


def normalize(sequence_weights, seq_names):
    """Normalize by total of the weights across sequences
    Make total of weights of all sequences specified sum to 1

    Args:
        sequence_weights: dictionary of unnormalized sequence weights. Sum
            must be greater than zero
        seq_names: sequences to include in normalization

    Returns:
        dictionary {sequence name: normalized weight}
    """
    normalization_factor = sum(sequence_weights[seq_name]
                               for seq_name in seq_names)
    if normalization_factor <= 0:
        raise ValueError("Weights must have a sum greater than or equal to 0; "
            "the weights currently sum to %f" %(normalization_factor))

    return {seq_name: (sequence_weights[seq_name] / normalization_factor)
            for seq_name in seq_names}


def percentile(activities, q, seq_weights):
    """Take a weighted percentile of activities, using 'lower' interpolation

    The weighted percentile uses total sequence weight, rather than the number
    of sequences, in order to determine percentile calculations. The 'lower'
    interpolation means that the percentile value returned is greater than or
    equal to *at most* that percent of sequence weight.

    For example, if activities were [5, 3, 4] and weights were [0.4, 0.2, 0.4],
    the 20th percentile would be 3, the 60th percentile would be 4, and the
    100th percentile would be 5. Percentiles in between would be interpolated
    to the lower value (so the 80th percentile would be 4), and percentiles
    less than 20 would return the lowest value (3).

    Args:
        activities: list of activities to take the percentile of
        q: list of percentiles, in range [0, 100]
        seq_weights: list of weights for each activity, in the same order as
            activities; if unnormalized, it will be normalized

    Returns:
        list of activity percentile values, matching q's ordering
    """
    if math.isclose(sum(seq_weights), 1):
        seq_norm_weights = seq_weights
    else:
        logger.warning("Weights have not been normalized for percentile "
            "calculations; normalizing them here.")
        seq_norm_weights = normalize(seq_weights, range(len(seq_weights)))
    # Order the activities, but get a list of the indexes of the ordering
    # to be able to match activities to their weights
    sorted_activities_idxs = np.argsort(activities)
    # Keep track of the current percentile we're on
    curr_p = 0
    # Keep track of the current sequence we're on
    i = 0
    # Change requested percentiles into decimals & keep track of their original
    # ordering
    q_pos = {q_i/100: pos for pos, q_i in enumerate(q)}
    # Order the requested percentiles to iterate through them
    ordered_qs = sorted(q_pos.keys())
    curr_q = ordered_qs.pop(0)
    # Keep track of the index of the item in the original percentile list that
    # matches the requested percentile (in the same order as the original qs)
    p_idxs = [-1 for _ in q]
    while len(ordered_qs) > 0 and i < len(sorted_activities_idxs):
        # Since sequence weights are normalized, the sum of weights of
        # sequences up to a given sequence is the current percentile
        curr_p = curr_p + seq_norm_weights[sorted_activities_idxs[i]]
        while curr_p > curr_q or math.isclose(curr_p, curr_q):
            if math.isclose(curr_p, curr_q):
                # Requested percentile equals current percentile
                p_idxs[q_pos[curr_q]] = i
            else:
                # Using 'lower' interpolation, so use the one prior
                # (unless it is the first one)
                p_idxs[q_pos[curr_q]] = max(i-1, 0)
            if len(ordered_qs) > 0:
                curr_q = ordered_qs.pop(0)
            else:
                break
        i += 1

    return [activities[sorted_activities_idxs[p_idx]] for p_idx in p_idxs]
