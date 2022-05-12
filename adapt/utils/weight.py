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
        sequence_weights: dictionary of unnormalized sequence weights. Sum of
            seq_names values must be greater than zero
        seq_names: sequences to include in normalization

    Returns:
        dictionary {sequence name: normalized weight}
    """
    if len(seq_names) == 0:
        return {}

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
        q: list of percentiles, in range [0, 100]. Should all be unique.
        seq_weights: list of weights for each activity, in the same order as
            activities; if unnormalized, it will be normalized

    Returns:
        list of activity percentile values, matching q's ordering
    """
    if len(q) == 0:
        return []

    if math.isclose(sum(seq_weights), 1):
        seq_norm_weights = seq_weights
    else:
        logger.info("Weights have not been normalized for percentile "
            "calculations; normalizing them here.")
        seq_norm_weights = normalize(seq_weights, range(len(seq_weights)))
    # Order the activities, but get a list of the indexes of the ordering
    # to be able to match activities to their weights
    sorted_activities_idxs = np.argsort(activities)
    # Change requested percentiles into decimals & keep track of their original
    # ordering.
    q_idxs = {q_i/100: idx for idx, q_i in enumerate(q)}
    # Order the requested percentiles to reduce the number of times q must be
    # checked
    q_queue = sorted(q_idxs.keys())
    curr_q = q_queue[0]

    # Keep track of percentile values in the same order as the original qs
    p = [None for _ in q]

    # Iterate through activities in order (sorted_activities_idxs[i]), while
    # keeping track of the current percentile value (curr_p)
    curr_p = 0
    i = 0
    while len(q_queue) > 0 and i < len(sorted_activities_idxs):
        # Since sequence weights are normalized, the sum of weights of
        # sequences up to a given sequence is the current percentile
        curr_p = curr_p + seq_norm_weights[sorted_activities_idxs[i]]
        # Set every requested percentile that falls within (prev_p, curr_p]
        while curr_p > curr_q or math.isclose(curr_p, curr_q):
            if math.isclose(curr_p, curr_q):
                # Requested percentile equals current percentile
                p[q_idxs[curr_q]] = activities[sorted_activities_idxs[i]]
            else:
                # Using 'lower' interpolation, so use the one prior
                # (unless it is the first one)
                p[q_idxs[curr_q]] = activities[sorted_activities_idxs[max(i-1, 0)]]
            # q has been set, so remove from queue and move to next q
            q_queue.pop(0)
            if len(q_queue) > 0:
                curr_q = q_queue[0]
            else:
                break
        i += 1

    return p
