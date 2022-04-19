"""Utilities involving weighting sequences.
"""

import logging
import math
import numpy as np

from adapt.prepare import ncbi_neighbors

__author__ = 'Priya P. Pillai <ppillai@broadinstitute.org>'


def weight_by_log_subtaxa(accessions, subtaxa_rank):
    """Weight sequences by the log of the number of sequences in their subtaxa

    Args:
        accessions: logging level below which logging messages are ignored
        subtaxa_rank: level of taxonomy at which to take the log of the number
            of sequences

    Returns:
        dictionary {accession: unnormalized weight}
    """
    taxonomies = ncbi_neighbors.fetch_taxonomies(accessions)
    subtaxa_count = {}
    acc_to_subtaxa = {}
    for acc, taxonomy in taxonomies.items():
        for subtaxon in subtaxa_count:
            if subtaxon in taxonomy:
                subtaxa_count[subtaxon] += 1
                acc_to_subtaxa[acc] = subtaxon
                break
        if acc not in acc_to_subtaxa:
            subtaxon = ncbi_neighbors.get_rank(taxonomy[-1],
                                               subtaxa_rank)
            subtaxa_count[subtaxon] = 1
            acc_to_subtaxa[acc] = subtaxon

    # Each subtaxa's weight should total to the log of the number of sequences
    # in that subtaxa
    subtaxa_weights = {subtaxon: math.log(subtaxa_count[subtaxon])
                       for subtaxon in subtaxa_count}

    # # Normalize by total of the subtaxa weights
    # normalization_factor = sum(subtaxa_weights.values())
    # subtaxa_weights_norm = {subtaxon: (subtaxa_weights[subtaxon] /
    #                                    normalization_factor)
    #                         for subtaxon in subtaxa_weights}

    # Divide by the number of sequences to determine the per sequence weight
    subtaxa_weights_per_seq = {subtaxon: (subtaxa_weights[subtaxon] /
                                          subtaxa_count[subtaxon])
                               for subtaxon in subtaxa_weights}

    # Return a dictionary that matches accessions to their weight
    return {acc: subtaxa_weights_per_seq[acc_to_subtaxa[acc]]
            for acc in acc_to_subtaxa}


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
            if math.isclose(curr_p, q_i):
                p_idxs[unset_q[q_i]] = i
            elif curr_p < q_i:
                continue
            else:
                p_idxs[unset_q[q_i]] = i-1
            set_qs.add(q_i)
        for set_q in set_qs:
            unset_q.pop(set_q)
        i += 1

    return [activities[sorted_activities_idxs[p_idx]] for p_idx in p_idxs]
