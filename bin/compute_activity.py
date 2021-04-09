import argparse
import os
import tensorflow as tf
import numpy as np
from adapt.utils.predict_activity import FASTA_CODES, onehot_idx, onehot

__author__ = 'Hayden Metsky <hmetsky@broadinstitute.org>, Priya P. Pillai <ppillai@broadinstitute.org>'


def model_input_from_nt(pairs, context_nt):
    """Create one-hot input to models from nucleotide sequence.

    Args:
        pairs: list of tuples (target with context, guide)

    Returns:
        list of one-hot encodings for each pair
    """
    if len(pairs) == 0:
        return []

    l = 2*context_nt + len(pairs[0][1])
    x = np.empty((len(pairs), l, 8), dtype='f')
    for i, (target_with_context, guide) in enumerate(pairs):
        # Determine one-hot encodings -- i.e., an input vector
        input_vec = []
        for pos in range(context_nt):
            v_target = onehot(target_with_context[pos])
            v_guide = [0, 0, 0, 0]
            input_vec += [v_target + v_guide]
        for pos in range(len(guide)):
            v_target = onehot(target_with_context[context_nt + pos])
            v_guide = onehot(guide[pos])
            input_vec += [v_target + v_guide]
        for pos in range(context_nt):
            v_target = onehot(target_with_context[context_nt + len(guide) + pos])
            v_guide = [0, 0, 0, 0]
            input_vec += [v_target + v_guide]
        input_vec = np.array(input_vec, dtype='f')
        x[i] = input_vec
    return x


def predict_from_onehot(model, pairs_onehot):
    """Predict activity, from one-hot encoded nucleotide sequence,
    using a model.

    Args:
        model: model object (serialized SavedModel) that has a
            call() function
        pairs_onehot: list of one-hot encoded pairs of target and guide

    Returns:
        list of outputs (float) directly from model, with one
        value per item in pairs
    """
    pred_activity = model.call(pairs_onehot, training=False)
    pred_activity = [p[0] for p in pred_activity.numpy()]
    return pred_activity


def main(args):
    if not os.path.isfile(args.seq_tsv):
        raise ValueError(("seq_tsv is not a valid file path"))
    if not os.path.isdir(args.regression_model):
        raise ValueError(("regression_model is not a valid directory path"))
    regression_guide_length_path = os.path.join(
            args.regression_model,
            'assets.extra/guide_length.arg')
    if not os.path.isfile(regression_guide_length_path):
        raise ValueError(("Unknown guide_length for regression model; "
            "the model should have a assets.extra/guide_length.arg file"))
    regression_context_nt_path = os.path.join(
            args.regression_model,
            'assets.extra/context_nt.arg')
    if not os.path.isfile(regression_context_nt_path):
        raise ValueError(("Unknown context_nt for regression model; "
            "the model should have a assets.extra/context_nt.arg file"))

    with open(regression_guide_length_path) as f:
        guide_length = int(f.readline().strip())
    with open(regression_context_nt_path) as f:
        context_length = int(f.readline().strip())

    target_length = 2*context_length + guide_length

    pairs = []
    with open(args.seq_tsv) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            if len(ls[0]) != target_length:
                raise ValueError("%s is not the correct target length for "
                    "this model; it should be %i" %(ls[0], target_length))
            if len(ls[1]) != guide_length:
                raise ValueError("%s is not the correct guide length for "
                    "this model; it should be %i" %(ls[1], guide_length))
            pairs.append((ls[0], ls[1]))

    pairs_onehot = model_input_from_nt(pairs, context_length)

    regression_model = tf.keras.models.load_model(args.regression_model)

    activity_values = predict_from_onehot(regression_model, pairs_onehot)

    for activity_value in activity_values:
        print(activity_value+4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input alignment and output file
    parser.add_argument('seq_tsv',
            help=("Path to TSV of sequences, where the first column is the target sequence and the second column is the guide sequence"))
    parser.add_argument('regression_model',
            help=("Path to the directory of the regression model"))

    args = parser.parse_args()
    main(args)
