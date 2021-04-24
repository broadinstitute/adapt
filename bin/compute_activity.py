import argparse
import os
import tensorflow as tf
import numpy as np
from adapt.utils import predict_activity

__author__ = ('Hayden Metsky <hmetsky@broadinstitute.org>, ' +
              'Priya P. Pillai <ppillai@broadinstitute.org>')


def main(args):
    if not os.path.isfile(args.seq_tsv):
        raise ValueError(("seq_tsv is not a valid file path"))

    predictor = predict_activity.Predictor(
            *args.predict_activity_model_path)

    target_length = 2*predictor.context_nt + predictor.guide_length

    pairs = []
    with open(args.seq_tsv) as f:
        for line in f:
            ls = line.rstrip().split('\t')
            if len(ls[0]) != target_length:
                raise ValueError("%s is not the correct target length for "
                    "this model; it should be %i" % (ls[0], target_length))
            if len(ls[1]) != predictor.guide_length:
                raise ValueError("%s is not the correct guide length for "
                                 "this model; it should be %i" %
                                 (ls[1], predictor.guide_length))
            pairs.append((ls[0], ls[1]))

    activities = predictor.compute_activity(predictor.context_nt, pairs)
    activities_with_order = list(zip(activities, range(len(activities))))
    activities_with_order.sort(reverse=True, key=lambda a: a[0])

    with open(args.out_tsv, 'w') as outf:
        # Write a header
        outf.write('\t'.join(['original-order',
                              'guide-target-sequences',
                              'full-target-sequences',
                              'guide-predicted-activity']) +
                   '\n')

        for (activity, original_order) in activities_with_order:
            line = [original_order+1, *pairs[original_order], activity]
            outf.write('\t'.join([str(x) for x in line]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input sequences
    parser.add_argument('seq_tsv',
        help=("Path to TSV of sequences, where the first column is the "
              "target sequence (with appropriate context for the model) "
              "and the second column is the guide sequence."))
    # Output file path
    parser.add_argument('-o', '--out-tsv',
        required=True,
        help=("Path to output TSV."))
    # Use models to predict activity
    parser.add_argument('-p', '--predict-activity-model-path',
        nargs=2, required=True,
        help=("Paths to directories containing serialized models in "
              "TensorFlow's SavedModel format for predicting guide-target "
              "activity. There are two arguments: (1) classification "
              "model to determine which guides are active; (2) regression "
              "model, which is used to determine which guides (among "
              "active ones) are highly active. The models/ directory "
              "contains example models. Required."))

    args = parser.parse_args()
    main(args)
