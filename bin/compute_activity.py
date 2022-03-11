#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
import numpy as np
from adapt.utils import predict_activity
from adapt.utils.version import get_project_path, get_latest_model_version

__author__ = ('Hayden Metsky <hmetsky@broadinstitute.org>, ' +
              'Priya P. Pillai <ppillai@broadinstitute.org>')


def main(args):
    if not os.path.isfile(args.seq_tsv):
        raise ValueError(("seq_tsv is not a valid file path"))

    if args.predict_activity_model_path:
        cla_path, reg_path = args.predict_activity_model_path
    else:
        dir_path = get_project_path()
        cla_path_all = os.path.join(dir_path, 'models', 'classify',
                                'cas13a')
        reg_path_all = os.path.join(dir_path, 'models', 'regress',
                                'cas13a')
        if len(args.predict_cas13a_activity_model) not in (0,2):
            raise Exception(("If setting versions for "
                "--predict-cas13a-activity-model, both a version for "
                "the classifier and the regressor must be set."))
        if (len(args.predict_cas13a_activity_model) == 0 or
                args.predict_cas13a_activity_model[0] == 'latest'):
            cla_version = get_latest_model_version(cla_path_all)
        else:
            cla_version = args.predict_cas13a_activity_model[0]
        if (len(args.predict_cas13a_activity_model) == 0 or
                args.predict_cas13a_activity_model[1] == 'latest'):
            reg_version = get_latest_model_version(reg_path_all)
        else:
            reg_version = args.predict_cas13a_activity_model[1]
        cla_path = os.path.join(cla_path_all, cla_version)
        reg_path = os.path.join(reg_path_all, reg_version)
    if args.predict_activity_thres:
        # Use specified thresholds on classification and regression
        cla_thres, reg_thres = args.predict_activity_thres
    else:
        # Use default thresholds specified with the model
        cla_thres, reg_thres = None, None

    predictor = predict_activity.Predictor(cla_path, reg_path)

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
    parser.add_argument('--predict-cas13a-activity-model',
        nargs='*',
        help=("Use ADAPT's premade Cas13a model to predict guide-target "
              "activity. Optionally, two arguments can be included to indicate "
              "version number, in the format 'v1_0' or 'latest'. Versions "
              "will default to latest. Required if "
              "predict-activity-model-path is not set."))
    parser.add_argument('-p', '--predict-activity-model-path',
        nargs=2,
        help=("Paths to directories containing serialized models in "
              "TensorFlow's SavedModel format for predicting guide-target "
              "activity. There are two arguments: (1) classification "
              "model to determine which guides are active; (2) regression "
              "model, which is used to determine which guides (among "
              "active ones) are highly active. The models/ directory "
              "contains example models. Required if "
              "predict-cas13a-activity-model is not set."))
    parser.add_argument('--predict-activity-thres',
        type=float,
        nargs=2,
        help=("Thresholds to use for decisions on output of predictive "
            "models. There are two arguments: (1) classification threshold "
            "for deciding which guide-target pairs are active (in [0,1], "
            "where higher values have higher precision but less recall); "
            "(2) regression threshold for deciding which guide-target pairs "
            "are highly active (>= 0, where higher values limit the number "
            "determined to be highly active). If not set, ADAPT uses default "
            "thresholds stored with the models."))

    args = parser.parse_args()
    main(args)
