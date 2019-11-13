"""Load and use a model to predict activity of a guide-target pair.

This is a crude, temporary module -- it hard codes a path to a separate
directory from which to load modules that enable prediction.
"""

import os
import pickle
import sys

sys.path.append(os.path.join(os.path.expanduser("~"), "adapt-seq-design"))
import predictor

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class Predictor:
    def __init__(self, model, activity_thres=-1.0):
        """
        Args:
            model: model object with a call() function
            activity_thres: call predicted activity >= this threshold
                to be positive
        """
        self.model = model
        self.activity_thres = activity_thres
        self.context_nt = model.context_nt

    def evaluate(self, target_with_context, guide):
        """Determine whether or not a guide-target pair has sufficient
        activity. 

        Args:
            target_with_context: target sequence with self.context_nt
                nt on each side
            guide: guide sequence

        Returns:
            False/True indicating whether the predicted activity of
            the guide-target pair exceeds self.activity_thres
        """
        pred_activity = predictor.pred_from_nt(self.model,
                target_with_context, guide)
        return pred_activity >= self.activity_thres


def construct_predictor(model_path, context_nt=10):
    """Construct a Predictor object.

    Args:
        model_path: path to saved model parameters and weights
        context_nt: nt of context to use on each side of target

    Returns:
        Predictor object
    """
    load_path_params = os.path.join(model_path,
            'model.params.pkl')
    with open(load_path_params, 'rb') as f:
        saved_params = pickle.load(f)
    params = {'dataset': 'cas13', 'cas13_subset': 'exp-and-pos',
            'cas13_regress_only_on_active': True, 'context_nt': context_nt}
    for k, v in saved_params.items():
        params[k] = v

    predictor.set_seed(1)

    model = predictor.load_model_for_cas13_regression_on_active(
            model_path, params)
    return Predictor(model)
