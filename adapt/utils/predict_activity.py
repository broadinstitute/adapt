"""Load and use a model to predict activity of a guide-target pair.

The function construct_predictor() has a crude and temporary approach, with
a hard-coded path to a separate directory from which to load modules that
enable prediction. That implementation is currently commented out.
"""

import os
import pickle
import sys

__author__ = 'Hayden Metsky <hayden@mit.edu>'


# Use the top ~25% of guide-target pairs as the default activity threshold
DEFAULT_ACTIVITY_THRES = -0.85


class Predictor:
    """This calls the activity model and memoizes results.
    """

    def __init__(self, model, pred_from_nt_fn, activity_thres=None):
        """
        Args:
            model: model object with a call() function
            pred_from_nt_fn: function that accepts list of guide-target pairs
                (in nucleotide space) and outputs predicted activities
            activity_thres: call predicted activity >= this threshold
                to be positive; if None, use default
        """
        self.model = model
        self.pred_from_nt_fn = pred_from_nt_fn
        if activity_thres is None:
            activity_thres = DEFAULT_ACTIVITY_THRES
        self.activity_thres = activity_thres
        self.context_nt = model.context_nt

        # Memoize evaluations, organized by guide start position:
        #   {guide start: {pair: result of evaluation}}
        self._memoized_evaluations = {}

    def evaluate(self, start_pos, pairs):
        """Evaluate guide-target pairs, with memoized results.

        Args:
            start_pos: start position of all guides in pairs
            pairs: list of tuples (target with context, guide)

        Returns:
            results of self.call() on these pairs
        """
        if start_pos not in self._memoized_evaluations:
            self._memoized_evaluations[start_pos] = {}
        mem = self._memoized_evaluations[start_pos]

        # Determine which pairs do not have memoized results, and call
        # these
        unique_pairs_to_evaluate = [pair for pair in set(pairs) if pair not in mem]
        evaluations = self.call(unique_pairs_to_evaluate)
        for pair, e in zip(unique_pairs_to_evaluate, evaluations):
            mem[pair] = e

        return [mem[pair] for pair in pairs]

    def cleanup_memoized(self, start_pos):
        """Cleanup memoizations no longer needed at a start position.

        Args:
            start_pos: start position of all guides to remove
        """
        if start_pos in self._memoized_evaluations:
            del self._memoized_evaluations[start_pos]

    def call(self, pairs):
        """Determine whether or not a guide-target pair has sufficient
        activity. 

        Args:
            pairs: list of tuples (target with context, guide)

        Returns:
            list of False/True indicating whether the predicted activity of
            each guide-target pair exceeds self.activity_thres
        """
        if len(pairs) == 0:
            return []
        pred_activity = self.pred_from_nt_fn(pairs)
        return [pa >= self.activity_thres for pa in pred_activity]


def construct_predictor(model_path, context_nt=10, activity_thres=None):
    """Construct a Predictor object.

    Args:
        model_path: path to saved model parameters and weights
        context_nt: nt of context to use on each side of target
        activity_thres: call predicted activity >= this threshold to be
            positive; if None, use default

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

    # Load predictor module from separate directory
    sys.path.append(os.path.join("/ebs/dgd-analysis", "adapt-seq-design-prod"))
    import predictor

    predictor.set_seed(1)

    # Load model and define function for predicting from nucleotide sequence
    model = predictor.load_model_for_cas13_regression_on_active(
            model_path, params)
    def pred_from_nt(pairs):
        return predictor.pred_from_nt(model, pairs)

    return Predictor(model, pred_from_nt, activity_thres=activity_thres)
