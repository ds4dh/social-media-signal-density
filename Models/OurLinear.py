"""Logistic-regression counterpart of OurTree.OurTreeModel.

This module changes only the estimator family: the Optuna search, the refit with
the best parameters, the threshold selection and every evaluation metric stay the
inherited ones from ``OurTreeModel``. Two things have to differ from a tree:

  - the hyperparameter space (linear and tree hyperparameters are disjoint), so
    ``_build_search_space`` points at ``build_logistic_search_space``;
  - the input pipeline. A linear model is scale-sensitive and cannot read the NaNs
    that the lag columns carry in their first rows, so the estimator is a
    ``Pipeline`` that imputes the training median, standardises, and only then fits
    the logistic regression. Both steps are fitted on the training days of the fold
    inside this pipeline, so they obey the same no-leakage rule as the model itself.
"""


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hyperparameter_searching_space import build_logistic_search_space
from OurTree import OurTreeModel


class _XgbFitTolerantPipeline(Pipeline):
    """Pipeline that ignores the XGBoost-specific fit kwargs of the shared loop.

    ``OurTreeModel`` fits with ``eval_set=[(X_val, y_val)]`` and ``verbose=False``,
    which are early-stopping arguments of ``XGBClassifier.fit`` and unknown to
    ``sklearn``. Dropping them keeps a single fit call for both model classes.
    """

    def fit(self, X, y=None, **fit_params):
        for unused in ("eval_set", "verbose"):
            fit_params.pop(unused, None)
        return super().fit(X, y)


class OurLinearModel(OurTreeModel):
    """Logistic regression on the same folds, tuning and evaluation as OurTreeModel.

    Parameters
    ----------
    param : argparse.Namespace
        Same config as :class:`OurTreeModel` (approach, random_seed, num_trials, ...).
    dataset : dict
        Dictionary containing the three datasets with the keywords "train",
        "validation" and "test".
    logdir : str, optional
        Directory for the saved hyperparameters; defaults to the class default.
    """

    _build_search_space = staticmethod(build_logistic_search_space)

    def _construct_model(self, hyperparameter=None):
        """
        Construct an unsupervised-preprocessing logistic regression.

        Builds and returns an unfitted ``Pipeline`` (median imputation, scaling,
        logistic regression). If ``hyperparameter`` is provided, its items are passed
        as keyword arguments to ``LogisticRegression``; otherwise, library defaults
        are used.

        Parameters
        ----------
        hyperparameter : dict, optional
            Keyword arguments for ``LogisticRegression``.

        Returns
        -------
        sklearn.pipeline.Pipeline
            An unfitted model exposing ``fit``/``predict_proba`` like ``XGBClassifier``.
        """
        if self.param.approach not in ('matching', 'BERT'):
            raise NotImplementedError("Currently only matching or BERT approaches are supported.")
        if self._is_regression:
            raise NotImplementedError(
                "OurLinearModel only supports the binary classification task; "
                "use OurTreeModel for the 'weekly_calls' labeling strategy."
            )

        params = dict(hyperparameter) if hyperparameter else {}
        # The imbalance knob is searched under the XGBoost name; LogisticRegression
        # weights classes instead of the positive logit.
        pos_weight = params.pop("scale_pos_weight", None)
        if pos_weight is not None:
            params["class_weight"] = {1: pos_weight}
        # As in OurTreeModel._construct_model, study.best_params holds only the
        # *suggested* values, so the fixed keys of the space are re-applied here
        # and every construction path ends up with the same solver and seed.
        params.setdefault("solver", "saga")
        params.setdefault("random_state", self.param.random_seed)
        return _XgbFitTolerantPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**params)),
        ])
