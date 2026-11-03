from turtle import pos
from scipy.stats import spearmanr
import pandas as pd
import argparse
import os

from constant import *
from typing import Tuple
import optuna
from optuna.samplers import TPESampler
from utils import *
import pickle
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
from hyperparameter_searching_space import build_hyperparameter_search_space
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


# to suppress annoying optuna default logs
optuna.logging.set_verbosity(optuna.logging.WARNING)



class OurTreeModel:
    #: Hyperparameter search space used by :meth:`_hyperparameter_search`. A model
    #: built around another estimator family overrides it (see OurLinear.py) while
    #: keeping this training, tuning and evaluation loop.
    _build_search_space = staticmethod(build_hyperparameter_search_space)

    def __init__(self, param: argparse.Namespace, dataset: dict[str, pd.DataFrame], logdir=None):
        """
        Initialize an Tree model customized for our task.

        This constructor:
          - Stores the general configuration parameters.
          - Save the training, validation and test sets.
          - Creates the logging directory for outputs.
          - Initializes the appropriate tree model.

        Parameters
        ----------
            param: general config (see main.py)
            dataset: dictionary containing the three datasets with the keywords "training", "validation" and "test".
        """
        
        # general config
        self.param = param

        # training, validation, test sets
        self.X_train, self.y_train = self._split_feature_label(dataset['train'])
        self.X_val, self.y_val = self._split_feature_label(dataset['validation'])
        self.X_test, self.y_test = self._split_feature_label(dataset['test'])

        # log dir
        if logdir is None:
            self._log_dir = os.path.join(LOG_DIR, self.param.approach, str(self.param.random_seed)) 
            os.makedirs(self._log_dir, exist_ok=True) 
        else:
            self._log_dir = logdir

        # check the task: regression or classification
        self._is_regression = (getattr(param, 'labeling_strategy', 'weekly_trends') == 'weekly_calls')

        # initialization of the tree model
        self.model = self._construct_model()

        
        
    def _construct_model(self, hyperparameter=None):
        """
        Construct a tree-based model. Currently, only XGBoost is implemented.

        Builds and returns an unfitted tree model. If ``hyperparameter`` is provided, its items are passed as keyword arguments;
        otherwise, library defaults are used.

        Parameters
        ----------
        hyperparameter : dict, optional
            Keyword arguments for the tree model

        Returns
        -------
        An unfitted tree model
        """
        if self.param.approach not in ('matching', 'BERT'):
            raise NotImplementedError("Currently only matching or BERT approaches are supported.")
        
        ModelCls = XGBRegressor if self._is_regression else XGBClassifier
        params = dict(hyperparameter) if hyperparameter else {}
        # study.best_params only carries the *suggested* values, so fixed knobs
        # like n_jobs are absent there and would silently fall back to XGBoost's
        # default (all logical CPUs). Apply them here so every construction path,
        # refit and load_and_evaluate included, is covered.
        params.setdefault("n_jobs", getattr(self.param, "n_jobs", N_JOBS))
        params.setdefault("random_state", self.param.random_seed)
        params.setdefault("verbosity", 0)
        return ModelCls(**params)


    def _split_feature_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split a DataFrame into features and label.

        This method separates the target label column from the rest of the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing both features and the label column.

        Returns
        -------
        feature : pandas.DataFrame
            DataFrame containing all columns except the column "label"
        label : pandas.Series
            Series containing the column called "label".
        """
        # extract label
        label = df["label"]
        # drop label
        feature =  df.drop(columns=["label", 'date'])
        return feature, label
    

    def hyperparameter_search_and_evaluation(self) -> None:
        """
        Perform hyperparameter optimization and evaluate the final model.

        This method:
            - Runs hyperparameter optimization with Optuna.
            - Trains an tree-based model using the best hyperparameters found.
            - Evaluates the trained model on the test set.
            - Saves the best hyperparameters for future use.

        Returns
        -------
        None
        """
        
        # conduct hyperpameter search using Optuna
        study = self._hyperparameter_search()

        # train a tree-based model using the best hyperparemter
        best_param = study.best_params

        # construct a tree model using these best hyperparameter
        self.model = self._construct_model(best_param)
  
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)

        # evaluation
        test_metrics = self._model_evaluation()

        # hyperparameter saving
        self._save_hyperparameters(study=study)

        return test_metrics


    def _hyperparameter_search(self) -> optuna.study.Study:

        if not self._is_regression:
            neg, pos   = (self.y_train == 0).sum(), (self.y_train == 1).sum()
            base_ratio = neg / pos
        else:
            base_ratio = None

        def objective(trial):
            params = self._build_search_space(
                self.param, trial, base_ratio
            )
            model = self._construct_model(params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )

            if self._is_regression:
                y_pred = model.predict(self.X_val)
                # Return negative RMSE so Optuna maximises
                return -np.sqrt(mean_squared_error(self.y_val, y_pred))
            else:
                y_proba = model.predict_proba(self.X_val)[:, 1]
                return roc_auc_score(self.y_val, y_proba)

        sampler      = TPESampler(seed=self.param.random_seed)
        study        = optuna.create_study(direction="maximize", sampler=sampler)
        progress_bar = TQDMProgressBar(self.param.num_trials, desc="Hyperparameter search")
        study.optimize(
            objective,
            n_trials=self.param.num_trials,
            callbacks=[progress_bar],
        )
        progress_bar.close()
        return study


    def _model_evaluation(self) -> dict:
        save_path = os.path.join(
            self._log_dir,
            f"{self.param.approach}_test_predictions.pkl"
        )
        if self._is_regression:
            return self._regression_evaluation(save_path)
        else:
            return self._classification_evaluation(save_path)
    

    def _regression_evaluation(self, save_path: str) -> dict:
        """Evaluate a regression model on the test set."""
        y_pred = self.model.predict(self.X_test)

        with open(save_path, "wb") as f:
            pickle.dump({"predictions": y_pred, "labels": self.y_test}, f)

        rmse     = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae      = mean_absolute_error(self.y_test, y_pred)
        r2       = r2_score(self.y_test, y_pred)
        spearman, _ = spearmanr(self.y_test, y_pred)

        nonzero = self.y_test != 0
        mape = (
            np.abs(
                (self.y_test[nonzero] - y_pred[nonzero]) / self.y_test[nonzero]
            ).mean() * 100
        )

        print(
            f"\n[regression_evaluation]"
            f"\n  RMSE     : {rmse:.4f}"
            f"\n  MAE      : {mae:.4f}"
            f"\n  MAPE     : {mape:.2f}%"
            f"\n  R²       : {r2:.4f}"
            f"\n  Spearman : {spearman:.4f}"
        )

        return {
            "rmse":     rmse,
            "mae":      mae,
            "mape":     mape,
            "r2":       r2,
            "spearman": spearman,
        }
    
    def _classification_evaluation(self, save_path: str) -> dict:
        """Evaluate a classification model on the test set."""
        with open(save_path, "wb") as f:
            pickle.dump({
                "proba predictions": self.model.predict_proba(self.X_test)[:, 1],
                "labels": self.y_test
            }, f)

        best_threshold = self._find_best_threshold()

        y_val_proba  = self.model.predict_proba(self.X_val)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        print(f"Val  probas: min={y_val_proba.min():.3f}, "
              f"max={y_val_proba.max():.3f}, mean={y_val_proba.mean():.3f}")
        print(f"Test probas: min={y_test_proba.min():.3f}, "
            f"max={y_test_proba.max():.3f}, mean={y_test_proba.mean():.3f}")
        print(f"Threshold: {best_threshold:.4f}")
        print(f"Positives predicted on val:  "
              f"{(y_val_proba >= best_threshold).sum()} / {len(y_val_proba)}")
        print(f"Positives predicted on test: "
             f"{(y_test_proba >= best_threshold).sum()} / {len(y_test_proba)}")
        print(f"True positives in val:  {self.y_val.sum()} / {len(self.y_val)}")
        print(f"True positives in test: {self.y_test.sum()} / {len(self.y_test)}")

        return binary_metrics(
            predictions=self.model.predict_proba(self.X_test)[:, 1],
            label=self.y_test,
            dataset="test",
            threshold=best_threshold,
        )

    def _find_best_threshold(self, print_info: bool = True) -> float:
        if self.param.approach == 'matching':
            return self._test_threshold_matching(print_info=print_info)
        

        """Find the decision threshold that maximizes F1 on the validation set."""
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
        prec, rec, thresholds = precision_recall_curve(self.y_val, y_val_proba)

        # precision_recall_curve returns one fewer threshold than prec/rec entries;
        # drop the last prec/rec point (which corresponds to "predict everything positive").
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        best_idx = f1s.argmax()
        best_threshold = thresholds[best_idx]
        if print_info:
            print(f"[threshold] Best F1 on validation: {f1s[best_idx]:.4f} "
                f"at threshold {best_threshold:.4f}")
        return float(best_threshold)
    
    
    def _test_threshold_matching(self, print_info: bool = True) -> float:
        """
        Use a rank-based threshold: predict the top-k% as positive,
        matching the known positive rate in the training set.
        Calibration-invariant — works regardless of scale_pos_weight.
        """
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
    
        # Use the actual positive rate from training set
        positive_rate = self.y_train.mean()
        print('Positive rate in training set: {:.3f}'.format(positive_rate))
        threshold = np.percentile(y_val_proba, 100 * (1 - positive_rate))
    
        if print_info:
            y_pred = (y_val_proba >= threshold).astype(int)
            f1 = f1_score(self.y_val, y_pred)
            print(f"[threshold] Positive rate={positive_rate:.3f}, "f"threshold={threshold:.4f}, val F1={f1:.4f}")
    
        return float(threshold)


    def _save_hyperparameters(self, study: optuna.Study) -> None:
        """
        Save the best hyperparameters from an Optuna study.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object containing the best hyperparameters in its ``study.best_params`` attribute.

        Returns
        -------
        None
        """

        suffix = self.param.approach

        fname = f'{suffix}_best_hyperparam.pkl'
        with open(os.path.join(self._log_dir, fname), "wb") as f:
            pickle.dump(study.best_params, f)
        # print(f"Best {self.param.model} hyperparameters have been saved to {fname}.")
    

    def load_and_evaluate(self) -> None:
        """
        Load the best hyperparameters, retrain the model, and evaluate it.

        Returns
        -------
        None
        """
        # loading model
        best_params = self._load_hyperparameters()

        # model initialization and training
        self.model = self._construct_model(best_params)

        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)

        # evaluation
        test_metrics = self._model_evaluation()
        return test_metrics

    def _load_hyperparameters(self) -> dict:
        """
        Load the best hyperparameters that is expected to be in self._log_dir + /xgb_best_hyperparam.pkl.

        Returns
        -------
        dict
            Dictionary containing the best model hyperparameters.
        """

        suffix = self.param.approach

        file_path = os.path.join(self._log_dir, f"{suffix}_best_hyperparam.pkl")
        with open(file_path, "rb") as f:
            best_params = pickle.load(f)
        return best_params
