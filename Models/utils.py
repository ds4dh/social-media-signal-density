import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score







class TQDMProgressBar:
    """
    Just a small class to be able to add a progress bar when conducting hyperparameter search using Optuna
    """
    def __init__(self, n_trials, desc):
        self.pbar = tqdm(total=n_trials, desc=desc)
    def __call__(self, study, trial):
        self.pbar.update(1)
    def close(self):
        self.pbar.close()


def binary_metrics(predictions, label, dataset="test", to_print=False, threshold=0.5):
    """
    Compute a set of binary classification metrics given the predictions and true labels. The predictions are expected to be probabilities (i.e. output of model.predict_proba()[:, 1])

    """
    
    if to_print:
        print('#'*50)
        print('Performance on the {} set using threshold {}'.format(dataset, threshold))
        print('#'*50)

    metrics = {
        "roc_auc": roc_auc_score(label, predictions),
        "accuracy": accuracy_score(label, predictions > threshold),
        "precision": precision_score(label, predictions > threshold),
        "recall": recall_score(label, predictions > threshold),
        "f1": f1_score(label, predictions > threshold)
    }
    return metrics


import numpy as np

def random_baseline_metrics(label, n_runs=100, strategy="uniform", random_state=42, dataset="test"):
    """
    Compute averaged binary classification metrics for random predictions.

    Provides a baseline to compare a real model against. Two strategies:

    - "uniform": probabilities sampled from Uniform(0, 1). The standard
      "what would a model with no signal do" baseline. Expected AUC ≈ 0.5.
    - "stratified": probabilities sampled from a Bernoulli with p equal to
      the empirical positive rate of `label`. Mimics a classifier that
      knows the base rate but nothing else.

    Parameters
    ----------
    label : array-like
        True binary labels (0/1).
    n_runs : int
        Number of random prediction sets to draw.
    strategy : {"uniform", "stratified"}
        How to generate random predictions.
    random_state : int
        Seed for reproducibility.
    dataset : str
        Label for the printed summary.

    Returns
    -------
    dict
        Mean of each metric across runs, plus standard deviations under
        keys suffixed with "_std".
    """
    label = np.asarray(label)
    rng = np.random.default_rng(random_state)
    n = len(label)
    pos_rate = label.mean()

    runs = []
    for _ in range(n_runs):
        if strategy == "uniform":
            preds = rng.uniform(0.0, 1.0, size=n)
        elif strategy == "stratified":
            preds = rng.binomial(1, pos_rate, size=n).astype(float)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        runs.append(binary_metrics(preds, label, dataset=dataset,
                                   to_print=False, threshold=0.5))

    # Aggregate: mean and std per metric across runs
    keys = runs[0].keys()
    summary = {}
    for k in keys:
        values = np.array([r[k] for r in runs])
        summary[k] = float(values.mean())
        summary[f"{k}_std"] = float(values.std())

    print('#' * 50)
    print(f"Random baseline ({strategy}) on {dataset} set, averaged over {n_runs} runs")
    print('#' * 50)
    for k in keys:
        print(f"{k}: {summary[k]:.4f} ± {summary[f'{k}_std']:.4f}")

    return summary
