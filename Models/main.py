from OurTree import OurTreeModel
from sklearn.decomposition import PCA
from constant import *
import pandas as pd
import os
import argparse
from dataset import construct_dataset
import pickle
from basic_baselines import compute_metrics_baselines



def parse_args():

    parser = argparse.ArgumentParser()

    # random seed
    parser.add_argument('--random_seed', action='store', type=int, default=15, 
                        help='random seed used to conduct these experiments.')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds to run. Results are aggregated as mean ± std.')
    parser.add_argument('--main_focus', type=str, default='patients', help='Define the main focus. Can be patients or calls.')
    
    ##########################
    ## Feature set considered
    ###########################
    parser.add_argument('--feature_set', type=str, nargs='+', default=['patients', 'time'],
                        help='Define the feature set to be used for training the model. Potential keywords: calls/patients, tweets.')
    
    parser.add_argument('--pca_variance_threshold', type=float, default=0.30, help=('Minimum cumulative explained variance for PCA reduction of BERT '
        'embeddings. Fit on training data only within each CV fold. Default 0.85 (~26-28 components). Do not exceed 0.90 given '
        'fold 1 sample size (430 training days).'))
    
    
    #########################
    ## Approach considered
    ##########################
    # ML model parameters
    parser.add_argument('--approach', action='store', type=str, default="matching", help='Method considered for prediction. ' \
    'Potential values: matching, BERT, moving_average, same_as_last, arima.')
    parser.add_argument('--num_trials', action='store', type=int, default=100, help='Number of trials for hyperparameter search.')
    parser.add_argument('--n_jobs', action='store', type=int, default=N_JOBS,
                        help='Threads per XGBoost fit. Keep well below the logical CPU count: '
                             'on this WSL2 host, n_jobs=-1 (22 threads on 11 physical cores) makes '
                             'each fit ~800x slower than single-threaded.')

    # specific for baselines
    parser.add_argument('--moving_average_window', type=int, default=4)
    parser.add_argument('--arima_order', type=int, nargs=3, default=[1, 1, 1],
                    metavar=('P', 'D', 'Q'))
    
    ##################################
    ## Labels type considered
    #################################
    parser.add_argument('--labeling_strategy', action='store', type=str, default='surge_detection', 
                        help='Potential values: weekly_calls OR weekly_trends OR surge_detection.')
    parser.add_argument('--horizon', action='store', default=3)


    ##########################
    ## Dataset splitting parameters
    ##########################
    parser.add_argument('--dataset_split_strategy', action='store', type=str, default="time_based",
                        help='Define the strategy for splitting the dataset into train, validation and test sets. ' \
                        'Potential values include: "time_based" (i.e. data is chronologically ordered before splitting ).')
    parser.add_argument('--train_set_proportion', action='store', type=float, default=0.7,
                        help='Proportion of the dataset to be included in the train set. The rest of the data will be split between the validation and test sets according to the "dataset_split_strategy" argument.')
    parser.add_argument('--validation_set_proportion', action='store', type=float, default=0.15,
                        help='Proportion of the dataset to be included in the validation set. The rest of the data will be included in the test set.')
    parser.add_argument('--test_set_proportion', action='store', type=float, default=0.15,
                        help='Proportion of the dataset to be included in the test set. The rest of the data will be split between the train and validation sets according to the "dataset_split_strategy" argument.')

    args = parser.parse_args()

    # On this WSL2 host, spanning all logical CPUs makes every XGBoost fit
    # orders of magnitude slower (see N_JOBS in constant.py).
    if args.n_jobs == -1 or args.n_jobs > physical_cores():
        print(f"[main] n_jobs={args.n_jobs} oversubscribes the "
              f"{os.cpu_count()} logical / {physical_cores()} physical CPUs, "
              f"which collapses XGBoost throughput. Using {N_JOBS} instead.")
        args.n_jobs = N_JOBS

    return args



def main():

    # loading arguments
    args = parse_args()

    # construct 3 folds datasets
    folds = construct_dataset(args=args)

    # Generate the list of seeds deterministically from the base seed
    base_seed = args.random_seed
    seeds = [base_seed + i for i in range(args.num_seeds)]

    all_folds_metrics = []

    # compute model performance on each fold
    for fold in folds:
        fold_idx = fold["fold"]
        fold_dataset = {
            "train":      fold["train"],
            "validation": fold["validation"],
            "test":       fold["test"],
        }

        print(f"\n{'#'*60}")
        print(f"FOLD {fold_idx} / {len(folds)}")
        print(f"{'#'*60}")

        fold_metrics = []
        fold_metrics = []

        if args.approach in BASELINE_APPROACHES:
            # Baselines are deterministic — run once, replicate for reporting consistency
            print(f"\n{'='*60}\nRunning baseline: {args.approach}\n{'='*60}")
            metrics = compute_metrics_baselines(param=args, dataset=fold_dataset)
            # Replicate across seeds so downstream summarisation code works unchanged
            fold_metrics = [metrics] * len(seeds)

        else:
            for seed in seeds:
                print(f"\n{'='*60}\nRunning with seed {seed}\n{'='*60}")
                args.random_seed = seed
                model = OurTreeModel(param=args, dataset=fold_dataset)
                metrics = model.hyperparameter_search_and_evaluation()
                fold_metrics.append(metrics)


        print(f"\n--- Fold {fold_idx} results ---")

        # print performance for this specific fold
        _summarize_across_seeds(fold_metrics, seeds)

        # save performance on this fold
        all_folds_metrics.append({"fold": fold_idx, "metrics": fold_metrics})

    # summarize global model performance
    _summarize_across_folds(all_folds_metrics, seeds)


def _summarize_across_seeds(fold_metrics: list[dict], seeds: list[int]) -> None:
    """Print aggregated metrics across seeds for a single fold."""
    df_metrics = pd.DataFrame(
        fold_metrics,
        index=[f"seed_{s}" for s in seeds]
    )
    print(f"\n{'='*60}\nAggregated results across {len(seeds)} seeds\n{'='*60}")
    print(df_metrics.round(4))
    print("\nMean ± Std:")
    for col in df_metrics.columns:
        mean = df_metrics[col].mean()
        std  = df_metrics[col].std()
        print(f"  {col}: {mean:.4f} ± {std:.4f}")


def _summarize_across_folds(
    all_folds_metrics: list[dict],
    seeds: list[int],
) -> None:
    """
    Print per-fold summary and global mean across all folds and seeds.
    """
    print(f"\n{'#'*60}")
    print(f"SUMMARY ACROSS ALL FOLDS")
    print(f"{'#'*60}")

    # One row per (fold, seed)
    rows = []
    for fold_entry in all_folds_metrics:
        fold_idx = fold_entry["fold"]
        for seed, metrics in zip(seeds, fold_entry["metrics"]):
            rows.append({"fold": fold_idx, "seed": seed, **metrics})

    df_all = pd.DataFrame(rows)
    metric_cols = [c for c in df_all.columns if c not in ("fold", "seed")]

    # Per-fold mean across seeds
    print("\nPer-fold mean (averaged across seeds):")
    fold_means = df_all.groupby("fold")[metric_cols].mean().round(4)
    print(fold_means)

    # Global mean and std across all folds and seeds
    print("\nGlobal mean ± std (across all folds and seeds):")
    for col in metric_cols:
        mean = df_all[col].mean()
        std  = df_all[col].std()
        print(f"  {col}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
