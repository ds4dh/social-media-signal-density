# Models

> 🚧 **Work in Progress** — This folder is under active development. Additional scripts, processed data files, and documentation will be added as the pipeline is finalized.

## Overview

This folder contains the scripts used to **implement**, **train**, and **evaluate** the machine learning models studied in this project. Dataset construction is handled separately — see [`Dataset Construction/`](../Dataset%20Construction) for that stage of the pipeline.

## Workflow

```
FINAL_DATASET/   →  Hyperparameter search →  Model training   →  Evaluation  →  LOGS/
```

## Models

| Model | Input features | Script |
|---|---|---|
| XGBoost (`OurTree`) | Daily scores from fuzzy keyword matching | `OurTree.py` |
| Logistic regression (`OurLinear`) | Same features as `OurTree`, scaled in a pipeline | `OurLinear.py` |
| LightGBM (`OurLightGBM`) | Same features as `OurTree` | `OurLightGBM.py` |
| *(additional models to be added)* | — | — |

## Folder Structure

| File / Folder | Description |
|---|---|
| `LOGS/` | Output folder where results from each run are saved. |
| `main.py` | Main entry point — runs training and evaluation. |
| `OurTree.py` | XGBoost implementation. Expects daily scores built from fuzzy keyword matching (see [`Dataset Construction/`](../Dataset%20Construction)). |
| `OurLinear.py` | Elastic-net logistic regression on the same folds, tuning and evaluation loop as `OurTree`; the estimator is an impute+scale+LogisticRegression pipeline. |
| `OurLightGBM.py` | LightGBM (leaf-wise gradient boosting) on the same loop; translates the XGBoost fit kwargs and loads the OpenMP runtime before importing lightgbm. |
| `hyperparameter_searching_space.py` | Hyperparameter search-space definitions used during optimization. |
| `constants.py` | Shared constants used across the scripts in this folder. |
| `utils.py` | Utility functions. |