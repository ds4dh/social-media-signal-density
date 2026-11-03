"""LightGBM counterpart of OurTree.OurTreeModel.

Like :mod:`OurLinear`, this module changes only the estimator family: the Optuna
search, the refit with the best parameters, the threshold selection and every
evaluation metric stay the inherited ones from ``OurTreeModel``. Three things
have to differ from an XGBoost tree:

  - the hyperparameter space, so ``_build_search_space`` points at
    ``build_lightgbm_search_space``;
  - the import. ``lib_lightgbm.so`` declares ``NEEDED libgomp.so.1`` and carries
    no RUNPATH, so the OpenMP runtime must already be resolvable when lightgbm
    is dlopened (see :func:`_load_openmp_runtime`);
  - the fit signature. The shared loop fits with the XGBoost arguments
    ``eval_set=[(X_val, y_val)]`` and ``verbose=False``; LightGBM renamed the
    former to ``eval_X``/``eval_y`` and removed the latter, so the estimators
    below translate them.
"""

import ctypes
import ctypes.util
import glob
import os
import site


def _openmp_candidates():
    """Absolute paths of every OpenMP runtime shipped by an installed wheel."""
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in roots:
        for pattern in ("*/lib/libgomp.so.1", "*/lib64/libgomp.so.1",
                        "*/lib/libiomp5.so", "*/lib/libomp.so"):
            yield from sorted(glob.glob(os.path.join(root, pattern)))


def _load_openmp_runtime():
    """Make ``libgomp.so.1`` resolvable before lightgbm is imported.

    Ubuntu ships libgomp as the apt package ``libgomp1``; where that is absent
    and cannot be installed, wheels that bundle it are the only copy available.
    dlopening one of those with ``RTLD_GLOBAL`` puts it in the global symbol
    namespace, and the loader then satisfies lightgbm's ``NEEDED`` entry from the
    already-loaded object instead of searching the filesystem.

    Raises
    ------
    ImportError
        When no OpenMP runtime can be found, with the fix for the two cases.
    """
    if ctypes.util.find_library("gomp") is not None:
        return  # system library, or an LD_LIBRARY_PATH entry
    for path in _openmp_candidates():
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    raise ImportError(
        "LightGBM needs the OpenMP runtime libgomp.so.1, which is not resolvable "
        "here. Install it with 'sudo apt-get install libgomp1', or point "
        "LD_LIBRARY_PATH at a wheel that bundles the library (a torch "
        "installation keeps one in torch/lib)."
    )


# Must run before lightgbm: after the failed dlopen the interpreter caches the
# missing library, so loading OpenMP any later does not help.
_load_openmp_runtime()

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from lightgbm import log_evaluation  # noqa: E402

from constant import N_JOBS  # noqa: E402
from OurTree import OurTreeModel  # noqa: E402
from hyperparameter_searching_space import build_lightgbm_search_space  # noqa: E402


def _fit_kwargs(fit_params: dict) -> dict:
    """Translate the XGBoost fit kwargs of the shared loop to LightGBM names.

    ``eval_set`` still works but is deprecated since 4.6, and ``verbose`` was
    removed outright (it raises ``TypeError``), so the pair is rewritten to
    ``eval_X``/``eval_y`` plus a ``log_evaluation`` callback that keeps the
    per-round metric lines out of the Optuna search.
    """
    eval_set = fit_params.pop("eval_set", None)
    fit_params.pop("verbose", None)
    if eval_set:
        X_val, y_val = eval_set[0]
        fit_params.setdefault("eval_X", X_val)
        fit_params.setdefault("eval_y", y_val)
    fit_params.setdefault("callbacks", [log_evaluation(period=-1)])
    return fit_params


class _LightGBMClassifier(LGBMClassifier):
    """``LGBMClassifier`` accepting the XGBoost fit kwargs of the shared loop."""

    def fit(self, X, y=None, **fit_params):
        return super().fit(X, y, **_fit_kwargs(fit_params))


class _LightGBMRegressor(LGBMRegressor):
    """``LGBMRegressor`` accepting the XGBoost fit kwargs of the shared loop."""

    def fit(self, X, y=None, **fit_params):
        return super().fit(X, y, **_fit_kwargs(fit_params))


class OurLightGBMModel(OurTreeModel):
    """LightGBM on the same folds, tuning and evaluation as OurTreeModel.

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

    _build_search_space = staticmethod(build_lightgbm_search_space)

    def _construct_model(self, hyperparameter=None):
        """
        Construct a LightGBM model.

        Builds and returns an unfitted gradient-boosting model, ``LGBMClassifier``
        or ``LGBMRegressor`` depending on the labeling strategy. If
        ``hyperparameter`` is provided, its items are passed as keyword arguments;
        otherwise, library defaults are used.

        Parameters
        ----------
        hyperparameter : dict, optional
            Keyword arguments for the LightGBM model

        Returns
        -------
        An unfitted LightGBM model exposing ``fit``/``predict_proba`` like
        ``XGBClassifier``.
        """
        if self.param.approach not in ('matching', 'BERT'):
            raise NotImplementedError("Currently only matching or BERT approaches are supported.")

        ModelCls = _LightGBMRegressor if self._is_regression else _LightGBMClassifier
        params = dict(hyperparameter) if hyperparameter else {}
        # Same reason as in OurTreeModel: study.best_params only carries the
        # *suggested* values, so the fixed knobs of the space are re-applied here
        # and every construction path ends up with the same thread cap and seed.
        # LightGBM sizes its OpenMP pool from the logical CPU count too, so the
        # WSL2 collapse documented in constant.N_JOBS applies verbatim.
        params.setdefault("n_jobs", getattr(self.param, "n_jobs", N_JOBS))
        params.setdefault("random_state", self.param.random_seed)
        # -1 silences the C++ library's per-round messages, which 0 does not.
        params.setdefault("verbosity", -1)
        return ModelCls(**params)
