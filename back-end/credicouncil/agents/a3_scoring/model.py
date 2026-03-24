"""
CREDICOUNCIL A3 — LightGBM Model Wrapper.

Handles model training (with ADASYN + Optuna tuning), prediction,
and model serialization. This is the deterministic ML backbone of CREDICOUNCIL.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from credicouncil.config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


# ─── Default Hyperparameters ─────────────────────────────────────────────────

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,
    "random_state": 42,
    "verbose": -1,
}


class CrediCouncilModel:
    """LightGBM model wrapper for credit scoring.

    Provides training, prediction, hyperparameter tuning via Optuna,
    and model persistence.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []
        self.model_version: str = "lgbm_v1.0_homecredit"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 50,
    ) -> dict[str, Any]:
        """Train LightGBM model.

        Args:
            X_train: Training features (may be ADASYN-resampled).
            y_train: Training targets.
            X_val: Validation features (real distribution, no ADASYN).
            y_val: Validation targets.
            early_stopping_rounds: Stop if no improvement for N rounds.

        Returns:
            Dict with training metrics.
        """
        logger.info(f"Training LightGBM — {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

        self.feature_names = list(X_train.columns)
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = [lgb.log_evaluation(100)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        # Collect training results
        results = {
            "n_estimators_used": self.model.best_iteration_ if hasattr(self.model, "best_iteration_") else self.params.get("n_estimators"),
            "feature_importances": dict(zip(self.feature_names, self.model.feature_importances_)),
        }

        if eval_set:
            val_pred = self.model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score
            results["val_auc"] = roc_auc_score(y_val, val_pred)
            logger.info(f"Validation AUC: {results['val_auc']:.4f}")

        return results

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability of default.

        Handles two model formats:
        1. Single LGBMClassifier (old CREDICOUNCIL format)
        2. List of BaggingClassifiers (new train_pipeline format)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # New format: list of BaggingClassifiers (one per fold)
        if isinstance(self.model, list):
            preds = np.zeros(len(X))
            for clf in self.model:
                preds += clf.predict_proba(X)[:, 1] / len(self.model)
            return preds

        # Old format: single LGBMClassifier
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions.

        Args:
            X: Feature DataFrame.
            threshold: Classification threshold.

        Returns:
            Array of binary predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Path | str | None = None) -> Path:
        """Save model to disk.

        Args:
            path: File path. Defaults to models/lgbm_v1.pkl.

        Returns:
            Path where model was saved.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        path = Path(path) if path else MODELS_DIR / "lgbm_v1.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
            "model_version": self.model_version,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: Path | str | None = None) -> None:
        """Load model from disk.

        Hỗ trợ 2 format:
        1. CREDICOUNCIL format: dict {"model": LGBMClassifier, "feature_names": [...], ...}
        2. Raw format: LGBMClassifier object được save trực tiếp (pkl từ external source)

        Args:
            path: File path. Defaults to models/lgbm_v1.pkl.
        """
        path = Path(path) if path else MODELS_DIR / "lgbm_v1.pkl"

        class _CustomUnpickler(pickle.Unpickler):
            """Remap classes saved under __main__ to their importable module path.

            When train_pipeline.py ran as __main__, BaggingClassifier was pickled
            as __main__.BaggingClassifier. This remaps it back so any caller can load.
            """
            _REMAP = {
                ("__main__", "BaggingClassifier"): ("training.train_pipeline", "BaggingClassifier"),
            }

            def find_class(self, module: str, name: str):
                module, name = self._REMAP.get((module, name), (module, name))
                return super().find_class(module, name)

        with open(path, "rb") as f:
            model_data = _CustomUnpickler(f).load()


        # CREDICOUNCIL original format: dict with 'model' key
        if isinstance(model_data, dict) and "model" in model_data:
            self.model = model_data["model"]
            self.feature_names = model_data.get("feature_names", [])
            self.params = model_data.get("params", self.params)
            self.model_version = model_data.get("model_version", "unknown")

        # New train_pipeline format: dict with 'classifiers' list
        elif isinstance(model_data, dict) and "classifiers" in model_data:
            self.model = model_data["classifiers"]  # list of BaggingClassifiers
            self.feature_names = model_data.get("feature_names", [])
            self.meanenc_feats = model_data.get("meanenc_feats", [])
            self.cat_feats = model_data.get("cat_feats", [])
            self.model_version = "lgb1_ref_v1"
            logger.info(f"Loaded {len(self.model)} BaggingClassifiers (folds), {len(self.feature_names)} features.")

        else:
            # Raw LGBMClassifier (pkl từ external source)
            self.model = model_data
            self.feature_names = list(getattr(model_data, "feature_name_", []))
            self.model_version = "external_model"
            logger.warning(
                f"Loaded raw LGBMClassifier từ {path} (external format). "
                f"Feature names: {len(self.feature_names)} features."
            )

        logger.info(f"Model loaded from {path} (version: {self.model_version})")


    def tune_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Hyperparameter tuning with Optuna.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            n_trials: Number of Optuna trials.

        Returns:
            Dict with best parameters and best AUC.
        """
        import optuna
        from sklearn.metrics import roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "n_estimators": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "is_unbalance": True,
                "random_state": 42,
                "verbose": -1,
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            val_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_pred)

        logger.info(f"Starting Optuna tuning with {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = {**DEFAULT_PARAMS, **study.best_params}
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Update model with best params
        self.params = best_params

        return {
            "best_auc": study.best_value,
            "best_params": study.best_params,
            "n_trials": n_trials,
        }
