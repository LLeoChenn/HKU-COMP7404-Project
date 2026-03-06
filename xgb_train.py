"""
Dual Feature Selection using XGBoost (Outcome & Treatment models).

This module implements a FeatureSelector class that:
  - Trains an XGBRegressor (Model A) to predict the outcome label.
  - Trains an XGBRegressor (Model B) to predict the treatment variable (eta_cut).
  - Compares feature importance across both models via Seaborn visualisations.
  - Selects a unified Top-20 feature set via normalised-score aggregation.
  - Identifies intersection features (potential confounders) ranked highly in
    both models.

Usage (CLI):
    python xgb_train.py --data_path data.csv \
        --feature_cols f1,f2,...,eta_cut \
        --label_col target \
        --treatment_col eta_cut \
        --output_dir ./output
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib / font setup
# ---------------------------------------------------------------------------
matplotlib.use("Agg")  # non-interactive backend suitable for scripts

_CHINESE_FONTS = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "Arial Unicode MS",
]


def _configure_matplotlib_fonts() -> None:
    """Try to configure a CJK-capable font; fall back gracefully."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in _CHINESE_FONTS:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            logger.info("Using font '%s' for matplotlib.", font)
            return
    # Fall back: use a safe Unicode font and replace minus sign
    matplotlib.rcParams["axes.unicode_minus"] = False
    logger.info(
        "No CJK font found; Chinese characters may not render correctly."
    )


_configure_matplotlib_fonts()


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------
class FeatureSelector:
    """Dual XGBoost feature selector (Outcome model + Treatment model).

    Parameters
    ----------
    feature_cols : list[str]
        All feature column names (must include *treatment_col*).
    label_col : str
        Name of the outcome/target column.
    treatment_col : str
        Name of the treatment column (default ``"eta_cut"``).  It is also
        expected to be present inside *feature_cols*.
    output_dir : str or Path
        Directory where text reports and plots are saved.
    xgb_params : dict, optional
        Keyword arguments forwarded to :class:`xgboost.XGBRegressor`.
    test_size : float
        Fraction of data held out for validation (default ``0.2``).
    random_state : int
        Random seed used throughout (default ``42``).
    early_stopping_rounds : int
        XGBoost early-stopping patience (default ``50``).
    top_n_save : int
        Number of top features saved to the importance text files
        (default ``50``).
    top_n_select : int
        Number of features chosen in the comprehensive selection step
        (default ``20``).
    confounder_top_n : int
        Rank cut-off for identifying intersection / confounder features
        (default ``30``).
    """

    def __init__(
        self,
        feature_cols: List[str],
        label_col: str,
        treatment_col: str = "eta_cut",
        output_dir: str = "output",
        xgb_params: Optional[dict] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        top_n_save: int = 50,
        top_n_select: int = 20,
        confounder_top_n: int = 30,
    ) -> None:
        if treatment_col not in feature_cols:
            raise ValueError(
                f"treatment_col '{treatment_col}' must be present in "
                "feature_cols."
            )

        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.treatment_col = treatment_col
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.top_n_save = top_n_save
        self.top_n_select = top_n_select
        self.confounder_top_n = confounder_top_n

        default_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            n_jobs=-1,
        )
        if xgb_params:
            default_params.update(xgb_params)
        self._xgb_params = default_params

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated during training
        self._outcome_importance: Optional[pd.Series] = None
        self._treatment_importance: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing: drop rows with NaN, return cleaned copy.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data containing all required columns.

        Returns
        -------
        pd.DataFrame
            Cleaned data frame.
        """
        required = set(self.feature_cols) | {self.label_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}"
            )

        before = len(data)
        data = data[list(required)].dropna()
        after = len(data)
        if before != after:
            logger.info(
                "Dropped %d rows containing NaN values (%d → %d).",
                before - after,
                before,
                after,
            )
        return data.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def _train_model(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        model_name: str,
    ) -> Tuple[xgb.XGBRegressor, dict]:
        """Train a single XGBRegressor with early stopping.

        Returns
        -------
        model : xgboost.XGBRegressor
            Fitted estimator.
        evals_result : dict
            Training / validation loss history (RMSE per round).
        """
        model = xgb.XGBRegressor(**self._xgb_params)
        logger.info("Training %s model …", model_name)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        evals_result = model.evals_result()
        best_round = model.best_iteration
        train_rmse = evals_result["validation_0"]["rmse"][best_round]
        val_rmse = evals_result["validation_1"]["rmse"][best_round]
        logger.info(
            "%s – best round: %d | train RMSE: %.6f | val RMSE: %.6f",
            model_name,
            best_round,
            train_rmse,
            val_rmse,
        )
        return model, evals_result

    # ------------------------------------------------------------------
    # Model A – Outcome model
    # ------------------------------------------------------------------

    def train_outcome_model(self, data: pd.DataFrame) -> xgb.XGBRegressor:
        """Train Model A: predict *label_col* using all *feature_cols*.

        Side effects
        ------------
        * Saves training/validation MSE loss curve as
          ``<output_dir>/outcome_loss_curve.png``.
        * Writes top-``top_n_save`` feature importances (Gain) to
          ``<output_dir>/outcome_importance.txt``.
        * Stores importances in ``self._outcome_importance``.

        Parameters
        ----------
        data : pd.DataFrame
            Pre-processed data frame.

        Returns
        -------
        xgboost.XGBRegressor
            Fitted outcome model.
        """
        X = data[self.feature_cols]
        y = data[self.label_col]
        X_tr, X_val, y_tr, y_val = self._split(X, y)

        model, evals = self._train_model(
            X_tr, X_val, y_tr, y_val, model_name="Outcome"
        )
        self._plot_loss_curve(
            evals, title="Outcome Model – MSE Loss Curve",
            filename="outcome_loss_curve.png",
        )
        importance = self._extract_importance(model, self.feature_cols)
        self._outcome_importance = importance
        self._save_importance(
            importance, "outcome_importance.txt",
            header="Outcome Model – Top Feature Importances (Gain)",
        )
        return model

    # ------------------------------------------------------------------
    # Model B – Treatment model
    # ------------------------------------------------------------------

    def train_treatment_model(self, data: pd.DataFrame) -> xgb.XGBRegressor:
        """Train Model B: predict *treatment_col* using the remaining features.

        Side effects
        ------------
        * Saves training/validation MSE loss curve as
          ``<output_dir>/treatment_loss_curve.png``.
        * Writes top-``top_n_save`` feature importances (Gain) to
          ``<output_dir>/treatment_importance.txt``.
        * Stores importances in ``self._treatment_importance``.

        Parameters
        ----------
        data : pd.DataFrame
            Pre-processed data frame.

        Returns
        -------
        xgboost.XGBRegressor
            Fitted treatment model.
        """
        treatment_features = [
            c for c in self.feature_cols if c != self.treatment_col
        ]
        X = data[treatment_features]
        y = data[self.treatment_col]
        X_tr, X_val, y_tr, y_val = self._split(X, y)

        model, evals = self._train_model(
            X_tr, X_val, y_tr, y_val, model_name="Treatment"
        )
        self._plot_loss_curve(
            evals, title="Treatment Model – MSE Loss Curve",
            filename="treatment_loss_curve.png",
        )
        importance = self._extract_importance(model, treatment_features)
        self._treatment_importance = importance
        self._save_importance(
            importance, "treatment_importance.txt",
            header="Treatment Model – Top Feature Importances (Gain)",
        )
        return model

    # ------------------------------------------------------------------
    # Analysis & visualisation
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> dict:
        """Execute the full dual feature-selection pipeline.

        Steps:
          1. Preprocess data.
          2. Train outcome model (Model A).
          3. Train treatment model (Model B).
          4. Plot comparative feature importance.
          5. Select comprehensive Top-20 features.
          6. Identify intersection/confounder features.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data.

        Returns
        -------
        dict with keys:
            ``"top_features"`` – list of Top-20 feature names.
            ``"confounders"``  – list of intersection feature names.
        """
        data = self.preprocess(data)
        self.train_outcome_model(data)
        self.train_treatment_model(data)

        self.plot_importance_comparison()
        top_features = self.select_top_features()
        confounders = self.identify_confounders()

        logger.info(
            "Top-%d comprehensive features: %s",
            self.top_n_select,
            top_features,
        )
        logger.info(
            "Intersection features (potential confounders, top-%d): %s",
            self.confounder_top_n,
            confounders,
        )
        return {"top_features": top_features, "confounders": confounders}

    def plot_importance_comparison(self) -> None:
        """Plot a horizontal Seaborn bar chart comparing Gain across models.

        Saves ``<output_dir>/importance_comparison.png``.
        """
        self._check_importances_available()

        # Align on common features
        all_features = sorted(
            set(self._outcome_importance.index)
            | set(self._treatment_importance.index)
        )

        out_vals = self._outcome_importance.reindex(all_features, fill_value=0.0)
        trt_vals = self._treatment_importance.reindex(all_features, fill_value=0.0)

        # Normalise to [0, 1] for fair comparison
        out_norm = out_vals / out_vals.sum() if out_vals.sum() > 0 else out_vals
        trt_norm = trt_vals / trt_vals.sum() if trt_vals.sum() > 0 else trt_vals

        # Keep features with any non-zero importance and sort by combined score
        combined = out_norm + trt_norm
        combined = combined[combined > 0].sort_values(ascending=False)
        show = combined.head(40).index

        plot_df = pd.DataFrame(
            {
                "Feature": list(show) * 2,
                "Normalised Importance (Gain)": list(out_norm[show])
                + list(trt_norm[show]),
                "Model": ["Outcome"] * len(show) + ["Treatment"] * len(show),
            }
        )

        fig, ax = plt.subplots(figsize=(10, max(8, len(show) * 0.35)))
        sns.barplot(
            data=plot_df,
            x="Normalised Importance (Gain)",
            y="Feature",
            hue="Model",
            orient="h",
            palette=["steelblue", "tomato"],
            ax=ax,
        )
        ax.set_title(
            "Feature Importance Comparison (Outcome vs Treatment)",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Normalised Gain", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        ax.legend(title="Model", loc="lower right")
        fig.tight_layout()

        path = self.output_dir / "importance_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved importance comparison chart → %s", path)

    def select_top_features(self) -> List[str]:
        """Select Top-``top_n_select`` features via normalised-rank aggregation.

        Each feature receives a *rank score* in each model (lower rank = higher
        score).  The two normalised rank scores are averaged and the features
        with the highest combined score are returned.

        Returns
        -------
        list[str]
            Top-``top_n_select`` feature names.
        """
        self._check_importances_available()

        def rank_score(series: pd.Series) -> pd.Series:
            """Convert Gain values to normalised inverse-rank scores."""
            ranked = series.rank(ascending=False, method="first")
            n = len(ranked)
            return (n + 1 - ranked) / n  # higher gain → score closer to 1

        out_score = rank_score(self._outcome_importance)
        trt_score = rank_score(self._treatment_importance)

        # Align on the union of features
        all_features = sorted(
            set(out_score.index) | set(trt_score.index)
        )
        out_score = out_score.reindex(all_features, fill_value=0.0)
        trt_score = trt_score.reindex(all_features, fill_value=0.0)

        combined = (out_score + trt_score) / 2
        top = combined.nlargest(self.top_n_select).index.tolist()

        # Persist to file
        result_path = self.output_dir / "top_features.txt"
        with open(result_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"Top-{self.top_n_select} features "
                "(normalised rank aggregation)\n"
            )
            fh.write("=" * 60 + "\n")
            for rank, feat in enumerate(top, 1):
                fh.write(
                    f"{rank:>3}. {feat:<40}  score={combined[feat]:.6f}\n"
                )
        logger.info("Saved Top-%d features → %s", self.top_n_select, result_path)
        return top

    def identify_confounders(self) -> List[str]:
        """Identify features in the top-``confounder_top_n`` of *both* models.

        Returns
        -------
        list[str]
            Features that appear in the top-``confounder_top_n`` of both the
            outcome and treatment importance rankings.
        """
        self._check_importances_available()

        n = self.confounder_top_n
        top_out = set(
            self._outcome_importance.nlargest(n).index
        )
        top_trt = set(
            self._treatment_importance.nlargest(n).index
        )
        confounders = sorted(top_out & top_trt)

        result_path = self.output_dir / "confounders.txt"
        with open(result_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"Intersection features (top-{n} in both models) "
                "– potential confounders\n"
            )
            fh.write("=" * 60 + "\n")
            for feat in confounders:
                fh.write(f"  {feat}\n")
        logger.info(
            "Identified %d confounder(s) → %s", len(confounders), result_path
        )
        return confounders

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_importance(
        model: xgb.XGBRegressor, feature_names: List[str]
    ) -> pd.Series:
        """Return a *sorted* Series of Gain-based feature importances."""
        scores = model.get_booster().get_score(importance_type="gain")
        # XGBoost may use 'f0', 'f1' … keys when feature names are not set;
        # map back to actual names.
        if scores:
            first_key = next(iter(scores))
            if first_key.startswith("f") and first_key[1:].isdigit():
                scores = {
                    feature_names[int(k[1:])]: v for k, v in scores.items()
                }
        importance = pd.Series(scores, name="gain").reindex(
            feature_names, fill_value=0.0
        )
        return importance.sort_values(ascending=False)

    def _save_importance(
        self, importance: pd.Series, filename: str, header: str
    ) -> None:
        """Write top-``top_n_save`` importances to a text file."""
        path = self.output_dir / filename
        top = importance.head(self.top_n_save)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            fh.write("=" * 60 + "\n")
            for rank, (feat, gain) in enumerate(top.items(), 1):
                fh.write(f"{rank:>3}. {feat:<40}  gain={gain:.6f}\n")
        logger.info("Saved importance report → %s", path)

    @staticmethod
    def _rmse_to_mse(rmse_values: List[float]) -> List[float]:
        """Convert RMSE sequence to MSE for loss-curve plotting."""
        return [v ** 2 for v in rmse_values]

    def _plot_loss_curve(
        self, evals: dict, title: str, filename: str
    ) -> None:
        """Plot train/val MSE loss curves and save to output directory."""
        train_rmse = evals["validation_0"]["rmse"]
        val_rmse = evals["validation_1"]["rmse"]
        train_mse = self._rmse_to_mse(train_rmse)
        val_mse = self._rmse_to_mse(val_rmse)

        rounds = list(range(1, len(train_mse) + 1))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, train_mse, label="Train MSE", color="steelblue")
        ax.plot(
            rounds, val_mse,
            label="Validation MSE",
            color="tomato",
            linestyle="--",
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Boosting Round", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.legend()
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved loss curve → %s", path)

    def _check_importances_available(self) -> None:
        """Raise RuntimeError if models have not been trained yet."""
        if self._outcome_importance is None:
            raise RuntimeError(
                "Outcome model has not been trained yet. "
                "Call train_outcome_model() first."
            )
        if self._treatment_importance is None:
            raise RuntimeError(
                "Treatment model has not been trained yet. "
                "Call train_treatment_model() first."
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dual XGBoost feature selection (Outcome + Treatment models)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input CSV (or Parquet) file.",
    )
    parser.add_argument(
        "--feature_cols",
        type=str,
        required=True,
        help=(
            "Comma-separated list of feature column names. "
            "Must include the treatment column."
        ),
    )
    parser.add_argument(
        "--label_col",
        type=str,
        required=True,
        help="Name of the outcome / target column.",
    )
    parser.add_argument(
        "--treatment_col",
        type=str,
        default="eta_cut",
        help="Name of the treatment column (must be in --feature_cols).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where results and plots are saved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation-set fraction.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=50,
        help="Early-stopping patience (boosting rounds).",
    )
    parser.add_argument(
        "--top_n_save",
        type=int,
        default=50,
        help="Number of top features saved to the importance text files.",
    )
    parser.add_argument(
        "--top_n_select",
        type=int,
        default=20,
        help="Number of top features selected in the comprehensive step.",
    )
    parser.add_argument(
        "--confounder_top_n",
        type=int,
        default=30,
        help=(
            "Rank cut-off for identifying intersection "
            "(confounder) features."
        ),
    )
    return parser.parse_args()


def _load_data(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    """CLI entry point for xgb_train.py."""
    args = _parse_args()

    feature_cols = [c.strip() for c in args.feature_cols.split(",")]

    logger.info("Loading data from '%s' …", args.data_path)
    data = _load_data(args.data_path)
    logger.info("Data shape: %s", data.shape)

    selector = FeatureSelector(
        feature_cols=feature_cols,
        label_col=args.label_col,
        treatment_col=args.treatment_col,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        top_n_save=args.top_n_save,
        top_n_select=args.top_n_select,
        confounder_top_n=args.confounder_top_n,
    )

    results = selector.run(data)

    print("\n" + "=" * 60)
    print(f"Top-{args.top_n_select} comprehensive features:")
    for i, f in enumerate(results["top_features"], 1):
        print(f"  {i:>2}. {f}")

    print(
        f"\nIntersection features "
        f"(potential confounders, top-{args.confounder_top_n}):"
    )
    for f in results["confounders"]:
        print(f"  • {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
