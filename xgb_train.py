"""
Dual Feature Selection using XGBoost for Outcome and Treatment models.

This script implements a FeatureSelector class that performs:
- Model A: Outcome regression (predicting label_col using feature_cols)
- Model B: Treatment regression (predicting eta_cut using remaining features)
- Comprehensive feature importance analysis and visualization
"""

import os
import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Font configuration: fall back gracefully when Chinese fonts are unavailable
# ---------------------------------------------------------------------------
_CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
_FALLBACK_FONT = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False


def _configure_font() -> None:
    """Set matplotlib font to the first available CJK font."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in _CHINESE_FONTS:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = _FALLBACK_FONT


_configure_font()

# ---------------------------------------------------------------------------
# Default XGBoost hyper-parameters
# ---------------------------------------------------------------------------
_DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

_EARLY_STOPPING_ROUNDS = 50
_TOP_N_IMPORTANCE = 50
_TOP_N_COMPREHENSIVE = 20
_TOP_N_INTERSECTION = 30
_ETA_THRESHOLD = 0.5  # |eta| threshold for eta_cut treatment in synthetic demo


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_data(data_source) -> pd.DataFrame:
    """Load a DataFrame from a file path or return as-is if already a DataFrame."""
    if isinstance(data_source, pd.DataFrame):
        return data_source.copy()
    if isinstance(data_source, (str, os.PathLike)):
        path = str(data_source)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {path}")
    raise TypeError(f"Expected DataFrame or file path, got {type(data_source)}")


def _preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Basic preprocessing: keep relevant columns, drop rows with NaN, encode
    any remaining categorical features with LabelEncoder.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with only `feature_cols`.
    y : pd.Series
        Target series for `label_col`.
    """
    cols = feature_cols + [label_col]
    df = df[cols].dropna().reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Encode object / category columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


def _plot_loss_curve(
    eval_results: dict,
    title: str,
    save_path: str,
) -> None:
    """Plot training and validation MSE loss curves and save to *save_path*."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_results["train"]["rmse"], label="Train RMSE", linewidth=1.5)
    if "val" in eval_results:
        ax.plot(eval_results["val"]["rmse"], label="Validation RMSE", linewidth=1.5)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _save_importance(
    importance_dict: dict[str, float],
    save_path: str,
    top_n: int = _TOP_N_IMPORTANCE,
) -> pd.DataFrame:
    """
    Sort features by importance, write the top *top_n* to *save_path*, and
    return the resulting DataFrame.
    """
    df_imp = (
        pd.Series(importance_dict, name="importance")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    df_imp.to_csv(save_path, sep="\t", index=False)
    return df_imp


# ---------------------------------------------------------------------------
# FeatureSelector class
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Dual feature selection via XGBoost for Outcome and Treatment targets.

    Parameters
    ----------
    feature_cols : list[str]
        All candidate feature column names (must include `treatment_col`).
    label_col : str
        Outcome / target column name for Model A.
    treatment_col : str
        Treatment variable column name (default: ``"eta_cut"``).
        Also used as the prediction target for Model B.
    xgb_params : dict, optional
        XGBoost hyper-parameters to override the defaults.
    val_size : float
        Fraction of data reserved for validation (default: 0.2).
    output_dir : str
        Directory where plots and importance files are written.
    top_n_importance : int
        Number of top features to save per model (default: 50).
    top_n_comprehensive : int
        Number of top features from comprehensive aggregation (default: 20).
    top_n_intersection : int
        Rank threshold used when identifying intersection features (default: 30).
    """

    def __init__(
        self,
        feature_cols: list[str],
        label_col: str,
        treatment_col: str = "eta_cut",
        xgb_params: Optional[dict] = None,
        val_size: float = 0.2,
        output_dir: str = ".",
        top_n_importance: int = _TOP_N_IMPORTANCE,
        top_n_comprehensive: int = _TOP_N_COMPREHENSIVE,
        top_n_intersection: int = _TOP_N_INTERSECTION,
    ) -> None:
        if treatment_col not in feature_cols:
            raise ValueError(
                f"treatment_col '{treatment_col}' must be present in feature_cols."
            )
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.treatment_col = treatment_col
        self.xgb_params = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.val_size = val_size
        self.output_dir = output_dir
        self.top_n_importance = top_n_importance
        self.top_n_comprehensive = top_n_comprehensive
        self.top_n_intersection = top_n_intersection

        os.makedirs(output_dir, exist_ok=True)

        # Populated after fit()
        self.model_outcome_: Optional[xgb.XGBRegressor] = None
        self.model_treatment_: Optional[xgb.XGBRegressor] = None
        self.outcome_importance_: Optional[pd.DataFrame] = None
        self.treatment_importance_: Optional[pd.DataFrame] = None
        self.comprehensive_features_: Optional[list[str]] = None
        self.intersection_features_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            **self.xgb_params,
            eval_metric="rmse",
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        )

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
    ) -> tuple[xgb.XGBRegressor, dict]:
        """Train one XGBRegressor with early stopping; return model + eval dict."""
        model = self._build_model()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        # Retrieve evals_result via the booster
        raw = model.evals_result()
        eval_results: dict = {
            "train": raw["validation_0"],
            "val": raw["validation_1"],
        }
        print(
            f"[{model_name}] best iteration: {model.best_iteration}, "
            f"best RMSE: {model.best_score:.4f}"
        )
        return model, eval_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data) -> "FeatureSelector":
        """
        Run the full dual feature selection pipeline on *data*.

        Parameters
        ----------
        data : pd.DataFrame or str or os.PathLike
            Raw dataset or path to a CSV / Parquet file.

        Returns
        -------
        self
        """
        df = _load_data(data)

        # ── Model A: Outcome ──────────────────────────────────────────
        print("=" * 60)
        print("Model A – Outcome (predicting label_col)")
        print("=" * 60)

        X_a, y_a = _preprocess(df, self.feature_cols, self.label_col)
        X_a_tr, X_a_val, y_a_tr, y_a_val = train_test_split(
            X_a, y_a, test_size=self.val_size, random_state=42
        )
        self.model_outcome_, eval_a = self._train(
            X_a_tr, y_a_tr, X_a_val, y_a_val, "Outcome"
        )
        _plot_loss_curve(
            eval_a,
            title="Outcome Model – RMSE Curve",
            save_path=os.path.join(self.output_dir, "outcome_loss_curve.png"),
        )
        imp_a = self.model_outcome_.get_booster().get_score(importance_type="gain")
        self.outcome_importance_ = _save_importance(
            imp_a,
            save_path=os.path.join(self.output_dir, "outcome_importance.txt"),
            top_n=self.top_n_importance,
        )
        print(f"  → Saved top-{self.top_n_importance} outcome features.")

        # ── Model B: Treatment ────────────────────────────────────────
        print("=" * 60)
        print(f"Model B – Treatment (predicting '{self.treatment_col}')")
        print("=" * 60)

        treat_features = [c for c in self.feature_cols if c != self.treatment_col]
        X_b, y_b = _preprocess(df, treat_features, self.treatment_col)
        X_b_tr, X_b_val, y_b_tr, y_b_val = train_test_split(
            X_b, y_b, test_size=self.val_size, random_state=42
        )
        self.model_treatment_, eval_b = self._train(
            X_b_tr, y_b_tr, X_b_val, y_b_val, "Treatment"
        )
        _plot_loss_curve(
            eval_b,
            title="Treatment Model – RMSE Curve",
            save_path=os.path.join(self.output_dir, "treatment_loss_curve.png"),
        )
        imp_b = self.model_treatment_.get_booster().get_score(importance_type="gain")
        self.treatment_importance_ = _save_importance(
            imp_b,
            save_path=os.path.join(self.output_dir, "treatment_importance.txt"),
            top_n=self.top_n_importance,
        )
        print(f"  → Saved top-{self.top_n_importance} treatment features.")

        # ── Comprehensive analysis ────────────────────────────────────
        self._comprehensive_analysis()

        return self

    def _comprehensive_analysis(self) -> None:
        """
        Aggregate feature importance from both models, visualise, and derive
        the comprehensive Top-N and intersection feature sets.
        """
        df_a = self.outcome_importance_.set_index("feature")["importance"]
        df_b = self.treatment_importance_.set_index("feature")["importance"]

        all_features = df_a.index.union(df_b.index)
        df_a = df_a.reindex(all_features, fill_value=0.0)
        df_b = df_b.reindex(all_features, fill_value=0.0)

        # Min-max normalise each model's scores independently
        _eps = 1e-8

        def _minmax(s: pd.Series) -> pd.Series:
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > _eps else s * 0.0

        norm_a = _minmax(df_a)
        norm_b = _minmax(df_b)

        # Weighted average (equal weight 0.5 each)
        combined = 0.5 * norm_a + 0.5 * norm_b
        combined_sorted = combined.sort_values(ascending=False)

        self.comprehensive_features_ = combined_sorted.head(
            self.top_n_comprehensive
        ).index.tolist()

        # Save comprehensive ranking
        comp_path = os.path.join(self.output_dir, "comprehensive_importance.txt")
        combined_sorted_df = combined_sorted.rename("combined_score").reset_index()
        combined_sorted_df.columns = ["feature", "combined_score"]
        combined_sorted_df.to_csv(comp_path, sep="\t", index=False)

        # Intersection: features in top-N of BOTH models
        top_a_set = set(df_a.nlargest(self.top_n_intersection).index)
        top_b_set = set(df_b.nlargest(self.top_n_intersection).index)
        self.intersection_features_ = sorted(top_a_set & top_b_set)

        int_path = os.path.join(self.output_dir, "intersection_features.txt")
        with open(int_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Intersection features (top-{self.top_n_intersection} in both models)\n"
            )
            for feat in self.intersection_features_:
                fh.write(feat + "\n")

        print(f"  → Comprehensive Top-{self.top_n_comprehensive}: {self.comprehensive_features_}")
        print(f"  → Intersection features ({len(self.intersection_features_)}): "
              f"{self.intersection_features_}")

        # ── Visualisation ─────────────────────────────────────────────
        self._plot_comparison(norm_a, norm_b)

    def _plot_comparison(
        self,
        norm_a: pd.Series,
        norm_b: pd.Series,
    ) -> None:
        """
        Horizontal bar chart comparing normalised feature importance of both
        models for the features that appear in either model's top-N list.
        """
        # Restrict to features present in either importance list
        feat_a = set(self.outcome_importance_["feature"])
        feat_b = set(self.treatment_importance_["feature"])
        display_features = sorted(feat_a | feat_b)

        plot_df = pd.DataFrame(
            {
                "Outcome Model": norm_a.reindex(display_features, fill_value=0.0),
                "Treatment Model": norm_b.reindex(display_features, fill_value=0.0),
            }
        )
        plot_df = plot_df.sort_values("Outcome Model", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(display_features) * 0.3)))
        plot_df.plot.barh(ax=ax, alpha=0.8)
        ax.set_xlabel("Normalised Importance (Gain)")
        ax.set_title(
            "Feature Importance Comparison\n(Outcome vs. Treatment Model)"
        )
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "importance_comparison.png"), dpi=150
        )
        plt.close(fig)

        # Also draw a seaborn version with the top-20 combined features
        combined = (0.5 * norm_a + 0.5 * norm_b).sort_values(ascending=False)
        top_features = combined.head(self.top_n_comprehensive).index.tolist()

        sns_df = pd.DataFrame(
            {
                "feature": top_features * 2,
                "importance": (
                    norm_a.reindex(top_features, fill_value=0.0).tolist()
                    + norm_b.reindex(top_features, fill_value=0.0).tolist()
                ),
                "model": (
                    ["Outcome Model"] * len(top_features)
                    + ["Treatment Model"] * len(top_features)
                ),
            }
        )

        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(top_features) * 0.4)))
        sns.barplot(
            data=sns_df,
            y="feature",
            x="importance",
            hue="model",
            orient="h",
            ax=ax2,
            palette="Set2",
        )
        ax2.set_xlabel("Normalised Importance (Gain)")
        ax2.set_title(
            f"Top-{self.top_n_comprehensive} Comprehensive Features\n"
            "(Outcome vs. Treatment)"
        )
        ax2.legend(title="Model")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "top_comprehensive_comparison.png"), dpi=150
        )
        plt.close(fig2)
        print("  → Comparison plots saved.")

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict summarising the selection results."""
        return {
            "outcome_top_features": (
                self.outcome_importance_["feature"].tolist()
                if self.outcome_importance_ is not None
                else []
            ),
            "treatment_top_features": (
                self.treatment_importance_["feature"].tolist()
                if self.treatment_importance_ is not None
                else []
            ),
            "comprehensive_top_features": self.comprehensive_features_ or [],
            "intersection_features": self.intersection_features_ or [],
        }


# ---------------------------------------------------------------------------
# Demo: synthetic dataset matching the xgb_train.py usage pattern
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 40,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Create a synthetic dataset that mimics the structure of a physics-style
    tabular dataset with an `eta_cut` treatment column.

    Returns
    -------
    df : pd.DataFrame
    feature_cols : list[str]
    label_col : str
    """
    rng = np.random.default_rng(random_state)

    # Physics-inspired feature names
    base_names = [
        "pt", "eta", "phi", "mass", "energy",
        "dR", "dPhi", "dEta", "met", "ht",
        "n_jets", "n_bjets", "csv_score", "jet_pt1", "jet_pt2",
        "lep_pt", "lep_eta", "lep_phi", "lep_iso", "mT",
        "mll", "dphi_met_lep", "dphi_met_jet", "mjj", "ptj1",
    ]
    extra = [f"feat_{i}" for i in range(n_features - len(base_names))]
    feature_names = (base_names + extra)[:n_features]

    X = rng.standard_normal((n_samples, n_features))
    df = pd.DataFrame(X, columns=feature_names)

    # eta_cut: binary treatment based on |eta| threshold (realistic)
    df["eta_cut"] = (np.abs(df["eta"]) > _ETA_THRESHOLD).astype(float)

    # Outcome: linear combo of some features + noise
    coefs = rng.standard_normal(n_features)
    y = X @ coefs + 0.5 * rng.standard_normal(n_samples)
    df["outcome"] = y

    feature_cols = feature_names + ["eta_cut"]
    label_col = "outcome"
    return df, feature_cols, label_col


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual feature selection with XGBoost (Outcome + Treatment)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV / Parquet data file. Omit to run with synthetic demo data.",
    )
    parser.add_argument(
        "--featureCols",
        type=str,
        default=None,
        help="Comma-separated feature column names.",
    )
    parser.add_argument(
        "--labelCol",
        type=str,
        default="outcome",
        help="Target / outcome column name.",
    )
    parser.add_argument(
        "--treatmentCol",
        type=str,
        default="eta_cut",
        help="Treatment variable column name (must be in featureCols).",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        default="xgb_output",
        help="Directory to save plots and importance files.",
    )
    parser.add_argument(
        "--valSize",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2).",
    )
    args = parser.parse_args()

    if args.data is not None:
        df = _load_data(args.data)
        if args.featureCols is None:
            raise ValueError("--featureCols is required when --data is provided.")
        feature_cols = args.featureCols.split(",")
        label_col = args.labelCol
    else:
        print("No --data provided; using synthetic demo dataset.")
        df, feature_cols, label_col = _generate_synthetic_data()

    selector = FeatureSelector(
        feature_cols=feature_cols,
        label_col=label_col,
        treatment_col=args.treatmentCol,
        val_size=args.valSize,
        output_dir=args.outputDir,
    )
    selector.fit(df)

    results = selector.summary()
    print("\n=== Summary ===")
    print(f"Outcome top features   : {results['outcome_top_features'][:10]}")
    print(f"Treatment top features : {results['treatment_top_features'][:10]}")
    print(f"Comprehensive top-20   : {results['comprehensive_top_features']}")
    print(f"Intersection features  : {results['intersection_features']}")
