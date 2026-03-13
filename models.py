

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PROC_PATH = ROOT / "data" / "processed" / "features_processed.csv"
MDL_DIR   = ROOT / "data" / "models"


# ═════════════════════════════════════════════════════════════════════════════
# Model classes
# ═════════════════════════════════════════════════════════════════════════════

class LogLinkGLM:
    """
    Log-link GLM (Tweedie-equivalent).
    Response = log(E[Loss]) — coefficients are log-rate relativities.
    Uses Ridge regularisation to avoid overfitting on correlated features.
    """

    def __init__(self, feature_names: list, alpha: float = 0.5):
        self.feature_names = list(feature_names)
        self.alpha         = alpha
        self.scaler        = StandardScaler()
        self.model         = Ridge(alpha=alpha)
        self.coef_         = None
        self.intercept_    = None
        self._r2_train     = None
        self._r2_cv        = None

    def fit(self, X_df: pd.DataFrame, y_log: np.ndarray) -> "LogLinkGLM":
        X               = self.scaler.fit_transform(X_df[self.feature_names])
        self.model.fit(X, y_log)
        self.coef_      = self.model.coef_
        self.intercept_ = self.model.intercept_
        y_hat           = self.model.predict(X)
        self._r2_train  = float(r2_score(y_log, y_hat))
        scores          = cross_val_score(self.model, X, y_log, cv=5, scoring="r2")
        self._r2_cv     = float(scores.mean())
        return self

    def predict_eta(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(X_df[self.feature_names])
        return self.model.predict(X)

    def predict_loss(self, X_df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_eta(X_df))

    def feature_importance(self) -> dict:
        return {f: round(float(c), 6)
                for f, c in zip(self.feature_names, self.coef_)}

    def relativities_table(self) -> pd.DataFrame:
        return (pd.DataFrame({
            "feature":    self.feature_names,
            "coef_scaled": self.coef_,
        })
        .assign(direction=lambda d: np.where(d.coef_scaled > 0, "Risk", "Mitigant"))
        .sort_values("coef_scaled", ascending=False)
        .reset_index(drop=True))


class SimpleGAM:
    """
    GAM residual model using degree-2 polynomial feature expansion + Ridge.

    Fits:  δ(x) = log(Y) − η_GLM(x)
    Captures:
      (1) Non-linear smooth terms — roof risk accelerates after 15yr,
          wildfire score is quadratic, etc.
      (2) Pairwise interaction surfaces the log-linear GLM cannot model.
    """

    def __init__(self, feature_names: list, degree: int = 2, alpha: float = 8.0):
        self.feature_names = list(feature_names)
        self.degree        = degree
        self.alpha         = alpha
        self.scaler        = StandardScaler()
        self.poly          = PolynomialFeatures(degree=degree, include_bias=False,
                                                 interaction_only=False)
        self.ridge         = Ridge(alpha=alpha)
        self._r2_train     = None
        self._r2_cv        = None

    def fit(self, X_df: pd.DataFrame, residuals: np.ndarray) -> "SimpleGAM":
        X               = self.scaler.fit_transform(X_df[self.feature_names])
        Xp              = self.poly.fit_transform(X)
        self.ridge.fit(Xp, residuals)
        r_hat           = self.ridge.predict(Xp)
        self._r2_train  = float(r2_score(residuals, r_hat))
        scores          = cross_val_score(self.ridge, Xp, residuals, cv=5, scoring="r2")
        self._r2_cv     = float(scores.mean())
        return self

    def predict_delta(self, X_df: pd.DataFrame) -> np.ndarray:
        X  = self.scaler.transform(X_df[self.feature_names])
        Xp = self.poly.transform(X)
        return self.ridge.predict(Xp)


# ═════════════════════════════════════════════════════════════════════════════
# Feature sets  
# ═════════════════════════════════════════════════════════════════════════════

# Tier 1 — Structural Foundation
T1_FEATURES = [
    "roof_age",  "mat_r",              # Roof vulnerability  (r=0.42, 0.38)
    "home_age",  "cnst_r",             # Home age / construction (r=0.12, 0.35)
    "prior_water_claim", "rec_f",      # Water loss recency  (r=0.29)
    "coverage_a", "rcv_ratio", "rcv_os",  # RCV validation   (r=0.18)
    "loc_off",                         # Location zone offset
]

# Tier 2 — Lifestyle & Secondary 
T2_EXTRA = [
    "prior_claims_5yr",   # r=0.26  strongest T2 predictor
    "ins_lapses",         # r=0.18
    "crime_idx",          # r=0.19
    "fire_hydrant_dist",  # r=0.15
    "pet_ownership",      # r=0.16
    "pool_r",             # r=0.14
    "trampoline",         # r=0.12
    "wood_stove",         # r=0.10
    "biz_r",              # r=0.09
    "recent_reno",        # r=0.08 (mitigant)
    "monitored_alarm",    # r=0.07 (mitigant)
    "sprnk_v",            # r=0.05 (mitigant)
    "gated_community",    # r=0.03 (mitigant)
    "building_code_comply",
    "iso_class",
]
T2_FEATURES = T1_FEATURES + T2_EXTRA

# Tier 3 — GAM (geographical + interaction features)
T3_FEATURES = [
    "wildfire_score", "wildfire_zone",
    "canopy_pct", "flood_zone", "flood_depth_in",
    "slope_deg",  "hail_zone",  "burn_history", "fnd_r",
    "ix_roof_wildfire", "ix_water_canopy", "ix_rcv_crime",
    "ix_flood_foundation", "ix_slope_burn", "ix_roof_hail",
]


# ═════════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═════════════════════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("GLM-GAM Model Training Pipeline")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────────
    if not PROC_PATH.exists():
        raise FileNotFoundError(
            f"Processed features not found at {PROC_PATH}\n"
            "Run  python generate_data.py  first."
        )
    df = pd.read_csv(PROC_PATH)
    print(f"\nLoaded {len(df):,} policies from {PROC_PATH.name}")

    y_log = df["eta_final"].values

    # Drop any missing columns gracefully
    t1 = [c for c in T1_FEATURES  if c in df.columns]
    t2 = [c for c in T2_FEATURES  if c in df.columns]
    t3 = [c for c in T3_FEATURES  if c in df.columns]
    missing = set(T1_FEATURES + T2_FEATURES + T3_FEATURES) - set(df.columns)
    if missing:
        print(f"  ⚠  Skipped missing columns: {sorted(missing)}")

    # ── Tier 1 GLM ────────────────────────────────────────────────────────────
    print("\n── Tier 1 GLM  (Structural Foundation) ──────────────────────")
    glm_t1 = LogLinkGLM(t1, alpha=0.5)
    glm_t1.fit(df, y_log)
    yhat_t1 = glm_t1.predict_eta(df)
    r2_t1   = r2_score(y_log, yhat_t1)
    print(f"  Features : {len(t1)}")
    print(f"  R² train : {glm_t1._r2_train:.4f}")
    print(f"  R² 5-CV  : {glm_t1._r2_cv:.4f}")

    # ── Tier 2 GLM ────────────────────────────────────────────────────────────
    print("\n── Tier 2 GLM  (T1 + Lifestyle variables) ───────────────────")
    glm_full = LogLinkGLM(t2, alpha=0.5)
    glm_full.fit(df, y_log)
    yhat_glm = glm_full.predict_eta(df)
    r2_glm   = r2_score(y_log, yhat_glm)
    print(f"  Features : {len(t2)}")
    print(f"  R² train : {glm_full._r2_train:.4f}")
    print(f"  R² 5-CV  : {glm_full._r2_cv:.4f}")

    # ── Tier 3 GAM ────────────────────────────────────────────────────────────
    print("\n── Tier 3 GAM  (Residual model, degree=2, α=8.0) ────────────")
    residuals = y_log - yhat_glm
    gam       = SimpleGAM(t3, degree=2, alpha=8.0)
    gam.fit(df, residuals)
    yhat_final = yhat_glm + gam.predict_delta(df)
    r2_final   = r2_score(y_log, yhat_final)
    print(f"  Features : {len(t3)}  (+ {gam.poly.n_output_features_} poly terms)")
    print(f"  R² on residuals (train) : {gam._r2_train:.4f}")
    print(f"  R² on residuals (5-CV)  : {gam._r2_cv:.4f}")
    print(f"  Combined R² (train)     : {r2_final:.4f}")

    # ── Variance decomposition ────────────────────────────────────────────────
    t1_pct  = r2_t1   / r2_final * 100 if r2_final > 0 else 0
    t2_pct  = (r2_glm  - r2_t1)  / r2_final * 100 if r2_final > 0 else 0
    t3_pct  = (r2_final - r2_glm) / r2_final * 100 if r2_final > 0 else 0
    res_pct = (1 - r2_final) * 100

    print("\n── Variance decomposition ────────────────────────────────────")
    print(f"  Tier 1 Foundation  : {t1_pct:.1f}% of explained variance")
    print(f"  Tier 2 Lifestyle   : +{t2_pct:.1f}% incremental")
    print(f"  Tier 3 GAM         : +{t3_pct:.1f}% incremental")
    print(f"  Residual           :  {res_pct:.1f}% unexplained")

    # ── Top features ──────────────────────────────────────────────────────────
    print("\n── Top feature importances (Tier 1 GLM) ─────────────────────")
    for feat, coef in sorted(glm_t1.feature_importance().items(),
                              key=lambda x: abs(x[1]), reverse=True)[:8]:
        print(f"  {feat:<28}  {coef:+.4f}")

    print("\n── Top feature importances (Tier 2 GLM) ─────────────────────")
    for feat, coef in sorted(glm_full.feature_importance().items(),
                              key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"  {feat:<28}  {coef:+.4f}")

    # ── Save models ───────────────────────────────────────────────────────────
    with open(MDL_DIR / "glm_t1.pkl",       "wb") as f: pickle.dump(glm_t1,   f)
    with open(MDL_DIR / "glm_full.pkl",     "wb") as f: pickle.dump(glm_full, f)
    with open(MDL_DIR / "gam_residual.pkl", "wb") as f: pickle.dump(gam,      f)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = {
        "glm_t1":   {"r2_train": round(glm_t1._r2_train,  4), "r2_cv": round(glm_t1._r2_cv,  4)},
        "glm_full": {"r2_train": round(glm_full._r2_train, 4), "r2_cv": round(glm_full._r2_cv, 4)},
        "gam_resid":{"r2_train": round(gam._r2_train,      4), "r2_cv": round(gam._r2_cv,      4)},
        "combined": {"r2_train": round(r2_final,            4)},
        "variance_decomp": {
            "tier1_pct":    round(t1_pct,  1),
            "tier2_pct":    round(t2_pct,  1),
            "tier3_pct":    round(t3_pct,  1),
            "residual_pct": round(res_pct, 1),
        },
        "n_features": {"t1": len(t1), "t2": len(t2), "t3": len(t3)},
        "feature_lists": {"t1": t1, "t2_extra": T2_EXTRA, "t3": t3},
    }
    with open(MDL_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Models saved to {MDL_DIR}/")
    print("   glm_t1.pkl · glm_full.pkl · gam_residual.pkl · model_metrics.json")
    print("\nDone. Run next: streamlit run app.py")


if __name__ == "__main__":
    train()
