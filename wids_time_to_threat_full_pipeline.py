# file: wids_time_to_threat_full_pipeline.py
"""
WiDS Global Datathon 2026 - Time-to-Threat - Full Pipeline (V11.4 - zero-floor + optional rules + SUPER SHARP ladder + one-ceil)

What’s included (vs your V11.3 paste):
1) Keeps V11.3 zero-floor thresholds + propagation (snap tiny probs to 0 + back-propagate zeros).
2) NEW: one-ceil thresholds + propagation (snap near-1 probs to 1 + forward-propagate ones).
3) NEW: --super-sharp postprocess (hard sharpen + optional ladder quantization)
   - hard sharpen (gamma) pushes probabilities toward {0,1}
   - optional ladder quantization snaps probabilities to nearest values in a provided list
4) CLI flags:
   - --super-sharp
   - --sharp-gamma
   - --sharp-top-to-one
   - --ladder "0,0.4074,0.7143,...,1"
   - --one-thresholds "1e-3" OR "t12,t24,t48,t72"
   - --disable-one-ceil

Everything else kept as-is from V11.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import argparse
import numpy as np
import pandas as pd

# Models
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# For scaled LogisticRegression baseline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------
# Global Knobs
# -------------------------

HORIZONS = np.array([12.0, 24.0, 48.0, 72.0], dtype=float)
INTERVALS: List[Tuple[float, float]] = [(0.0, 12.0), (12.0, 24.0), (24.0, 48.0), (48.0, 72.0)]

DEFAULT_BEST_CRITERION = "hybrid_ipcw"  # "mean_logloss" or "hybrid_ipcw"
DEFAULT_MONOTONIC = "isotonic"          # "accumulate" or "isotonic"

GLOBAL_ALPHAS = [0.60, 0.70, 0.80, 0.90, 0.95]
HORIZON_ALPHAS = [0.60, 0.70, 0.80, 0.90, 0.95]

# Hazard model constraint modes
CONSTRAINT_MODES = ["core", "full", "off"]  # used only by hazard_lgb

# Sharpening (CV variants)
ENABLE_SHARPEN_DEFAULT = True
ENABLE_SHARPEN_GAMMA_DEFAULT = False  # default runtime
SHARPEN_TARGET_CANDIDATES_DEFAULT = ("blend_horizon", "blend_global")  # bases to sharpen
SHARPEN_TEMP_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.00, 1.20]  # T<1 sharper
SHARPEN_GAMMA_GRID = [1.00, 1.15, 1.30, 1.45, 1.60, 1.80, 2.00]             # gamma>1 sharper
SHARPEN_ITERS = 2
SHARPEN_MASKS: Dict[str, np.ndarray] = {
    "all": np.array([1, 1, 1, 1], dtype=bool),
    "24_48_72": np.array([0, 1, 1, 1], dtype=bool),
    "48_72": np.array([0, 0, 1, 1], dtype=bool),
}

SUPPORTED_MODELS = ["hazard_lgb", "direct_lgb", "direct_hgb", "direct_rf", "direct_et", "direct_logreg"]

# Default per-horizon snap-to-zero thresholds (12h,24h,48h,72h)
DEFAULT_ZERO_THRESHOLDS = (1e-4, 1e-4, 1e-4, 1e-4)

# Default per-horizon snap-to-one thresholds (12h,24h,48h,72h)
# rule: if p > 1 - thr -> set to 1.0
DEFAULT_ONE_THRESHOLDS = (5e-4, 5e-4, 5e-4, 5e-4)


@dataclass(frozen=True)
class CFG:
    # CV
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 42

    # hazard ensemble (test-time averaging)
    n_seeds_hazard: int = 7
    early_stopping_rounds: int = 200

    # calibration
    ece_bins: int = 10
    platt_C: float = 10.0
    platt_max_iter: int = 2000

    # feature cleaning
    min_non_null_frac: float = 0.01
    corr_threshold: float = 0.995
    top_k_features: int = 90
    winsor_p_low: float = 0.01
    winsor_p_high: float = 0.99

    # hybrid metric weights
    ipcw_brier_horizon_weights: Tuple[float, float, float, float] = (0.0, 0.25, 0.50, 0.25)
    hybrid_cindex_weight: float = 0.3
    hybrid_brier_weight: float = 0.7

    # Base params
    lgb_params_hazard: Dict[str, Any] | None = None
    lgb_params_direct: Dict[str, Any] | None = None

    # Direct model sizes (keep sane)
    rf_params: Dict[str, Any] | None = None
    et_params: Dict[str, Any] | None = None
    hgb_params: Dict[str, Any] | None = None
    logreg_params: Dict[str, Any] | None = None


CFG_INSTANCE = CFG(
    lgb_params_hazard=dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=10000,
        num_leaves=15,
        max_depth=5,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=15.0,
        min_split_gain=0.01,
        random_state=42,  # overridden per fold/seed
        n_jobs=-1,
        verbosity=-1,
    ),
    lgb_params_direct=dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=4000,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=10.0,
        random_state=42,  # overridden per fold
        n_jobs=-1,
        verbosity=-1,
    ),
    rf_params=dict(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=30,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    et_params=dict(
        n_estimators=900,
        max_depth=None,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    hgb_params=dict(
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        min_samples_leaf=40,
        l2_regularization=0.0,
        random_state=42,
    ),
    # robust logreg defaults (used inside a StandardScaler pipeline)
    logreg_params=dict(
        C=1.0,
        solver="saga",
        max_iter=12000,
        class_weight="balanced",
    ),
)


# -------------------------
# IO
# -------------------------

def _resolve_path(filename: str) -> Path:
    p1 = Path(filename)
    if p1.exists():
        return p1
    p2 = Path("/mnt/data") / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {filename} in . or /mnt/data")


# -------------------------
# Feature engineering
# -------------------------

def signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- ensure time columns exist ---
    for col in ["event_start_hour", "event_start_dayofweek", "event_start_month"]:
        if col not in out.columns:
            out[col] = np.nan

    # --- cyclic time encodings ---
    out["hour_sin"] = np.sin(2 * np.pi * out["event_start_hour"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["event_start_hour"].astype(float) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["event_start_dayofweek"].astype(float) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["event_start_dayofweek"].astype(float) / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["event_start_month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["event_start_month"].astype(float) / 12.0)

    # extra time indicators
    hour = out["event_start_hour"].astype(float)
    dow = out["event_start_dayofweek"].astype(float)
    out["is_weekend"] = dow.isin([5.0, 6.0]).astype(float)
    out["is_night"] = ((hour >= 0.0) & (hour <= 6.0)).astype(float)

    eps = 1e-6

    def safe_col(name: str) -> pd.Series:
        return out[name].astype(float) if name in out.columns else pd.Series(np.nan, index=out.index)

    # --- core kinematics / geometry ---
    if "dist_min_ci_0_5h" not in out.columns:
        out["dist_min_ci_0_5h"] = np.nan
    dist = safe_col("dist_min_ci_0_5h").clip(lower=eps)

    closing = safe_col("closing_speed_m_per_h")
    closing_abs = safe_col("closing_speed_abs_m_per_h").abs()
    projected_adv = safe_col("projected_advance_m")
    along = safe_col("along_track_speed")
    cross = safe_col("cross_track_component")
    align = safe_col("alignment_abs")
    radial = safe_col("radial_growth_rate_m_per_h")
    area_rate = safe_col("area_growth_rate_ha_per_h")

    # ratios over distance
    out["closing_over_dist"] = closing / dist
    out["advance_over_dist"] = projected_adv / dist
    out["along_over_dist"] = along / dist
    out["radial_rate_over_dist"] = radial / dist
    out["area_rate_over_dist"] = area_rate / dist

    # interactions
    out["align_x_closing"] = align * closing
    out["align_x_along"] = align * along
    out["abs_cross_track"] = cross.abs()

    # logs / signed logs
    out["log1p_dist_min"] = np.log1p(safe_col("dist_min_ci_0_5h").clip(lower=0.0))
    out["log1p_centroid_speed"] = np.log1p(safe_col("centroid_speed_m_per_h").clip(lower=0.0))
    out["log1p_radial_rate"] = np.log1p(radial.clip(lower=0.0))
    out["slog1p_dist_change"] = signed_log1p(safe_col("dist_change_ci_0_5h"))
    out["slog1p_projected_adv"] = signed_log1p(projected_adv)

    # existing ETA baseline
    out["eta_hours_dist_over_close"] = dist / (closing_abs + 1.0)

    # =========================
    # 1) ETA + Slack by horizon
    # =========================
    for H in [12.0, 24.0, 48.0, 72.0]:
        out[f"eta_{int(H)}h"] = dist / (closing_abs + 1.0)
        out[f"slack_{int(H)}h_m"] = dist - closing_abs * H
        out[f"log1p_eta_{int(H)}h"] = np.log1p(out[f"eta_{int(H)}h"].clip(lower=0.0))
        out[f"slog1p_slack_{int(H)}h"] = signed_log1p(out[f"slack_{int(H)}h_m"])

    # =========================
    # 2) Growth + direction combos
    # =========================
    out["toward_speed"] = closing * align
    out["toward_speed_abs"] = closing_abs * align
    out["toward_radial"] = radial * align
    out["toward_area_rate"] = area_rate * align

    # combined risk proxies
    out["risk_proxy_v1"] = (closing_abs + radial.clip(lower=0.0)) / dist
    out["risk_proxy_v2"] = (projected_adv.abs() + radial.clip(lower=0.0)) / dist
    out["risk_proxy_v3"] = (closing_abs + projected_adv.abs() + radial.clip(lower=0.0)) / dist

    out["log1p_risk_proxy_v1"] = np.log1p(out["risk_proxy_v1"].clip(lower=0.0))
    out["log1p_risk_proxy_v2"] = np.log1p(out["risk_proxy_v2"].clip(lower=0.0))
    out["log1p_risk_proxy_v3"] = np.log1p(out["risk_proxy_v3"].clip(lower=0.0))

    # =========================
    # 3) "Quality of measurement" / temporal coverage features
    # =========================
    num_perim = safe_col("num_perimeters_0_5h")
    dt = safe_col("dt_first_last_0_5h")
    low_tr = safe_col("low_temporal_resolution_0_5h")
    r2 = safe_col("dist_fit_r2_0_5h")

    out["dt_first_last_0_5h_clip"] = dt.clip(lower=0.0)
    out["dense_obs_0_5h"] = (num_perim >= 4.0).astype(float)
    out["good_track_r2_0_5h"] = (r2 >= 0.70).astype(float)
    out["low_temporal_res_flag_0_5h"] = (low_tr > 0.0).astype(float)

    out["dist_change_per_hour_0_5h"] = safe_col("dist_change_ci_0_5h") / (dt.abs() + 1.0)
    out["area_growth_abs_per_hour_0_5h"] = safe_col("area_growth_abs_ha_0_5h") / (dt.abs() + 1.0)

    out["closing_abs_x_goodtrack"] = closing_abs * out["good_track_r2_0_5h"]
    out["radial_x_denseobs"] = radial * out["dense_obs_0_5h"]
    out["risk_v1_x_goodtrack"] = out["risk_proxy_v1"] * out["good_track_r2_0_5h"]
    out["risk_v1_x_lowtempres"] = out["risk_proxy_v1"] * out["low_temporal_res_flag_0_5h"]

    # =========================
    # 4) Binary extremes (regime indicators)
    # =========================
    dist_rank = dist.rank(pct=True)
    close_rank = closing_abs.rank(pct=True)
    radial_rank = radial.rank(pct=True)

    out["dist_rank"] = dist_rank
    out["closing_abs_rank"] = close_rank
    out["radial_rank"] = radial_rank

    out["very_close_q10"] = (dist_rank <= 0.10).astype(float)
    out["very_fast_close_q90"] = (close_rank >= 0.90).astype(float)
    out["very_high_radial_q90"] = (radial_rank >= 0.90).astype(float)

    # =========================
    # 5) Regime bins
    # =========================
    def decile_from_rank(r: pd.Series) -> pd.Series:
        r = r.clip(lower=0.0, upper=1.0).fillna(0.5)
        return np.floor(r * 10.0).clip(lower=0.0, upper=9.0)

    out["dist_decile"] = decile_from_rank(dist_rank)
    out["eta72_rank"] = out["eta_72h"].rank(pct=True)
    out["eta72_decile"] = decile_from_rank(out["eta72_rank"])
    out["risk_v1_rank"] = out["risk_proxy_v1"].rank(pct=True)
    out["risk_v1_decile"] = decile_from_rank(out["risk_v1_rank"])

    # =========================
    # 6) Seasonality interactions
    # =========================
    out["closing_abs_x_month_sin"] = closing_abs * out["month_sin"]
    out["closing_abs_x_month_cos"] = closing_abs * out["month_cos"]
    out["radial_x_month_sin"] = radial * out["month_sin"]
    out["radial_x_month_cos"] = radial * out["month_cos"]
    out["risk_v1_x_month_sin"] = out["risk_proxy_v1"] * out["month_sin"]
    out["risk_v1_x_month_cos"] = out["risk_proxy_v1"] * out["month_cos"]

    return out


# -------------------------
# Robust feature cleaning
# -------------------------

def _winsorize_inplace(df: pd.DataFrame, cols: List[str], q_low: pd.Series, q_high: pd.Series) -> None:
    for c in cols:
        lo = float(q_low.get(c, np.nan))
        hi = float(q_high.get(c, np.nan))
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            df[c] = df[c].clip(lower=lo, upper=hi)


def clean_and_prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    *,
    min_non_null_frac: float,
    winsor_p_low: float,
    winsor_p_high: float,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    tr = train.copy()
    te = test.copy()

    for c in feature_cols:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        te[c] = pd.to_numeric(te[c], errors="coerce")
    tr[feature_cols] = tr[feature_cols].replace([np.inf, -np.inf], np.nan)
    te[feature_cols] = te[feature_cols].replace([np.inf, -np.inf], np.nan)

    # missing flags
    tr_isna = tr[feature_cols].isna().astype(np.int8)
    te_isna = te[feature_cols].isna().astype(np.int8)
    tr_isna.columns = [f"{c}__isna" for c in feature_cols]
    te_isna.columns = [f"{c}__isna" for c in feature_cols]
    tr = pd.concat([tr, tr_isna], axis=1)
    te = pd.concat([te, te_isna], axis=1)
    miss_cols = tr_isna.columns.tolist()

    # sparse filter
    keep = []
    for c in feature_cols:
        tr_frac = float(tr[c].notna().mean()) if len(tr) else 0.0
        te_frac = float(te[c].notna().mean()) if len(te) else 0.0
        if max(tr_frac, te_frac) >= min_non_null_frac:
            keep.append(c)
    dropped_sparse = sorted(set(feature_cols) - set(keep))

    # constants
    keep2 = []
    dropped_constant = []
    for c in keep:
        if tr[c].nunique(dropna=True) >= 2:
            keep2.append(c)
        else:
            dropped_constant.append(c)

    # winsorize + impute
    q_low = tr[keep2].quantile(winsor_p_low, numeric_only=True)
    q_high = tr[keep2].quantile(winsor_p_high, numeric_only=True)
    _winsorize_inplace(tr, keep2, q_low, q_high)
    _winsorize_inplace(te, keep2, q_low, q_high)

    med = tr[keep2].median(numeric_only=True).fillna(0.0)
    tr[keep2] = tr[keep2].fillna(med)
    te[keep2] = te[keep2].fillna(med)

    # correlation filter
    dropped_corr = []
    if len(keep2) >= 2:
        corr = tr[keep2].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
        dropped_corr = sorted(to_drop)
        keep3 = [c for c in keep2 if c not in set(to_drop)]
    else:
        keep3 = keep2

    final_features = keep3 + miss_cols

    report_rows = []
    for c in feature_cols:
        report_rows.append(
            dict(
                feature=c,
                kept=int(c in set(keep3)),
                dropped_sparse=int(c in set(dropped_sparse)),
                dropped_constant=int(c in set(dropped_constant)),
                dropped_corr=int(c in set(dropped_corr)),
                train_non_null_frac=float(train[c].notna().mean()) if c in train.columns else 0.0,
            )
        )
    report = pd.DataFrame(report_rows).sort_values(["kept", "train_non_null_frac"], ascending=[True, True])
    return tr, te, final_features, report


# -------------------------
# Labels
# -------------------------

def build_horizon_labels(train_base: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"event_id": train_base["event_id"].values})
    t = train_base["time_to_hit_hours"].astype(float).values
    e = train_base["event"].astype(int).values
    for h in HORIZONS:
        out[f"y_{int(h)}"] = ((e == 1) & (t <= h)).astype(int)
    return out


# -------------------------
# Monotonic post-process
# -------------------------

def enforce_monotonic_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.maximum.accumulate(p, axis=1), 0.0, 1.0)


def isotonic_project_rows(p: np.ndarray) -> np.ndarray:
    # PAV (pool-adjacent-violators) per row, equal weights
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    n, m = p.shape
    out = np.empty_like(p)

    for i in range(n):
        v = p[i].tolist()
        w = [1.0] * m
        k = 0
        while k < len(v) - 1:
            if v[k] <= v[k + 1]:
                k += 1
                continue
            new_v = (w[k] * v[k] + w[k + 1] * v[k + 1]) / (w[k] + w[k + 1])
            new_w = w[k] + w[k + 1]
            v[k] = new_v
            w[k] = new_w
            del v[k + 1]
            del w[k + 1]
            k = max(k - 1, 0)

        expanded: List[float] = []
        for val, wt in zip(v, w):
            expanded.extend([float(val)] * int(round(wt)))

        arr = np.asarray(expanded, dtype=float)
        if arr.size != m:
            idx = np.linspace(0, max(arr.size - 1, 0), m)
            arr = np.interp(idx, np.arange(arr.size), arr)
        out[i] = np.clip(arr, 0.0, 1.0)

    return out


def apply_monotonic(p: np.ndarray, mode: str) -> np.ndarray:
    return isotonic_project_rows(p) if mode == "isotonic" else enforce_monotonic_probs(p)


# -------------------------
# Zero-floor thresholds + propagation
# -------------------------

def _parse_thresholds_any(s: str, m: int, arg_name: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return np.zeros(m, dtype=float)
    vals = np.asarray([float(x) for x in parts], dtype=float)
    if vals.size == 1:
        return np.full(m, float(vals[0]), dtype=float)
    if vals.size != m:
        raise ValueError(f"{arg_name} must have 1 or {m} values, got {vals.size}")
    return vals


def apply_zero_floor_and_propagate(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Rules:
      1) Snap-to-zero: if prob_h < threshold_h => prob_h = 0
      2) Propagation: if any prob at horizon j is 0, all earlier horizons [0..j] are 0
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0).copy()
    m = p.shape[1]
    thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    if thr.shape[1] != m:
        raise ValueError(f"thresholds shape mismatch: expected {m}, got {thr.shape[1]}")

    # (1) floor
    p[p < thr] = 0.0

    # (2) propagate zeros backward, per row
    is_zero = (p == 0.0)
    idx = np.where(is_zero, np.arange(m, dtype=int)[None, :], m)
    first_zero = idx.min(axis=1)  # m means "no zeros"
    cols = np.arange(m, dtype=int)[None, :]
    keep = (first_zero == m)[:, None] | (cols > first_zero[:, None])
    return p * keep.astype(float)


# -------------------------
# One-ceil thresholds + propagation (near-1 -> 1)
# -------------------------

def apply_one_ceil_and_propagate(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Rules:
      1) Snap-to-one: if prob_h > 1 - threshold_h => prob_h = 1
      2) Propagation: if any prob at horizon j is 1, all later horizons [j..end] are 1
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0).copy()
    m = p.shape[1]
    thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    if thr.shape[1] != m:
        raise ValueError(f"thresholds shape mismatch: expected {m}, got {thr.shape[1]}")

    # (1) ceil
    p[p > (1.0 - thr)] = 1.0

    # (2) propagate ones forward, per row
    is_one = (p == 1.0)
    idx = np.where(is_one, np.arange(m, dtype=int)[None, :], -1)
    last_one = idx.max(axis=1)  # -1 means "no ones"
    cols = np.arange(m, dtype=int)[None, :]
    keep = (last_one == -1)[:, None] | (cols < last_one[:, None])
    p = p * keep.astype(float) + (1.0 - keep.astype(float)) * 1.0
    return p


# -------------------------
# SUPER SHARP postprocess (hard gamma + optional ladder quantization)
# -------------------------

def hard_sharpen_probs(
    probs: np.ndarray,
    *,
    gamma: float = 3.0,
    top_to_one: float = 0.98,
) -> np.ndarray:
    """
    Empuja p hacia {0,1}.
    gamma > 1 => más sharp.
    top_to_one: si p >= este valor => 1.0
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0 - 1e-9)
    g = float(gamma)
    a = np.power(p, g)
    b = np.power(1.0 - p, g)
    out = a / np.clip(a + b, 1e-15, None)
    out = np.clip(out, 0.0, 1.0)
    out[out >= float(top_to_one)] = 1.0
    return out


def quantize_to_ladder(probs: np.ndarray, ladder: List[float]) -> np.ndarray:
    """
    Cuantiza cada prob al valor más cercano en ladder.
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    L = np.asarray(sorted(set([float(x) for x in ladder])), dtype=float)
    dif = np.abs(p[:, :, None] - L[None, None, :])
    idx = np.argmin(dif, axis=2)
    return L[idx]


def _parse_ladder(s: str) -> Optional[List[float]]:
    s = (s or "").strip()
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return vals if len(vals) >= 2 else None


def apply_super_sharp_postprocess(
    probs: np.ndarray,
    *,
    zero_thresholds: np.ndarray,
    one_thresholds: np.ndarray,
    gamma: float,
    top_to_one: float,
    ladder: Optional[List[float]],
    monotonic_mode: str,
    enable_zero: bool,
    enable_one: bool,
) -> np.ndarray:
    """
    Pipeline (order matters):
      1) zero-floor + propagate (optional)
      2) hard gamma sharpen + top_to_one
      3) (optional) ladder quantization
      4) monotonic
      5) one-ceil + propagate (optional)
      6) monotonic
      7) zero-floor again (optional) + monotonic final
    """
    p = np.clip(np.asarray(probs, float), 0.0, 1.0)

    if enable_zero:
        p = apply_zero_floor_and_propagate(p, zero_thresholds)

    p = hard_sharpen_probs(p, gamma=gamma, top_to_one=top_to_one)

    if ladder is not None and len(ladder) >= 2:
        p = quantize_to_ladder(p, ladder)

    p = apply_monotonic(p, monotonic_mode)

    if enable_one:
        p = apply_one_ceil_and_propagate(p, one_thresholds)
        p = apply_monotonic(p, monotonic_mode)

    if enable_zero:
        p = apply_zero_floor_and_propagate(p, zero_thresholds)
        p = apply_monotonic(p, monotonic_mode)

    return np.clip(p, 0.0, 1.0)


# -------------------------
# Optional “behavior rules” (OFF by default)
# -------------------------

def apply_behavior_rules(
    df_features: pd.DataFrame,
    probs: np.ndarray,
    *,
    enable: bool = False,
) -> np.ndarray:
    """
    Hook to inject hard/soft rules based on raw variables or engineered features.

    IMPORTANT: Always re-apply monotonicity after rule edits.
    """
    if not enable:
        return probs

    p = np.asarray(probs, float).copy()
    return np.clip(p, 0.0, 1.0)


# -------------------------
# Sharpening (CV variants)
# -------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sharpen_logit_temperature(p: np.ndarray, temps: Iterable[float]) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    T = np.asarray(list(temps), float).reshape(1, -1)
    z = _logit(p) / np.clip(T, 1e-6, 1e6)
    return np.clip(_sigmoid(z), 0.0, 1.0)


def sharpen_power_gamma(p: np.ndarray, gammas: Iterable[float]) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    g = np.asarray(list(gammas), float).reshape(1, -1)
    a = np.power(p, g)
    b = np.power(1.0 - p, g)
    out = a / np.clip(a + b, 1e-12, None)
    return np.clip(out, 0.0, 1.0)


# -------------------------
# Metrics
# -------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def train_metrics(y_h: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(HORIZONS):
        y = y_h[f"y_{int(h)}"].values.astype(int)
        p = np.clip(probs[:, i], 1e-6, 1 - 1e-6)
        rows.append(
            dict(
                horizon=int(h),
                positives=int(y.sum()),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])),
                ece=float(expected_calibration_error(y, p, n_bins=CFG_INSTANCE.ece_bins)),
            )
        )
    return pd.DataFrame(rows)


def mean_over_horizons(df: pd.DataFrame) -> Dict[str, float]:
    return dict(
        mean_brier=float(df["brier"].mean()),
        mean_logloss=float(df["logloss"].mean()),
        mean_ece=float(df["ece"].mean()),
    )


def km_censor_survival(times: np.ndarray, censored: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, float)
    d = np.asarray(censored, int)

    order = np.argsort(t)
    t = t[order]
    d = d[order]

    uniq = np.unique(t)
    n = len(t)

    at_risk = n
    G = 1.0
    t_out = []
    G_out = []

    for tu in uniq:
        mask = (t == tu)
        di = int(d[mask].sum())
        ni = int(mask.sum())
        if at_risk > 0 and di > 0:
            G *= (1.0 - di / at_risk)
        t_out.append(float(tu))
        G_out.append(float(G))
        at_risk -= ni

    return np.asarray(t_out, float), np.asarray(G_out, float)


def step_survival_at(t_unique: np.ndarray, G_unique: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    tq = np.asarray(t_query, float)
    out = np.ones_like(tq, dtype=float)
    if t_unique.size == 0:
        return out
    idx = np.searchsorted(t_unique, tq, side="right") - 1
    valid = idx >= 0
    out[valid] = G_unique[idx[valid]]
    return np.clip(out, 1e-6, 1.0)


def ipcw_brier_event_prob(
    train_df: pd.DataFrame,
    probs: np.ndarray,
    horizon_weights: Tuple[float, float, float, float],
) -> Dict[str, float]:
    t = train_df["time_to_hit_hours"].astype(float).values
    e = train_df["event"].astype(int).values

    cens = (e == 0).astype(int)
    t_u, G_u = km_censor_survival(t, cens)

    out: Dict[str, float] = {}
    wb = 0.0
    wsum = 0.0

    for i, H in enumerate(HORIZONS.astype(float)):
        p = np.clip(probs[:, i], 1e-6, 1 - 1e-6)

        is_event_by_H = (e == 1) & (t <= H)
        is_at_risk_at_H = t > H
        include = is_event_by_H | is_at_risk_at_H

        if int(include.sum()) == 0:
            out[f"ipcw_brier_{int(H)}"] = float("nan")
            continue

        G_t = step_survival_at(t_u, G_u, t)
        G_H = float(step_survival_at(t_u, G_u, np.asarray([H]))[0])

        w = np.zeros_like(t, dtype=float)
        w[is_event_by_H] = 1.0 / np.clip(G_t[is_event_by_H], 1e-6, 1.0)
        w[is_at_risk_at_H] = 1.0 / np.clip(G_H, 1e-6, 1.0)

        loss = np.zeros_like(t, dtype=float)
        loss[is_event_by_H] = (1.0 - p[is_event_by_H]) ** 2 * w[is_event_by_H]
        loss[is_at_risk_at_H] = (0.0 - p[is_at_risk_at_H]) ** 2 * w[is_at_risk_at_H]

        denom = float(w[include].sum())
        b = float(loss[include].sum() / max(denom, 1e-12))
        out[f"ipcw_brier_{int(H)}"] = b

        hw = float(horizon_weights[i])
        if np.isfinite(b) and hw > 0:
            wb += hw * b
            wsum += hw

    out["ipcw_wbrier"] = float(wb / max(wsum, 1e-12))
    return out


def cindex_canonical(train_df: pd.DataFrame, risk: np.ndarray) -> float:
    t = train_df["time_to_hit_hours"].astype(float).values
    e = train_df["event"].astype(int).values
    r = np.asarray(risk, dtype=float)

    concordant = 0.0
    comparable = 0.0
    n = len(t)

    for i in range(n):
        if e[i] != 1:
            continue
        ti = t[i]
        ri = r[i]
        for j in range(n):
            if ti < t[j]:
                comparable += 1.0
                rj = r[j]
                if ri > rj:
                    concordant += 1.0
                elif ri == rj:
                    concordant += 0.5

    return float(concordant / max(comparable, 1e-12))


def hybrid_ipcw_metrics(train_df: pd.DataFrame, probs: np.ndarray) -> Dict[str, float]:
    b = ipcw_brier_event_prob(train_df, probs, CFG_INSTANCE.ipcw_brier_horizon_weights)
    c = float(cindex_canonical(train_df, probs[:, -1]))
    h = float(CFG_INSTANCE.hybrid_cindex_weight * c + CFG_INSTANCE.hybrid_brier_weight * (1.0 - b["ipcw_wbrier"]))
    out = dict(**b)
    out["cindex"] = c
    out["hybrid_ipcw"] = h
    return out


def score_pack(train_df: pd.DataFrame, y_h: pd.DataFrame, probs: np.ndarray) -> Dict[str, float]:
    std = mean_over_horizons(train_metrics(y_h, probs))
    hyb = hybrid_ipcw_metrics(train_df, probs)
    return dict(**std, **hyb)


def better_by_criterion(a: Dict[str, float], b: Dict[str, float], criterion: str) -> bool:
    if criterion == "hybrid_ipcw":
        return float(a["hybrid_ipcw"]) > float(b["hybrid_ipcw"])
    return float(a["mean_logloss"]) < float(b["mean_logloss"])


# -------------------------
# CV folds
# -------------------------

def build_repeated_folds(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(CFG_INSTANCE.n_repeats):
        rs = CFG_INSTANCE.random_state + 100 * r
        skf = StratifiedKFold(n_splits=CFG_INSTANCE.n_splits, shuffle=True, random_state=rs)
        for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
            folds.append((tr_idx, va_idx))
    return folds


# -------------------------
# Calibration (Platt)
# -------------------------

def _fit_platt_1d(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    clf = LogisticRegression(C=CFG_INSTANCE.platt_C, solver="lbfgs", max_iter=CFG_INSTANCE.platt_max_iter)
    clf.fit(x, y)
    return clf


def cross_fitted_platt(oof_raw: np.ndarray, y_h: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    oof_platt = np.zeros_like(oof_raw, dtype=float)
    for train_idx, val_idx in folds:
        models: List[LogisticRegression] = []
        for i, h in enumerate(HORIZONS):
            models.append(_fit_platt_1d(oof_raw[train_idx, i], y_h.iloc[train_idx][f"y_{int(h)}"].values))
        fold_p = np.zeros((len(val_idx), len(HORIZONS)), dtype=float)
        for i, m in enumerate(models):
            fold_p[:, i] = m.predict_proba(oof_raw[val_idx, i].reshape(-1, 1))[:, 1]
        oof_platt[val_idx] += enforce_monotonic_probs(fold_p)

    counts = np.zeros(len(oof_raw), dtype=float)
    for _, val_idx in folds:
        counts[val_idx] += 1.0
    counts = np.clip(counts, 1.0, None).reshape(-1, 1)
    oof_platt = oof_platt / counts
    return enforce_monotonic_probs(oof_platt)


def fit_platt_full(oof_raw: np.ndarray, y_h: pd.DataFrame) -> List[LogisticRegression]:
    models: List[LogisticRegression] = []
    for i, h in enumerate(HORIZONS):
        models.append(_fit_platt_1d(oof_raw[:, i], y_h[f"y_{int(h)}"].values))
    return models


def apply_platt(p: np.ndarray, models: List[LogisticRegression]) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)
    for i, m in enumerate(models):
        out[:, i] = m.predict_proba(p[:, i].reshape(-1, 1))[:, 1]
    return enforce_monotonic_probs(out)


# -------------------------
# Blending
# -------------------------

def blend_global(p_raw: np.ndarray, p_platt: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(alpha * p_raw + (1.0 - alpha) * p_platt, 0.0, 1.0)


def blend_per_horizon(p_raw: np.ndarray, p_platt: np.ndarray, alphas: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(alphas), dtype=float).reshape(1, -1)
    return np.clip(a * p_raw + (1.0 - a) * p_platt, 0.0, 1.0)


def search_best_alphas(
    train_df: pd.DataFrame,
    y_h: pd.DataFrame,
    oof_raw: np.ndarray,
    oof_platt: np.ndarray,
    criterion: str,
    monotonic: str,
) -> Dict[str, object]:
    rows_g = []
    best_a = float(GLOBAL_ALPHAS[0])
    best_sg: Optional[Dict[str, float]] = None

    for a in GLOBAL_ALPHAS:
        p = apply_monotonic(blend_global(oof_raw, oof_platt, float(a)), monotonic)
        s = score_pack(train_df, y_h, p)
        rows_g.append(dict(alpha=float(a), **s))
        if best_sg is None or better_by_criterion(s, best_sg, criterion):
            best_sg = s
            best_a = float(a)

    df_global = pd.DataFrame(rows_g).sort_values(
        "hybrid_ipcw" if criterion == "hybrid_ipcw" else "mean_logloss",
        ascending=(criterion != "hybrid_ipcw"),
    ).reset_index(drop=True)

    rows_h = []
    best_alphas = [0.90, 0.90, 0.90, 0.90]
    best_sh: Optional[Dict[str, float]] = None

    for a12 in HORIZON_ALPHAS:
        for a24 in HORIZON_ALPHAS:
            for a48 in HORIZON_ALPHAS:
                for a72 in HORIZON_ALPHAS:
                    alphas = [a12, a24, a48, a72]
                    p = apply_monotonic(blend_per_horizon(oof_raw, oof_platt, alphas), monotonic)
                    s = score_pack(train_df, y_h, p)
                    rows_h.append(dict(a12=a12, a24=a24, a48=a48, a72=a72, **s))
                    if best_sh is None or better_by_criterion(s, best_sh, criterion):
                        best_sh = s
                        best_alphas = list(map(float, alphas))

    df_hgrid = pd.DataFrame(rows_h).sort_values(
        "hybrid_ipcw" if criterion == "hybrid_ipcw" else "mean_logloss",
        ascending=(criterion != "hybrid_ipcw"),
    ).reset_index(drop=True)

    return dict(best_global_alpha=best_a, best_horizon_alphas=best_alphas, df_global=df_global, df_hgrid=df_hgrid)


# -------------------------
# Sharpen coordinate search (masked)
# -------------------------

def _sharpen_coordinate_search(
    train_df: pd.DataFrame,
    y_h: pd.DataFrame,
    base_probs: np.ndarray,
    *,
    method: str,
    grid: List[float],
    criterion: str,
    monotonic: str,
    iters: int,
    mask: np.ndarray,
    default_value: float,
) -> Tuple[List[float], Dict[str, float], np.ndarray]:
    p0 = np.clip(np.asarray(base_probs, float), 1e-6, 1.0 - 1e-6)
    m = p0.shape[1]
    params = [float(default_value)] * m

    best_score: Optional[Dict[str, float]] = None
    best_probs: Optional[np.ndarray] = None

    for _ in range(max(1, iters)):
        for j in range(m):
            if not bool(mask[j]):
                continue

            best_val = params[j]
            best_sj: Optional[Dict[str, float]] = None
            best_pj: Optional[np.ndarray] = None

            for v in grid:
                trial = params.copy()
                trial[j] = float(v)

                if method == "temp":
                    p = sharpen_logit_temperature(p0, trial)
                else:
                    p = sharpen_power_gamma(p0, trial)

                p = apply_monotonic(p, monotonic)
                s = score_pack(train_df, y_h, p)

                if best_sj is None or better_by_criterion(s, best_sj, criterion):
                    best_sj = s
                    best_val = float(v)
                    best_pj = p

            params[j] = best_val
            if best_score is None or (best_sj is not None and better_by_criterion(best_sj, best_score, criterion)):
                best_score = best_sj
                best_probs = best_pj

    if best_score is None or best_probs is None:
        p = apply_monotonic(p0, monotonic)
        return params, score_pack(train_df, y_h, p), p

    return params, best_score, best_probs


# -------------------------
# hazard_lgb (discrete-time hazard)
# -------------------------

def _interval_rows_for_event(time_to_hit: float, event: int) -> List[Tuple[int, float, float, int]]:
    t = float(time_to_hit)
    e = int(event)
    rows: List[Tuple[int, float, float, int]] = []
    for k, (a, b) in enumerate(INTERVALS):
        if e == 1:
            if t <= a:
                break
            if t <= b:
                rows.append((k, float(b), float(np.log1p(b)), 1))
                break
            rows.append((k, float(b), float(np.log1p(b)), 0))
        else:
            if b <= t:
                rows.append((k, float(b), float(np.log1p(b)), 0))
            else:
                break
    return rows


def expand_to_interval_rows(df: pd.DataFrame, feature_cols: List[str], is_train: bool) -> pd.DataFrame:
    out_rows = []
    if is_train:
        for _, r in df.iterrows():
            intervals = _interval_rows_for_event(r["time_to_hit_hours"], r["event"])
            for (k, t_end, logt, y) in intervals:
                d = {
                    "event_id": r["event_id"],
                    "interval_idx": int(k),
                    "t_end": float(t_end),
                    "log1p_t_end": float(logt),
                    "y": int(y),
                }
                for c in feature_cols:
                    d[c] = r[c]
                out_rows.append(d)
    else:
        for _, r in df.iterrows():
            for k, (_, b) in enumerate(INTERVALS):
                d = {
                    "event_id": r["event_id"],
                    "interval_idx": int(k),
                    "t_end": float(b),
                    "log1p_t_end": float(np.log1p(b)),
                }
                for c in feature_cols:
                    d[c] = r[c]
                out_rows.append(d)
    return pd.DataFrame(out_rows)


def hazards_to_cdf(hazards: np.ndarray) -> np.ndarray:
    hazards = np.clip(hazards, 1e-6, 1.0 - 1e-6)
    surv = np.cumprod(1.0 - hazards, axis=1)
    cdf = 1.0 - surv
    cdf = np.maximum.accumulate(cdf, axis=1)
    return np.clip(cdf, 0.0, 1.0)


def _constraint_signs_full() -> Dict[str, int]:
    signs = {
        "dist_min_ci_0_5h": -1,
        "log1p_dist_min": -1,
        "dist_change_ci_0_5h": -1,
        "dist_slope_ci_0_5h": -1,
        "slog1p_dist_change": -1,
        "closing_speed_m_per_h": +1,
        "closing_speed_abs_m_per_h": +1,
        "projected_advance_m": +1,
        "slog1p_projected_adv": +1,
        "closing_over_dist": +1,
        "advance_over_dist": +1,
        "radial_growth_rate_m_per_h": +1,
        "area_growth_rate_ha_per_h": +1,
        "radial_rate_over_dist": +1,
        "area_rate_over_dist": +1,
        "log1p_radial_rate": +1,
        "alignment_abs": +1,
        "align_x_closing": +1,
        "eta_hours_dist_over_close": -1,
        "eta_12h": -1,
        "eta_24h": -1,
        "eta_48h": -1,
        "eta_72h": -1,
        "log1p_eta_12h": -1,
        "log1p_eta_24h": -1,
        "log1p_eta_48h": -1,
        "log1p_eta_72h": -1,
        "slack_12h_m": -1,
        "slack_24h_m": -1,
        "slack_48h_m": -1,
        "slack_72h_m": -1,
        "slog1p_slack_12h": -1,
        "slog1p_slack_24h": -1,
        "slog1p_slack_48h": -1,
        "slog1p_slack_72h": -1,
        "risk_proxy_v1": +1,
        "risk_proxy_v2": +1,
        "risk_proxy_v3": +1,
        "log1p_risk_proxy_v1": +1,
        "log1p_risk_proxy_v2": +1,
        "log1p_risk_proxy_v3": +1,
        "dist_rank": -1,
        "dist_decile": -1,
        "closing_abs_rank": +1,
        "radial_rank": +1,
        "eta72_rank": -1,
        "eta72_decile": -1,
        "risk_v1_rank": +1,
        "risk_v1_decile": +1,
        "very_close_q10": +1,
        "very_fast_close_q90": +1,
        "very_high_radial_q90": +1,
        "toward_speed": +1,
        "toward_speed_abs": +1,
        "toward_radial": +1,
        "toward_area_rate": +1,
    }
    return signs


def _constraint_signs_core() -> Dict[str, int]:
    return {
        "dist_min_ci_0_5h": -1,
        "log1p_dist_min": -1,
        "closing_speed_m_per_h": +1,
        "projected_advance_m": +1,
        "radial_growth_rate_m_per_h": +1,
        "area_growth_rate_ha_per_h": +1,
        "eta_hours_dist_over_close": -1,
        "eta_72h": -1,
        "log1p_eta_72h": -1,
        "risk_proxy_v1": +1,
        "log1p_risk_proxy_v1": +1,
        "dist_rank": -1,
        "closing_abs_rank": +1,
    }


def monotone_constraints_vector(feature_cols: List[str], mode: str) -> Optional[List[int]]:
    if mode == "off":
        return None
    signs = _constraint_signs_core() if mode == "core" else _constraint_signs_full()
    vec = [0, 0, 0]  # interval_idx, t_end, log1p_t_end
    vec.extend([signs.get(c, 0) for c in feature_cols])
    return vec


def fit_lgb_hazard(
    train_int: pd.DataFrame,
    valid_int: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    constraints_mode: str,
) -> lgb.LGBMClassifier:
    X_cols = ["interval_idx", "t_end", "log1p_t_end"] + feature_cols
    params = dict(CFG_INSTANCE.lgb_params_hazard or {})
    params["random_state"] = int(seed)

    cvec = monotone_constraints_vector(feature_cols, constraints_mode)
    if cvec is not None:
        params["monotone_constraints"] = cvec

    model = lgb.LGBMClassifier(**params)
    model.fit(
        train_int[X_cols],
        train_int["y"].astype(int),
        eval_set=[(valid_int[X_cols], valid_int["y"].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(CFG_INSTANCE.early_stopping_rounds, verbose=False)],
    )
    return model


def predict_event_hazards(model: lgb.LGBMClassifier, base_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X_cols = ["interval_idx", "t_end", "log1p_t_end"] + feature_cols
    hazards = np.zeros((len(base_df), len(INTERVALS)), dtype=float)

    tmp = base_df.copy()
    tmp["interval_idx"] = 0
    tmp["t_end"] = float(INTERVALS[0][1])
    tmp["log1p_t_end"] = float(np.log1p(INTERVALS[0][1]))

    for k, (_, b) in enumerate(INTERVALS):
        tmp["interval_idx"] = int(k)
        tmp["t_end"] = float(b)
        tmp["log1p_t_end"] = float(np.log1p(b))
        hazards[:, k] = model.predict_proba(tmp[X_cols])[:, 1]
    return hazards


def run_hazard_lgb_base(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    constraints_mode: str,
    *,
    collect_importance: bool = False,
) -> Dict[str, object]:
    y_h = build_horizon_labels(train)

    oof_raw_sum = np.zeros((len(train), len(HORIZONS)), dtype=float)
    oof_counts = np.zeros(len(train), dtype=float)

    y_strat = train["event"].astype(int).values
    folds = build_repeated_folds(y_strat)

    feat_imp = pd.Series(0.0, index=feature_cols, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr_base = train.iloc[tr_idx].reset_index(drop=True)
        va_base = train.iloc[va_idx].reset_index(drop=True)

        tr_int = expand_to_interval_rows(tr_base, feature_cols, is_train=True)
        va_int = expand_to_interval_rows(va_base, feature_cols, is_train=True)

        model = fit_lgb_hazard(
            tr_int,
            va_int,
            feature_cols,
            seed=CFG_INSTANCE.random_state + fold,
            constraints_mode=constraints_mode,
        )

        if collect_importance:
            booster = model.booster_
            imp = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            imp_map = dict(zip(names, imp))
            for c in feature_cols:
                feat_imp[c] += float(imp_map.get(c, 0.0))

        va_haz = predict_event_hazards(model, va_base, feature_cols)
        va_cdf = hazards_to_cdf(va_haz)
        oof_raw_sum[va_idx] += va_cdf
        oof_counts[va_idx] += 1.0

    oof_counts = np.clip(oof_counts, 1.0, None).reshape(-1, 1)
    oof_raw = enforce_monotonic_probs(oof_raw_sum / oof_counts)
    oof_platt = cross_fitted_platt(oof_raw, y_h, folds)

    # test-time: simple seed ensemble using one internal split
    full_int = expand_to_interval_rows(train, feature_cols, is_train=True)
    rng = np.random.default_rng(CFG_INSTANCE.random_state)
    perm = rng.permutation(len(train))
    split = int(0.85 * len(train))
    tr_ids = set(train.iloc[perm[:split]]["event_id"].tolist())
    va_ids = set(train.iloc[perm[split:]]["event_id"].tolist())
    tr_int = full_int[full_int["event_id"].isin(tr_ids)].reset_index(drop=True)
    va_int = full_int[full_int["event_id"].isin(va_ids)].reset_index(drop=True)

    models: List[lgb.LGBMClassifier] = []
    for s in range(CFG_INSTANCE.n_seeds_hazard):
        models.append(
            fit_lgb_hazard(
                tr_int,
                va_int,
                feature_cols,
                seed=CFG_INSTANCE.random_state + 1000 + s,
                constraints_mode=constraints_mode,
            )
        )

    test_haz = np.zeros((len(test), len(INTERVALS)), dtype=float)
    for m in models:
        test_haz += predict_event_hazards(m, test, feature_cols)
    test_haz /= float(len(models))
    test_raw = enforce_monotonic_probs(hazards_to_cdf(test_haz))

    platt_full = fit_platt_full(oof_raw, y_h)
    test_platt = apply_platt(test_raw, platt_full)

    out = dict(
        model="hazard_lgb",
        constraints_mode=constraints_mode,
        y_h=y_h,
        folds=folds,
        oof_raw=oof_raw,
        oof_platt=oof_platt,
        test_raw=test_raw,
        test_platt=test_platt,
    )
    if collect_importance:
        out["feature_importance_gain"] = feat_imp.sort_values(ascending=False)
    return out


# -------------------------
# Direct models (4 independent horizon classifiers)
# -------------------------

def _fit_predict_direct_model(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    y_h = build_horizon_labels(train)
    Xtr = train[feature_cols].values
    Xte = test[feature_cols].values

    oof_raw = np.zeros((len(train), len(HORIZONS)), dtype=float)
    counts = np.zeros(len(train), dtype=float)

    def make_estimator(seed: int):
        if model_name == "direct_lgb":
            params = dict(CFG_INSTANCE.lgb_params_direct or {})
            params["random_state"] = int(seed)
            return lgb.LGBMClassifier(**params)
        if model_name == "direct_hgb":
            params = dict(CFG_INSTANCE.hgb_params or {})
            params["random_state"] = int(seed)
            return HistGradientBoostingClassifier(**params)
        if model_name == "direct_rf":
            params = dict(CFG_INSTANCE.rf_params or {})
            params["random_state"] = int(seed)
            return RandomForestClassifier(**params)
        if model_name == "direct_et":
            params = dict(CFG_INSTANCE.et_params or {})
            params["random_state"] = int(seed)
            return ExtraTreesClassifier(**params)
        if model_name == "direct_logreg":
            params = dict(CFG_INSTANCE.logreg_params or {})
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(**params),
            )
        raise ValueError(f"Unknown direct model {model_name}")

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = Xtr[tr_idx], Xtr[va_idx]

        fold_pred = np.zeros((len(va_idx), len(HORIZONS)), dtype=float)
        for hi, h in enumerate(HORIZONS):
            y_tr = y_h.iloc[tr_idx][f"y_{int(h)}"].values.astype(int)

            est = make_estimator(CFG_INSTANCE.random_state + fold_id + 13 * hi)
            if len(np.unique(y_tr)) < 2:
                p_const = float(np.mean(y_tr))
                fold_pred[:, hi] = p_const
                continue

            est.fit(X_tr, y_tr)
            fold_pred[:, hi] = est.predict_proba(X_va)[:, 1] if hasattr(est, "predict_proba") else est.predict(X_va)

        oof_raw[va_idx] += enforce_monotonic_probs(fold_pred)
        counts[va_idx] += 1.0

    counts = np.clip(counts, 1.0, None).reshape(-1, 1)
    oof_raw = enforce_monotonic_probs(oof_raw / counts)

    oof_platt = cross_fitted_platt(oof_raw, y_h, folds)

    # Train on full data to get test preds
    test_raw = np.zeros((len(test), len(HORIZONS)), dtype=float)
    for hi, h in enumerate(HORIZONS):
        y_full = y_h[f"y_{int(h)}"].values.astype(int)
        est = make_estimator(CFG_INSTANCE.random_state + 999 + 17 * hi)
        if len(np.unique(y_full)) < 2:
            test_raw[:, hi] = float(np.mean(y_full))
            continue
        est.fit(Xtr, y_full)
        test_raw[:, hi] = est.predict_proba(Xte)[:, 1] if hasattr(est, "predict_proba") else est.predict(Xte)

    test_raw = enforce_monotonic_probs(test_raw)

    platt_full = fit_platt_full(oof_raw, y_h)
    test_platt = apply_platt(test_raw, platt_full)

    return dict(
        model=model_name,
        constraints_mode="n/a",
        y_h=y_h,
        folds=folds,
        oof_raw=oof_raw,
        oof_platt=oof_platt,
        test_raw=test_raw,
        test_platt=test_platt,
    )


# -------------------------
# Variants generation & evaluation
# -------------------------

def _format_h_alphas(alphas: List[float]) -> str:
    return ",".join([f"{x:.2f}" for x in alphas])


def _variant_id(model: str, constraints_mode: str, monotonic: str, candidate: str) -> str:
    return f"{model}__{constraints_mode}__{monotonic}__{candidate}"


def evaluate_all_variants_for_base(
    *,
    train_df: pd.DataFrame,
    base_pack: Dict[str, object],
    model: str,
    constraints_mode: str,
    criterion: str,
    monotonic: str,
    enable_sharpen: bool,
    enable_gamma: bool,
    sharpen_targets: Tuple[str, ...],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    y_h: pd.DataFrame = base_pack["y_h"]
    oof_raw: np.ndarray = base_pack["oof_raw"]
    oof_platt: np.ndarray = base_pack["oof_platt"]
    test_raw: np.ndarray = base_pack["test_raw"]
    test_platt: np.ndarray = base_pack["test_platt"]

    alpha_pack = search_best_alphas(train_df, y_h, oof_raw, oof_platt, criterion, monotonic)
    best_global_alpha = float(alpha_pack["best_global_alpha"])
    best_horizon_alphas = list(map(float, alpha_pack["best_horizon_alphas"]))

    oof_candidates: Dict[str, np.ndarray] = {
        "raw": apply_monotonic(oof_raw, monotonic),
        "platt": apply_monotonic(oof_platt, monotonic),
        "blend_global": apply_monotonic(blend_global(oof_raw, oof_platt, best_global_alpha), monotonic),
        "blend_horizon": apply_monotonic(blend_per_horizon(oof_raw, oof_platt, best_horizon_alphas), monotonic),
    }
    test_candidates: Dict[str, np.ndarray] = {
        "raw": apply_monotonic(test_raw, monotonic),
        "platt": apply_monotonic(test_platt, monotonic),
        "blend_global": apply_monotonic(blend_global(test_raw, test_platt, best_global_alpha), monotonic),
        "blend_horizon": apply_monotonic(blend_per_horizon(test_raw, test_platt, best_horizon_alphas), monotonic),
    }

    sharpen_meta: Dict[str, Dict[str, str]] = {
        k: {"sharp_method": "", "sharp_mask": "", "sharp_params": ""} for k in oof_candidates
    }

    if enable_sharpen:
        for base_name in sharpen_targets:
            if base_name not in oof_candidates:
                continue

            base_oof = oof_candidates[base_name]
            base_test = test_candidates[base_name]

            for mask_name, mask in SHARPEN_MASKS.items():
                t_params, _, t_oof = _sharpen_coordinate_search(
                    train_df,
                    y_h,
                    base_oof,
                    method="temp",
                    grid=SHARPEN_TEMP_GRID,
                    criterion=criterion,
                    monotonic=monotonic,
                    iters=SHARPEN_ITERS,
                    mask=mask,
                    default_value=1.0,
                )
                t_test = apply_monotonic(sharpen_logit_temperature(base_test, t_params), monotonic)

                suffix = "sharpT" if mask_name == "all" else f"sharpT_{mask_name}"
                name_t = f"{base_name}_{suffix}"
                oof_candidates[name_t] = t_oof
                test_candidates[name_t] = t_test
                sharpen_meta[name_t] = {
                    "sharp_method": "temp",
                    "sharp_mask": mask_name,
                    "sharp_params": ",".join([f"{x:.3f}" for x in t_params]),
                }

            if enable_gamma:
                for mask_name, mask in SHARPEN_MASKS.items():
                    g_params, _, g_oof = _sharpen_coordinate_search(
                        train_df,
                        y_h,
                        base_oof,
                        method="gamma",
                        grid=SHARPEN_GAMMA_GRID,
                        criterion=criterion,
                        monotonic=monotonic,
                        iters=SHARPEN_ITERS,
                        mask=mask,
                        default_value=1.0,
                    )
                    g_test = apply_monotonic(sharpen_power_gamma(base_test, g_params), monotonic)

                    suffix = "sharpG" if mask_name == "all" else f"sharpG_{mask_name}"
                    name_g = f"{base_name}_{suffix}"
                    oof_candidates[name_g] = g_oof
                    test_candidates[name_g] = g_test
                    sharpen_meta[name_g] = {
                        "sharp_method": "gamma",
                        "sharp_mask": mask_name,
                        "sharp_params": ",".join([f"{x:.3f}" for x in g_params]),
                    }

    rows: List[Dict[str, Any]] = []
    cand_scores: Dict[str, Dict[str, float]] = {}
    for cand_name, oof_p in oof_candidates.items():
        s = score_pack(train_df, y_h, oof_p)
        cand_scores[cand_name] = s
        meta = sharpen_meta.get(cand_name, {"sharp_method": "", "sharp_mask": "", "sharp_params": ""})
        rows.append(
            dict(
                variant_id=_variant_id(model, constraints_mode, monotonic, cand_name),
                model=model,
                constraints_mode=constraints_mode,
                monotonic=monotonic,
                best_global_alpha=best_global_alpha,
                best_horizon_alphas=_format_h_alphas(best_horizon_alphas),
                candidate=cand_name,
                sharpen_method=meta["sharp_method"],
                sharpen_mask=meta["sharp_mask"],
                sharpen_params=meta["sharp_params"],
                **s,
            )
        )

    if criterion == "hybrid_ipcw":
        best_name = max(cand_scores.keys(), key=lambda k: cand_scores[k]["hybrid_ipcw"])
    else:
        best_name = min(cand_scores.keys(), key=lambda k: cand_scores[k]["mean_logloss"])

    best = dict(
        best_candidate=best_name,
        best_global_alpha=best_global_alpha,
        best_horizon_alphas=best_horizon_alphas,
        oof_best=oof_candidates[best_name],
        test_best=test_candidates[best_name],
        scores_best=cand_scores[best_name],
        sharp_method=sharpen_meta.get(best_name, {}).get("sharp_method", ""),
        sharp_mask=sharpen_meta.get(best_name, {}).get("sharp_mask", ""),
        sharp_params=sharpen_meta.get(best_name, {}).get("sharp_params", ""),
        df_global=alpha_pack["df_global"],
        df_hgrid=alpha_pack["df_hgrid"],
    )
    return rows, best


# -------------------------
# Feature selection helper
# -------------------------

def select_top_k_features(feature_cols: List[str], importance_gain: pd.Series, top_k: int) -> List[str]:
    if top_k <= 0:
        return feature_cols
    base = [c for c in feature_cols if not c.endswith("__isna")]
    flags = [c for c in feature_cols if c.endswith("__isna")]
    imp = importance_gain.reindex(base).fillna(0.0).sort_values(ascending=False)
    k = min(top_k, len(imp))
    chosen = imp.head(k).index.tolist()
    return chosen + flags


# -------------------------
# Submission
# -------------------------

def write_submission(sample: pd.DataFrame, probs: np.ndarray, out_path: str) -> None:
    sub = sample.copy()
    sub["prob_12h"] = probs[:, 0]
    sub["prob_24h"] = probs[:, 1]
    sub["prob_48h"] = probs[:, 2]
    sub["prob_72h"] = probs[:, 3]
    sub.to_csv(out_path, index=False)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--best-criterion",
        type=str,
        default=DEFAULT_BEST_CRITERION,
        choices=["mean_logloss", "hybrid_ipcw"],
        help="Selection criterion",
    )
    p.add_argument(
        "--monotonic",
        type=str,
        default=DEFAULT_MONOTONIC,
        choices=["accumulate", "isotonic"],
        help="Monotonic postprocess method",
    )
    p.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated list of models or 'all'. Supported: {','.join(SUPPORTED_MODELS)}",
    )
    p.add_argument("--no-sharpen", action="store_true", help="Disable sharpening variants")
    p.add_argument("--enable-gamma", action="store_true", help="Enable gamma sharpening variants (slower)")
    p.add_argument(
        "--sharpen-targets",
        type=str,
        default=",".join(SHARPEN_TARGET_CANDIDATES_DEFAULT),
        help="Comma-separated base candidates to sharpen (e.g. blend_horizon,blend_global)",
    )
    p.add_argument("--enable-rules", action="store_true", help="Enable rule-based postprocess (OFF by default)")

    # zero-floor knobs
    p.add_argument(
        "--zero-thresholds",
        type=str,
        default=",".join([f"{x:g}" for x in DEFAULT_ZERO_THRESHOLDS]),
        help="Snap-to-zero thresholds. Provide 1 value or 4 (12h,24h,48h,72h).",
    )
    p.add_argument(
        "--disable-zero-floor",
        action="store_true",
        help="Disable zero-floor + propagation postprocess.",
    )

    # one-ceil knobs
    p.add_argument(
        "--one-thresholds",
        type=str,
        default=",".join([f"{x:g}" for x in DEFAULT_ONE_THRESHOLDS]),
        help="Snap-to-one thresholds. Provide 1 value or 4 (12h,24h,48h,72h). If p > 1-thr => 1.",
    )
    p.add_argument(
        "--disable-one-ceil",
        action="store_true",
        help="Disable one-ceil + forward propagation postprocess.",
    )

    # SUPER SHARP knobs
    p.add_argument("--super-sharp", action="store_true", help="Enable hard-sharpen + optional ladder quantization.")
    p.add_argument("--sharp-gamma", type=float, default=3.0, help="Gamma for hard-sharpen (higher = sharper).")
    p.add_argument("--sharp-top-to-one", type=float, default=0.98, help="If p>=x => 1.0")
    p.add_argument(
        "--ladder",
        type=str,
        default="",
        help="Comma-separated ladder values for quantization. Example: '0,0.4074074,0.7142857,0.8823529,1'",
    )

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    # Models selection
    if args.models.strip().lower() == "all":
        models_to_run = SUPPORTED_MODELS
    else:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
        bad = [m for m in models_to_run if m not in SUPPORTED_MODELS]
        if bad:
            raise ValueError(f"Unknown model(s): {bad}. Supported: {SUPPORTED_MODELS}")

    enable_sharpen = (not args.no_sharpen) and ENABLE_SHARPEN_DEFAULT
    enable_gamma = bool(args.enable_gamma)
    sharpen_targets = tuple([x.strip() for x in args.sharpen_targets.split(",") if x.strip()])

    # Load data
    train = pd.read_csv(_resolve_path("train.csv"))
    test = pd.read_csv(_resolve_path("test.csv"))
    # sample_submission.csv is the Kaggle template (event_id + target columns).
    # It is not tracked in this repo; when absent, rebuild it from the test ids so
    # the pipeline still runs end-to-end. write_submission fills the prob_* columns
    # positionally, so keeping test's row order here is the correct alignment.
    try:
        sample = pd.read_csv(_resolve_path("sample_submission.csv"))
    except FileNotFoundError:
        sample = test[["event_id"]].copy()

    # FE
    train = add_feature_engineering(train)
    test = add_feature_engineering(test)

    raw_feature_cols = [c for c in train.columns if c not in ("event_id", "time_to_hit_hours", "event")]

    train, test, feature_cols, report = clean_and_prepare_features(
        train,
        test,
        raw_feature_cols,
        min_non_null_frac=CFG_INSTANCE.min_non_null_frac,
        winsor_p_low=CFG_INSTANCE.winsor_p_low,
        winsor_p_high=CFG_INSTANCE.winsor_p_high,
        corr_threshold=CFG_INSTANCE.corr_threshold,
    )
    report.to_csv("feature_report.csv", index=False)

    # Shared folds for direct models
    y_strat = train["event"].astype(int).values
    folds = build_repeated_folds(y_strat)

    # Pass 1 probe for feature selection
    if CFG_INSTANCE.top_k_features > 0 and "hazard_lgb" in models_to_run:
        print("\n--- Pass 1: importance probe (hazard_lgb constraints_mode=core) ---", flush=True)
        probe = run_hazard_lgb_base(train, test, feature_cols, constraints_mode="core", collect_importance=True)
        imp_gain: pd.Series = probe["feature_importance_gain"]
        feature_cols = select_top_k_features(feature_cols, imp_gain, CFG_INSTANCE.top_k_features)
        print(
            f"Selected Top-K features: K={CFG_INSTANCE.top_k_features} (+ missing flags). Total={len(feature_cols)}",
            flush=True,
        )
    else:
        print(f"\n--- Top-K probe skipped. Using features: {len(feature_cols)} ---", flush=True)

    # Storage
    all_variant_rows: List[Dict[str, Any]] = []
    best_per_model_rows: List[Dict[str, Any]] = []

    best_overall: Optional[Dict[str, Any]] = None
    best_val = -float("inf") if args.best_criterion == "hybrid_ipcw" else float("inf")

    best_oof: Optional[np.ndarray] = None
    best_test: Optional[np.ndarray] = None
    best_meta: Optional[Dict[str, Any]] = None

    # ---- Run models ----
    for mdl in models_to_run:
        if mdl == "hazard_lgb":
            for cmode in CONSTRAINT_MODES:
                print(f"\n--- Running {mdl} constraints_mode={cmode} ---", flush=True)
                base_pack = run_hazard_lgb_base(
                    train, test, feature_cols, constraints_mode=cmode, collect_importance=False
                )

                rows, best = evaluate_all_variants_for_base(
                    train_df=train,
                    base_pack=base_pack,
                    model=mdl,
                    constraints_mode=cmode,
                    criterion=args.best_criterion,
                    monotonic=args.monotonic,
                    enable_sharpen=enable_sharpen,
                    enable_gamma=enable_gamma,
                    sharpen_targets=sharpen_targets,
                )
                all_variant_rows.extend(rows)

                s = best["scores_best"]
                best_per_model_rows.append(
                    dict(
                        model=mdl,
                        constraints_mode=cmode,
                        monotonic=args.monotonic,
                        best_candidate=best["best_candidate"],
                        best_global_alpha=best["best_global_alpha"],
                        best_horizon_alphas=_format_h_alphas(best["best_horizon_alphas"]),
                        sharp_method=best["sharp_method"],
                        sharp_mask=best["sharp_mask"],
                        sharp_params=best["sharp_params"],
                        **s,
                    )
                )

                cur = float(s["hybrid_ipcw"]) if args.best_criterion == "hybrid_ipcw" else float(s["mean_logloss"])
                better = cur > best_val if args.best_criterion == "hybrid_ipcw" else cur < best_val
                if better:
                    best_val = cur
                    best_overall = dict(model=mdl, constraints_mode=cmode, monotonic=args.monotonic, **best)
                    best_oof = best["oof_best"]
                    best_test = best["test_best"]
                    best_meta = dict(model=mdl, constraints_mode=cmode)

                print(
                    f"Finished {mdl} cmode={cmode} | best={best['best_candidate']} | {args.best_criterion}={cur:.6f}",
                    flush=True,
                )

        else:
            print(f"\n--- Running {mdl} ---", flush=True)
            base_pack = _fit_predict_direct_model(mdl, train, test, feature_cols, folds)

            rows, best = evaluate_all_variants_for_base(
                train_df=train,
                base_pack=base_pack,
                model=mdl,
                constraints_mode="n/a",
                criterion=args.best_criterion,
                monotonic=args.monotonic,
                enable_sharpen=enable_sharpen,
                enable_gamma=enable_gamma,
                sharpen_targets=sharpen_targets,
            )
            all_variant_rows.extend(rows)

            s = best["scores_best"]
            best_per_model_rows.append(
                dict(
                    model=mdl,
                    constraints_mode="n/a",
                    monotonic=args.monotonic,
                    best_candidate=best["best_candidate"],
                    best_global_alpha=best["best_global_alpha"],
                    best_horizon_alphas=_format_h_alphas(best["best_horizon_alphas"]),
                    sharp_method=best["sharp_method"],
                    sharp_mask=best["sharp_mask"],
                    sharp_params=best["sharp_params"],
                    **s,
                )
            )

            cur = float(s["hybrid_ipcw"]) if args.best_criterion == "hybrid_ipcw" else float(s["mean_logloss"])
            better = cur > best_val if args.best_criterion == "hybrid_ipcw" else cur < best_val
            if better:
                best_val = cur
                best_overall = dict(model=mdl, constraints_mode="n/a", monotonic=args.monotonic, **best)
                best_oof = best["oof_best"]
                best_test = best["test_best"]
                best_meta = dict(model=mdl, constraints_mode="n/a")

            print(f"Finished {mdl} | best={best['best_candidate']} | {args.best_criterion}={cur:.6f}", flush=True)

    # ---- Save variant tables ----
    df_variants = pd.DataFrame(all_variant_rows)
    if not df_variants.empty:
        df_variants = df_variants.sort_values(
            "hybrid_ipcw" if args.best_criterion == "hybrid_ipcw" else "mean_logloss",
            ascending=(args.best_criterion != "hybrid_ipcw"),
        ).reset_index(drop=True)
    df_variants.to_csv("metrics_variants_compare.csv", index=False)

    df_best_per_model = pd.DataFrame(best_per_model_rows)
    if not df_best_per_model.empty:
        df_best_per_model = df_best_per_model.sort_values(
            "hybrid_ipcw" if args.best_criterion == "hybrid_ipcw" else "mean_logloss",
            ascending=(args.best_criterion != "hybrid_ipcw"),
        ).reset_index(drop=True)
    df_best_per_model.to_csv("metrics_best_per_model.csv", index=False)

    # ---- Best overall outputs ----
    assert best_overall is not None and best_oof is not None and best_test is not None and best_meta is not None

    # thresholds parsed
    thr0 = _parse_thresholds_any(args.zero_thresholds, m=len(HORIZONS), arg_name="--zero-thresholds")
    thr1 = _parse_thresholds_any(args.one_thresholds, m=len(HORIZONS), arg_name="--one-thresholds")

    enable_zero = not bool(args.disable_zero_floor)
    enable_one = not bool(args.disable_one_ceil)

    # optional rules
    if bool(args.enable_rules):
        best_oof = apply_behavior_rules(train, best_oof, enable=True)
        best_test = apply_behavior_rules(test, best_test, enable=True)
        best_oof = apply_monotonic(best_oof, args.monotonic)
        best_test = apply_monotonic(best_test, args.monotonic)

    # baseline end-stage postprocess (if not super-sharp)
    if not bool(args.super_sharp):
        if enable_zero:
            best_oof = apply_zero_floor_and_propagate(best_oof, thr0)
            best_test = apply_zero_floor_and_propagate(best_test, thr0)
            best_oof = apply_monotonic(best_oof, args.monotonic)
            best_test = apply_monotonic(best_test, args.monotonic)

        if enable_one:
            best_oof = apply_one_ceil_and_propagate(best_oof, thr1)
            best_test = apply_one_ceil_and_propagate(best_test, thr1)
            best_oof = apply_monotonic(best_oof, args.monotonic)
            best_test = apply_monotonic(best_test, args.monotonic)

    # SUPER SHARP final postprocess (optionally quantized)
    if bool(args.super_sharp):
        ladder = _parse_ladder(args.ladder)
        best_oof = apply_super_sharp_postprocess(
            best_oof,
            zero_thresholds=thr0,
            one_thresholds=thr1,
            gamma=float(args.sharp_gamma),
            top_to_one=float(args.sharp_top_to_one),
            ladder=ladder,
            monotonic_mode=args.monotonic,
            enable_zero=enable_zero,
            enable_one=enable_one,
        )
        best_test = apply_super_sharp_postprocess(
            best_test,
            zero_thresholds=thr0,
            one_thresholds=thr1,
            gamma=float(args.sharp_gamma),
            top_to_one=float(args.sharp_top_to_one),
            ladder=ladder,
            monotonic_mode=args.monotonic,
            enable_zero=enable_zero,
            enable_one=enable_one,
        )

    best_row = dict(
        chosen_model=best_meta["model"],
        constraints_mode=best_meta["constraints_mode"],
        monotonic=args.monotonic,
        criterion=args.best_criterion,
        best_candidate=best_overall["best_candidate"],
        best_global_alpha=best_overall["best_global_alpha"],
        best_horizon_alphas=_format_h_alphas(best_overall["best_horizon_alphas"]),
        sharp_method=best_overall["sharp_method"],
        sharp_mask=best_overall["sharp_mask"],
        sharp_params=best_overall["sharp_params"],
        zero_floor_enabled=int(enable_zero),
        zero_thresholds=str(args.zero_thresholds),
        one_ceil_enabled=int(enable_one),
        one_thresholds=str(args.one_thresholds),
        super_sharp_enabled=int(bool(args.super_sharp)),
        super_sharp_gamma=float(args.sharp_gamma),
        super_sharp_top_to_one=float(args.sharp_top_to_one),
        ladder=str(args.ladder),
        **best_overall["scores_best"],
    )
    pd.DataFrame([best_row]).to_csv("metrics_best_overall.csv", index=False)

    # This is saved in the CURRENT working directory where you run the script:
    # ./submission_best_overall.csv
    write_submission(sample, best_test, "submission_best_overall.csv")

    df_oof = pd.DataFrame(
        {
            "event_id": train["event_id"].values,
            "prob_12h_oof": best_oof[:, 0],
            "prob_24h_oof": best_oof[:, 1],
            "prob_48h_oof": best_oof[:, 2],
            "prob_72h_oof": best_oof[:, 3],
            "chosen_model": best_meta["model"],
            "constraints_mode": best_meta["constraints_mode"],
            "candidate": best_overall["best_candidate"],
            "criterion": args.best_criterion,
            "monotonic": args.monotonic,
            "rules_enabled": int(bool(args.enable_rules)),
            "zero_floor_enabled": int(enable_zero),
            "zero_thresholds": str(args.zero_thresholds),
            "one_ceil_enabled": int(enable_one),
            "one_thresholds": str(args.one_thresholds),
            "super_sharp_enabled": int(bool(args.super_sharp)),
            "sharp_gamma": float(args.sharp_gamma),
            "sharp_top_to_one": float(args.sharp_top_to_one),
            "ladder": str(args.ladder),
        }
    )
    df_test = pd.DataFrame(
        {
            "event_id": test["event_id"].values,
            "prob_12h": best_test[:, 0],
            "prob_24h": best_test[:, 1],
            "prob_48h": best_test[:, 2],
            "prob_72h": best_test[:, 3],
            "chosen_model": best_meta["model"],
            "constraints_mode": best_meta["constraints_mode"],
            "candidate": best_overall["best_candidate"],
            "criterion": args.best_criterion,
            "monotonic": args.monotonic,
            "rules_enabled": int(bool(args.enable_rules)),
            "zero_floor_enabled": int(enable_zero),
            "zero_thresholds": str(args.zero_thresholds),
            "one_ceil_enabled": int(enable_one),
            "one_thresholds": str(args.one_thresholds),
            "super_sharp_enabled": int(bool(args.super_sharp)),
            "sharp_gamma": float(args.sharp_gamma),
            "sharp_top_to_one": float(args.sharp_top_to_one),
            "ladder": str(args.ladder),
        }
    )
    df_oof.to_csv("preds_oof_bestoverall.csv", index=False)
    df_test.to_csv("preds_test_bestoverall.csv", index=False)

    print("\n=== WROTE FILES ===")
    for f in [
        "feature_report.csv",
        "metrics_variants_compare.csv",
        "metrics_best_per_model.csv",
        "metrics_best_overall.csv",
        "preds_oof_bestoverall.csv",
        "preds_test_bestoverall.csv",
        "submission_best_overall.csv",
    ]:
        print(f"- {Path(f).resolve()}")

    print("\n=== BEST OVERALL ===")
    print(pd.DataFrame([best_row]).to_string(index=False))


if __name__ == "__main__":
    main()
