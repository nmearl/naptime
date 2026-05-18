from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

BANDS = ["u", "g", "r", "i", "z", "y"]
BAND2IDX = {band: i for i, band in enumerate(BANDS)}
EBV_COEFFS = {
    "u": 4.757,
    "g": 3.661,
    "r": 2.701,
    "i": 2.054,
    "z": 1.590,
    "y": 1.308,
}


def load_all_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    split_dirs = sorted(data_dir.glob("split_*"))
    lc_parts = [
        pd.read_csv(split_dir / "train_full_lightcurves.csv")
        for split_dir in split_dirs
        if (split_dir / "train_full_lightcurves.csv").exists()
    ]
    log_path = data_dir / "train_log.csv"

    if lc_parts and log_path.exists():
        lc = pd.concat(lc_parts, ignore_index=True)
    else:
        lc_path = data_dir / "train_full_lightcurves.csv"
        if not lc_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Cannot find Mallorn training data under {data_dir!s}")
        lc = pd.read_csv(lc_path)

    log = pd.read_csv(log_path)
    lc = lc.rename(columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "band"})
    keep_cols = [col for col in ["object_id", "Z", "EBV", "target", "SpecType"] if col in log.columns]
    log = log[keep_cols].copy()
    if "target" in log.columns:
        log["target"] = log["target"].fillna(0).astype(int)
    lc = lc[lc["band"].isin(BANDS)].dropna(subset=["flux", "flux_err"])
    lc = lc[lc["flux_err"] > 0].copy()
    return lc, log


def load_test_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    split_dirs = sorted(data_dir.glob("split_*"))
    lc_parts = [
        pd.read_csv(split_dir / "test_full_lightcurves.csv")
        for split_dir in split_dirs
        if (split_dir / "test_full_lightcurves.csv").exists()
    ]
    log_path = data_dir / "test_log.csv"
    if lc_parts and log_path.exists():
        lc = pd.concat(lc_parts, ignore_index=True)
    else:
        lc_path = data_dir / "test_full_lightcurves.csv"
        if not lc_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Cannot find Mallorn test data under {data_dir!s}")
        lc = pd.read_csv(lc_path)

    log = pd.read_csv(log_path)
    lc = lc.rename(columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "band"})
    keep_cols = [col for col in ["object_id", "Z", "EBV", "SpecType"] if col in log.columns]
    log = log[keep_cols].copy()
    lc = lc[lc["band"].isin(BANDS)].dropna(subset=["flux", "flux_err"])
    lc = lc[lc["flux_err"] > 0].copy()
    return lc, log


def apply_ebv_correction(lc: pd.DataFrame, log: pd.DataFrame, ebv_coeffs: dict[str, float] | None = None) -> pd.DataFrame:
    ebv_coeffs = EBV_COEFFS if ebv_coeffs is None else ebv_coeffs
    ebv_map = dict(zip(log["object_id"], log["EBV"].fillna(0.0))) if "EBV" in log.columns else {}
    out = lc.copy()
    out["_ebv"] = out["object_id"].map(ebv_map).fillna(0.0)
    out["_coeff"] = out["band"].map(ebv_coeffs).fillna(0.0)
    scale = 10.0 ** (0.4 * out["_ebv"] * out["_coeff"])
    out["flux"] *= scale
    out["flux_err"] *= scale
    out.drop(columns=["_ebv", "_coeff"], inplace=True)
    return out


def cap_observations(obj: pd.DataFrame, max_obs: int, snr_threshold: float) -> pd.DataFrame:
    if len(obj) <= max_obs:
        return obj.sort_values("mjd").reset_index(drop=True)

    obj = obj.sort_values("mjd").reset_index(drop=True).copy()
    snr = np.abs(obj["flux"].to_numpy()) / (obj["flux_err"].to_numpy() + 1e-6)
    high_snr = np.where(snr > snr_threshold)[0]
    low_snr = np.setdiff1d(np.arange(len(obj)), high_snr)
    budget = max_obs - len(high_snr)

    if budget <= 0:
        idx = np.linspace(0, len(high_snr) - 1, max_obs).round().astype(int)
        chosen = np.unique(high_snr[idx])
    elif len(low_snr) <= budget:
        chosen = np.sort(np.concatenate([high_snr, low_snr]))
    else:
        idx = np.linspace(0, len(low_snr) - 1, budget).round().astype(int)
        chosen = np.sort(np.concatenate([high_snr, np.unique(low_snr[idx])]))

    return obj.iloc[chosen].sort_values("mjd").reset_index(drop=True)


def valid_object_ids(lc: pd.DataFrame, min_obs: int = 3) -> list[str]:
    counts = lc.groupby("object_id").size()
    return counts[counts >= min_obs].index.tolist()


def preprocess_mallorn_training_tables(
    lc: pd.DataFrame,
    log: pd.DataFrame,
    *,
    max_obs: int = 200,
    keep_all_snr_gt: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lc = apply_ebv_correction(lc, log)
    lc_ids = set(lc["object_id"].unique())
    log = log[log["object_id"].isin(lc_ids)].reset_index(drop=True)

    long_obj = lc.groupby("object_id").size()
    long_obj = long_obj[long_obj > max_obs].index.tolist()
    if long_obj:
        parts = [lc[~lc["object_id"].isin(long_obj)]]
        for oid in long_obj:
            parts.append(cap_observations(lc[lc["object_id"] == oid].copy(), max_obs, keep_all_snr_gt))
        lc = pd.concat(parts, ignore_index=True)

    usable_ids = valid_object_ids(lc, min_obs=3)
    lc = lc[lc["object_id"].isin(usable_ids)].reset_index(drop=True)
    log = log[log["object_id"].isin(usable_ids)].reset_index(drop=True)
    return lc, log


def compute_flux_norm_stats(lc: pd.DataFrame, object_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if object_ids is not None:
        lc = lc[lc["object_id"].isin(object_ids)]
    centers = np.zeros(len(BANDS), dtype=np.float32)
    scales = np.ones(len(BANDS), dtype=np.float32)
    for i, band in enumerate(BANDS):
        vals = lc.loc[lc["band"] == band, "flux"].dropna().to_numpy(dtype=np.float32)
        if len(vals) == 0:
            continue
        center = float(np.median(vals))
        mad = float(np.median(np.abs(vals - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            q25, q75 = np.quantile(vals, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349) if (q75 - q25) > 1e-6 else float(np.std(vals))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        centers[i] = center
        scales[i] = scale
    return centers, scales


def normalize_redshift(z_raw: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    span = max(z_max - z_min, 1e-6)
    z_norm = (z_raw.clone() - z_min) / span
    z_norm = z_norm.clamp(0.0, 1.0)
    z_norm[~torch.isfinite(z_norm)] = 0.5
    return z_norm
