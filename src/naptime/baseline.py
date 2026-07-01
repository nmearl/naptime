from dataclasses import asdict, dataclass
import gc
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .data import (
    BAND2IDX,
    BANDS,
    apply_ebv_correction,
    cap_observations,
    compute_flux_norm_stats,
    load_all_data,
    load_test_data,
    normalize_redshift,
    preprocess_mallorn_training_tables,
    valid_object_ids,
)
from .elasticc import extract_elasticc_object_from_ref
from .modules import (
    ConvBackbone1D,
    FourierTimeEmbedding,
    GaussianSetConv1D,
    GlobalLatentEncoder,
    MLP,
)

MIN_FLUX_SCALE = 1.0
MAX_ABS_FLUX_NORM = 200.0
MAX_FERR_NORM = 200.0


@dataclass(slots=True)
class BaselineBatch:
    context_x: torch.Tensor
    context_y: torch.Tensor
    context_yerr: torch.Tensor
    context_band: torch.Tensor
    context_mask: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor
    target_yerr: torch.Tensor
    target_band: torch.Tensor
    target_mask: torch.Tensor
    labels: torch.Tensor
    redshift: torch.Tensor
    t_span_log: torch.Tensor
    n_obs_log: torch.Tensor
    meta_values: torch.Tensor
    meta_mask: torch.Tensor
    object_ids: list[str]

    def to(self, device: torch.device | str) -> "BaselineBatch":
        return BaselineBatch(
            context_x=self.context_x.to(device),
            context_y=self.context_y.to(device),
            context_yerr=self.context_yerr.to(device),
            context_band=self.context_band.to(device),
            context_mask=self.context_mask.to(device),
            target_x=self.target_x.to(device),
            target_y=self.target_y.to(device),
            target_yerr=self.target_yerr.to(device),
            target_band=self.target_band.to(device),
            target_mask=self.target_mask.to(device),
            labels=self.labels.to(device),
            redshift=self.redshift.to(device),
            t_span_log=self.t_span_log.to(device),
            n_obs_log=self.n_obs_log.to(device),
            meta_values=self.meta_values.to(device),
            meta_mask=self.meta_mask.to(device),
            object_ids=self.object_ids,
        )


@dataclass(slots=True)
class BaselineCollateConfig:
    seed: int
    z_min: float
    z_max: float
    mask_prob: float = 0.4  # used when mask_prob_min == mask_prob_max (fixed masking)
    mask_prob_min: float = 0.4  # lower bound for variable masking during training
    mask_prob_max: float = 0.4  # upper bound; set min < max to enable variable masking
    min_context_points: int = 3
    min_target_points: int = 1
    deterministic_val_fraction: float = 0.6
    epoch: int = 0


@dataclass(slots=True)
class ConvGNPBaselineConfig:
    num_bands: int = 6
    grid_size: int = 256
    grid_min: float = 0.0
    grid_max: float = 1.0
    band_emb_dim: int = 8
    time_fourier_dim: int = 8
    point_feat_dim: int = 64
    grid_feat_dim: int = 128
    conv_hidden_dim: int = 128
    conv_layers: int = 6
    conv_kernel_size: int = 5
    conv_dropout: float = 0.0
    classifier_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    setconv_sigmas: tuple[float, ...] = (0.015, 0.03, 0.06)
    min_std: float = 1e-3
    use_redshift: bool = True
    use_rest_frame_time: bool = False
    use_metadata: bool = False
    metadata_dim: int = 0
    metadata_hidden_dim: int = 64
    metadata_embed_dim: int = 32
    use_latent: bool = True
    latent_dim: int = 8
    latent_hidden_dim: int = 64
    num_classes: int = 1


@dataclass(slots=True)
class ConvGNPBaselineOutput:
    pred_mean: torch.Tensor
    pred_var: torch.Tensor
    class_logits: torch.Tensor
    grid_features: torch.Tensor
    latent_mu: torch.Tensor | None = None
    latent_logvar: torch.Tensor | None = None


@dataclass(slots=True)
class BaselineLossConfig:
    lambda_recon: float = 1.0
    lambda_cls: float = 1.0
    pos_weight: float | None = None
    beta_kl: float = 1e-3
    kl_warmup_epochs: int = (
        0  # linearly anneal beta_kl from 0 to beta_kl over this many epochs
    )
    class_weights: tuple[float, ...] | None = None


@dataclass(slots=True)
class BaselineLosses:
    total: torch.Tensor
    recon: torch.Tensor
    cls: torch.Tensor
    kl: torch.Tensor | None = None


MALLORN_CLASS_NAMES: list[str] = ["TDE", "AGN", "SLSN", "SN-Ia", "SN-II", "SN-Ibc"]


def map_mallorn_spectype_to_class(spec_type: object) -> int:
    if spec_type is None or (
        isinstance(spec_type, float) and not np.isfinite(spec_type)
    ):
        return -1
    text = str(spec_type).strip()
    if text == "TDE":
        return 0
    if text == "AGN":
        return 1
    if text.startswith("SLSN"):
        return 2
    if text.startswith("SN Ia") or text.startswith("SN Iax"):
        return 3
    if text in {"SN II", "SN IIn", "SN IIP"}:
        return 4
    if text in {"SN Ib", "SN Ic", "SN Ic-BL", "SN IIb"}:
        return 5
    return -1


def build_mallorn_task_log(log, *, num_classes: int = 1):
    task_log = log.copy()
    if num_classes <= 1:
        task_log["task_target"] = task_log["target"].fillna(0).astype(int)
        return task_log
    task_log["task_target"] = (
        task_log["SpecType"].map(map_mallorn_spectype_to_class).fillna(-1).astype(int)
    )
    task_log = task_log[task_log["task_target"] >= 0].reset_index(drop=True)
    return task_log


class MallornBaselineDataset(Dataset):
    def __init__(
        self,
        object_ids: list[str],
        lc,
        log,
        *,
        flux_center_by_band: np.ndarray | None = None,
        flux_scale_by_band: np.ndarray | None = None,
        use_rest_frame_time: bool = False,
    ):
        target_col = "task_target" if "task_target" in log.columns else "target"
        target_map = (
            dict(zip(log["object_id"], log[target_col]))
            if target_col in log.columns
            else {}
        )
        z_map = (
            dict(zip(log["object_id"], log["Z"].fillna(np.nan).astype(float)))
            if "Z" in log.columns
            else {}
        )
        lc_grp = lc.groupby("object_id")
        self.flux_center_by_band = (
            np.zeros(len(BANDS), dtype=np.float32)
            if flux_center_by_band is None
            else np.asarray(flux_center_by_band, dtype=np.float32)
        )
        self.flux_scale_by_band = (
            np.ones(len(BANDS), dtype=np.float32)
            if flux_scale_by_band is None
            else np.asarray(flux_scale_by_band, dtype=np.float32)
        )
        self.data: dict[str, dict] = {}
        for oid in object_ids:
            if oid not in lc_grp.groups:
                continue
            obj = lc_grp.get_group(oid).sort_values("mjd").reset_index(drop=True)
            if len(obj) < 4:
                continue
            t_raw = obj["mjd"].to_numpy(dtype=np.float32)
            flux_raw = obj["flux"].to_numpy(dtype=np.float32)
            ferr_raw = obj["flux_err"].to_numpy(dtype=np.float32)
            band_idx = obj["band"].map(BAND2IDX).to_numpy(dtype=np.int64)
            z_val = float(z_map.get(oid, np.nan))
            t_series = t_raw.copy()
            if use_rest_frame_time and np.isfinite(z_val):
                t_series = (t_series - float(t_series.min())) / max(1.0 + z_val, 1e-6)
            t_min = float(t_series.min())
            t_span = max(float(t_series.max() - t_series.min()), 1.0)
            t_norm = (t_series - t_min) / t_span
            band_centers = self.flux_center_by_band[band_idx]
            band_scales = np.maximum(self.flux_scale_by_band[band_idx], MIN_FLUX_SCALE)
            flux_norm = np.clip(
                (flux_raw - band_centers) / band_scales,
                -MAX_ABS_FLUX_NORM,
                MAX_ABS_FLUX_NORM,
            )
            ferr_norm = np.clip(ferr_raw / band_scales, 1e-3, MAX_FERR_NORM)
            self.data[oid] = {
                "t_norm": t_norm.astype(np.float32),
                "flux_norm": flux_norm.astype(np.float32),
                "ferr_norm": ferr_norm.astype(np.float32),
                "band_idx": band_idx.astype(np.int64),
                "target": int(target_map.get(oid, 0)),
                "z": z_val,
                "t_series_span": float(t_series.max() - t_series.min()),
                "n_obs": len(t_raw),
                "meta_values": np.zeros(0, dtype=np.float32),
                "meta_mask": np.zeros(0, dtype=np.float32),
            }
        self.object_ids = [oid for oid in object_ids if oid in self.data]

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict:
        oid = self.object_ids[idx]
        out = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self.data[oid].items()
        }
        out["oid"] = oid
        return out


class PrecomputedBaselineDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        *,
        flux_center_by_band: np.ndarray,
        flux_scale_by_band: np.ndarray,
        use_rest_frame_time: bool = False,
        meta_center: np.ndarray | None = None,
        meta_scale: np.ndarray | None = None,
    ):
        self.flux_center_by_band = np.asarray(flux_center_by_band, dtype=np.float32)
        self.flux_scale_by_band = np.asarray(flux_scale_by_band, dtype=np.float32)
        self.meta_center = meta_center
        self.meta_scale = meta_scale
        self.data: dict[str, dict] = {}
        for record in records:
            oid = str(record["oid"])
            t_raw = np.asarray(record["t_raw"], dtype=np.float32)
            flux_raw = np.asarray(record["flux_raw"], dtype=np.float32)
            ferr_raw = np.asarray(record["ferr_raw"], dtype=np.float32)
            band_idx = np.asarray(record["band_idx"], dtype=np.int64)
            if len(t_raw) < 4:
                continue
            z_val = float(record.get("z", np.nan))
            t_series = t_raw.copy()
            if use_rest_frame_time and np.isfinite(z_val):
                t_series = (t_series - float(t_series.min())) / max(1.0 + z_val, 1e-6)
            t_min = float(t_series.min())
            t_span = max(float(t_series.max() - t_series.min()), 1.0)
            t_norm = (t_series - t_min) / t_span
            band_centers = self.flux_center_by_band[band_idx]
            band_scales = np.maximum(self.flux_scale_by_band[band_idx], MIN_FLUX_SCALE)
            flux_norm = np.clip(
                (flux_raw - band_centers) / band_scales,
                -MAX_ABS_FLUX_NORM,
                MAX_ABS_FLUX_NORM,
            )
            ferr_norm = np.clip(ferr_raw / band_scales, 1e-3, MAX_FERR_NORM)
            meta_values = np.asarray(
                record.get("meta_values", np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            )
            meta_mask = np.asarray(
                record.get("meta_mask", np.zeros_like(meta_values)), dtype=np.float32
            )
            if self.meta_center is not None and len(meta_values) > 0:
                meta_values = (meta_values - self.meta_center) / np.where(
                    self.meta_scale > 1e-6, self.meta_scale, 1.0
                )
                meta_values = np.where(meta_mask > 0, meta_values, 0.0).astype(
                    np.float32
                )
                meta_values = np.clip(meta_values, -10.0, 10.0)
            self.data[oid] = {
                "t_norm": t_norm.astype(np.float32),
                "flux_norm": flux_norm.astype(np.float32),
                "ferr_norm": ferr_norm.astype(np.float32),
                "band_idx": band_idx.astype(np.int64),
                "target": int(record.get("target", 0)),
                "z": z_val,
                "t_series_span": float(t_series.max() - t_series.min()),
                "n_obs": len(t_raw),
                "meta_values": meta_values,
                "meta_mask": meta_mask,
            }
        self.object_ids = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict:
        oid = self.object_ids[idx]
        out = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self.data[oid].items()
        }
        out["oid"] = oid
        return out


class LazyElasticcDataset(Dataset):
    def __init__(
        self,
        refs: list[dict],
        *,
        flux_center_by_band: np.ndarray,
        flux_scale_by_band: np.ndarray,
        use_rest_frame_time: bool = False,
        meta_center: np.ndarray | None = None,
        meta_scale: np.ndarray | None = None,
        max_cached_shards: int = 2,
    ):
        self.refs = list(refs)
        self.flux_center_by_band = np.asarray(flux_center_by_band, dtype=np.float32)
        self.flux_scale_by_band = np.asarray(flux_scale_by_band, dtype=np.float32)
        self.use_rest_frame_time = bool(use_rest_frame_time)
        self.meta_center = (
            None if meta_center is None else np.asarray(meta_center, dtype=np.float32)
        )
        self.meta_scale = (
            None if meta_scale is None else np.asarray(meta_scale, dtype=np.float32)
        )
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.object_ids = [str(ref["oid"]) for ref in self.refs]
        self.target_by_oid = {
            str(ref["oid"]): int(ref.get("target", 0)) for ref in self.refs
        }
        self.z_values = np.array(
            [float(ref.get("z", np.nan)) for ref in self.refs], dtype=np.float32
        )
        self._shard_cache: dict[
            tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = {}
        self._shard_cache_order: list[tuple[str, str]] = []

    def __len__(self) -> int:
        return len(self.refs)

    def _cache_get_or_load(self, ref: dict) -> dict:
        cache_key = (str(ref["head_path"]), str(ref["phot_path"]))
        if cache_key not in self._shard_cache:
            while len(self._shard_cache_order) >= self.max_cached_shards:
                stale_key = self._shard_cache_order.pop(0)
                self._shard_cache.pop(stale_key, None)
            gc.collect()
        record = extract_elasticc_object_from_ref(ref, shard_cache=self._shard_cache)
        if cache_key in self._shard_cache_order:
            self._shard_cache_order.remove(cache_key)
        self._shard_cache_order.append(cache_key)
        self._shard_cache_order = [
            key for key in self._shard_cache_order if key in self._shard_cache
        ]
        while len(self._shard_cache_order) > self.max_cached_shards:
            stale_key = self._shard_cache_order.pop(0)
            self._shard_cache.pop(stale_key, None)
            gc.collect()
        return record

    def __getitem__(self, idx: int) -> dict:
        ref = self.refs[idx]
        record = self._cache_get_or_load(ref)
        oid = str(record["oid"])
        t_raw = np.asarray(record["t_raw"], dtype=np.float32)
        flux_raw = np.asarray(record["flux_raw"], dtype=np.float32)
        ferr_raw = np.asarray(record["ferr_raw"], dtype=np.float32)
        band_idx = np.asarray(record["band_idx"], dtype=np.int64)
        z_val = float(record.get("z", np.nan))
        t_series = t_raw.copy()
        if self.use_rest_frame_time and np.isfinite(z_val):
            t_series = (t_series - float(t_series.min())) / max(1.0 + z_val, 1e-6)
        t_min = float(t_series.min())
        t_span = max(float(t_series.max() - t_series.min()), 1.0)
        t_norm = (t_series - t_min) / t_span
        band_centers = self.flux_center_by_band[band_idx]
        band_scales = np.maximum(self.flux_scale_by_band[band_idx], MIN_FLUX_SCALE)
        flux_norm = np.clip(
            (flux_raw - band_centers) / band_scales,
            -MAX_ABS_FLUX_NORM,
            MAX_ABS_FLUX_NORM,
        )
        ferr_norm = np.clip(ferr_raw / band_scales, 1e-3, MAX_FERR_NORM)
        meta_values = np.asarray(
            record.get("meta_values", np.zeros(0, dtype=np.float32)), dtype=np.float32
        )
        meta_mask = np.asarray(
            record.get("meta_mask", np.zeros_like(meta_values)), dtype=np.float32
        )
        if self.meta_center is not None and len(meta_values) > 0:
            safe_scale = np.where(
                np.asarray(self.meta_scale) > 1e-6, self.meta_scale, 1.0
            )
            meta_values = (meta_values - self.meta_center) / safe_scale
            meta_values = np.where(meta_mask > 0, meta_values, 0.0).astype(np.float32)
            meta_values = np.clip(meta_values, -10.0, 10.0)
        return {
            "oid": oid,
            "t_norm": t_norm.astype(np.float32),
            "flux_norm": flux_norm.astype(np.float32),
            "ferr_norm": ferr_norm.astype(np.float32),
            "band_idx": band_idx.astype(np.int64),
            "target": int(record.get("target", 0)),
            "z": z_val,
            "t_series_span": float(t_series.max() - t_series.min()),
            "n_obs": len(t_raw),
            "meta_values": meta_values,
            "meta_mask": meta_mask,
        }


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16) % (2**32)


def prepare_mallorn_baseline_datasets(
    data_dir: str | Path,
    *,
    seed: int = 42,
    val_frac: float = 0.15,
    max_obs: int = 200,
    keep_all_snr_gt: float = 5.0,
    use_rest_frame_time: bool = False,
    num_classes: int = 1,
):
    lc, log = load_all_data(data_dir)
    lc, log = preprocess_mallorn_training_tables(
        lc, log, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt
    )
    log = build_mallorn_task_log(log, num_classes=num_classes)
    lc = lc[lc["object_id"].isin(log["object_id"])].reset_index(drop=True)
    all_ids = log["object_id"].tolist()
    all_tgts = log["task_target"].tolist()
    train_ids, val_ids = train_test_split(
        all_ids, test_size=val_frac, stratify=all_tgts, random_state=seed
    )
    flux_center_by_band, flux_scale_by_band = compute_flux_norm_stats(lc, train_ids)
    train_ds = MallornBaselineDataset(
        train_ids,
        lc,
        log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
    )
    val_ds = MallornBaselineDataset(
        val_ids,
        lc,
        log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
    )
    return lc, log, train_ds, val_ds, train_ds.object_ids, val_ds.object_ids


def load_mallorn_training_tables(
    data_dir: str | Path,
    *,
    max_obs: int = 200,
    keep_all_snr_gt: float = 5.0,
):
    lc, log = load_all_data(data_dir)
    return preprocess_mallorn_training_tables(
        lc, log, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt
    )


def build_mallorn_baseline_datasets(
    lc,
    log,
    *,
    train_ids: list[str],
    val_ids: list[str],
    use_rest_frame_time: bool = False,
    num_classes: int = 1,
):
    log = build_mallorn_task_log(log, num_classes=num_classes)
    keep_ids = set(log["object_id"])
    lc = lc[lc["object_id"].isin(keep_ids)].reset_index(drop=True)
    train_ids = [oid for oid in train_ids if oid in keep_ids]
    val_ids = [oid for oid in val_ids if oid in keep_ids]
    flux_center_by_band, flux_scale_by_band = compute_flux_norm_stats(lc, train_ids)
    train_ds = MallornBaselineDataset(
        train_ids,
        lc,
        log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
    )
    val_ds = MallornBaselineDataset(
        val_ids,
        lc,
        log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
    )
    return train_ds, val_ds


def compute_flux_norm_stats_from_records(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(len(BANDS), dtype=np.float32)
    scales = np.ones(len(BANDS), dtype=np.float32)
    for i in range(len(BANDS)):
        vals = []
        for record in records:
            band_idx = np.asarray(record["band_idx"])
            mask = band_idx == i
            if np.any(mask):
                vals.append(np.asarray(record["flux_raw"], dtype=np.float32)[mask])
        if not vals:
            continue
        band_vals = np.concatenate(vals).astype(np.float32)
        center = float(np.median(band_vals))
        mad = float(np.median(np.abs(band_vals - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            q25, q75 = np.quantile(band_vals, [0.25, 0.75])
            scale = (
                float((q75 - q25) / 1.349)
                if (q75 - q25) > 1e-6
                else float(np.std(band_vals))
            )
        if not np.isfinite(scale) or scale < MIN_FLUX_SCALE:
            scale = MIN_FLUX_SCALE
        centers[i] = center
        scales[i] = scale
    return centers, scales


def compute_flux_norm_stats_from_refs(
    refs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(len(BANDS), dtype=np.float32)
    scales = np.ones(len(BANDS), dtype=np.float32)
    shard_cache: dict[
        tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = {}
    for i in range(len(BANDS)):
        vals = []
        for ref in refs:
            cache_key = (str(ref["head_path"]), str(ref["phot_path"]))
            if cache_key not in shard_cache:
                shard_cache.clear()
                gc.collect()
            record = extract_elasticc_object_from_ref(ref, shard_cache=shard_cache)
            band_idx = np.asarray(record["band_idx"])
            mask = band_idx == i
            if np.any(mask):
                vals.append(np.asarray(record["flux_raw"], dtype=np.float32)[mask])
        if not vals:
            continue
        band_vals = np.concatenate(vals).astype(np.float32)
        center = float(np.median(band_vals))
        mad = float(np.median(np.abs(band_vals - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            q25, q75 = np.quantile(band_vals, [0.25, 0.75])
            scale = (
                float((q75 - q25) / 1.349)
                if (q75 - q25) > 1e-6
                else float(np.std(band_vals))
            )
        if not np.isfinite(scale) or scale < MIN_FLUX_SCALE:
            scale = MIN_FLUX_SCALE
        centers[i] = center
        scales[i] = scale
    return centers, scales


def compute_meta_norm_stats_from_records(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    all_values = [
        np.asarray(
            r.get("meta_values", np.zeros(0, dtype=np.float32)), dtype=np.float32
        )
        for r in records
    ]
    all_masks = [
        np.asarray(r.get("meta_mask", np.zeros_like(v)), dtype=np.float32)
        for v, r in zip(all_values, records)
    ]
    if not all_values or all_values[0].shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    n_fields = all_values[0].shape[0]
    centers = np.zeros(n_fields, dtype=np.float32)
    scales = np.ones(n_fields, dtype=np.float32)
    for j in range(n_fields):
        vals = np.array(
            [v[j] for v, m in zip(all_values, all_masks) if m[j] > 0], dtype=np.float32
        )
        if len(vals) < 2:
            continue
        center = float(np.median(vals))
        mad = float(np.median(np.abs(vals - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(vals)) or 1.0
        centers[j] = center
        scales[j] = scale
    return centers, scales


def compute_meta_norm_stats_from_refs(
    refs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    all_values = [
        np.asarray(
            ref.get("meta_values", np.zeros(0, dtype=np.float32)), dtype=np.float32
        )
        for ref in refs
    ]
    all_masks = [
        np.asarray(ref.get("meta_mask", np.zeros_like(v)), dtype=np.float32)
        for v, ref in zip(all_values, refs)
    ]
    if not all_values or all_values[0].shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    n_fields = all_values[0].shape[0]
    centers = np.zeros(n_fields, dtype=np.float32)
    scales = np.ones(n_fields, dtype=np.float32)
    for j in range(n_fields):
        vals = np.array(
            [v[j] for v, m in zip(all_values, all_masks) if m[j] > 0], dtype=np.float32
        )
        if len(vals) < 2:
            continue
        center = float(np.median(vals))
        mad = float(np.median(np.abs(vals - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(vals)) or 1.0
        centers[j] = center
        scales[j] = scale
    return centers, scales


def build_precomputed_baseline_datasets(
    *,
    train_records: list[dict],
    val_records: list[dict],
    use_rest_frame_time: bool = False,
):
    flux_center_by_band, flux_scale_by_band = compute_flux_norm_stats_from_records(
        train_records
    )
    meta_center, meta_scale = compute_meta_norm_stats_from_records(train_records)
    train_ds = PrecomputedBaselineDataset(
        train_records,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
        meta_center=meta_center,
        meta_scale=meta_scale,
    )
    val_ds = PrecomputedBaselineDataset(
        val_records,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
        meta_center=meta_center,
        meta_scale=meta_scale,
    )
    return train_ds, val_ds


def build_lazy_elasticc_datasets(
    *,
    train_refs: list[dict],
    val_refs: list[dict],
    use_rest_frame_time: bool = False,
    max_cached_shards: int = 1,
):
    flux_center_by_band, flux_scale_by_band = compute_flux_norm_stats_from_refs(
        train_refs
    )
    meta_center, meta_scale = compute_meta_norm_stats_from_refs(train_refs)
    train_ds = LazyElasticcDataset(
        train_refs,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
        meta_center=meta_center,
        meta_scale=meta_scale,
        max_cached_shards=max_cached_shards,
    )
    val_ds = LazyElasticcDataset(
        val_refs,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
        meta_center=meta_center,
        meta_scale=meta_scale,
        max_cached_shards=max_cached_shards,
    )
    return train_ds, val_ds


def prepare_mallorn_baseline_test_dataset(
    data_dir: str | Path,
    *,
    flux_center_by_band: np.ndarray,
    flux_scale_by_band: np.ndarray,
    max_obs: int = 200,
    keep_all_snr_gt: float = 5.0,
    use_rest_frame_time: bool = False,
):
    lc, log = load_test_data(data_dir)
    lc, log = preprocess_test_tables(
        lc, log, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt
    )
    object_ids = log["object_id"].tolist()
    ds = MallornBaselineDataset(
        object_ids,
        lc,
        log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        use_rest_frame_time=use_rest_frame_time,
    )
    return lc, log, ds, ds.object_ids


def compute_z_bounds(
    dataset: MallornBaselineDataset, p1: float = 0.01, p99: float = 0.99
) -> tuple[float, float]:
    if hasattr(dataset, "z_values"):
        z_vals = np.asarray(dataset.z_values, dtype=np.float32)
        z_vals = z_vals[np.isfinite(z_vals)]
    else:
        z_vals = np.array(
            [row["z"] for row in dataset.data.values() if np.isfinite(row["z"])],
            dtype=np.float32,
        )
    if len(z_vals) == 0:
        return 0.0, 1.0
    return float(np.quantile(z_vals, p1)), float(np.quantile(z_vals, p99))


def preprocess_test_tables(
    lc, log, *, max_obs: int = 200, keep_all_snr_gt: float = 5.0
):
    lc = apply_ebv_correction(lc, log)
    lc_ids = set(lc["object_id"].unique())
    log = log[log["object_id"].isin(lc_ids)].reset_index(drop=True)
    long_obj = lc.groupby("object_id").size()
    long_obj = long_obj[long_obj > max_obs].index.tolist()
    if long_obj:
        parts = [lc[~lc["object_id"].isin(long_obj)]]
        for oid in long_obj:
            parts.append(
                cap_observations(
                    lc[lc["object_id"] == oid].copy(), max_obs, keep_all_snr_gt
                )
            )
        lc = __import__("pandas").concat(parts, ignore_index=True)
    usable_ids = valid_object_ids(lc, min_obs=3)
    lc = lc[lc["object_id"].isin(usable_ids)].reset_index(drop=True)
    log = log[log["object_id"].isin(usable_ids)].reset_index(drop=True)
    return lc, log


def _pad_1d(
    values: list[torch.Tensor], pad_value: float = 0.0, dtype=None
) -> torch.Tensor:
    max_len = max((len(v) for v in values), default=1)
    dtype = values[0].dtype if values else (torch.float32 if dtype is None else dtype)
    out = torch.full((len(values), max_len), pad_value, dtype=dtype)
    for i, value in enumerate(values):
        if len(value) > 0:
            out[i, : len(value)] = value
    return out


def _split_indices(
    n: int, *, n_ctx: int, min_target: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_ctx = max(1, min(n_ctx, n - min_target))
    perm = rng.permutation(n)
    ctx_idx = np.sort(perm[:n_ctx])
    tgt_idx = np.sort(perm[n_ctx:])
    return ctx_idx, tgt_idx


def collate_baseline_batch(
    samples: Iterable[dict], *, training: bool, cfg: BaselineCollateConfig
) -> BaselineBatch:
    samples = list(samples)
    ctx_x, ctx_y, ctx_yerr, ctx_band, ctx_mask = [], [], [], [], []
    tgt_x, tgt_y, tgt_yerr, tgt_band, tgt_mask = [], [], [], [], []
    labels, redshifts, t_span_logs, n_obs_logs, object_ids = [], [], [], [], []
    meta_values, meta_masks = [], []

    for sample in samples:
        n = len(sample["t_norm"])
        if training:
            rng = np.random.default_rng(
                _stable_seed(cfg.seed, sample["oid"], cfg.epoch)
            )
            if cfg.mask_prob_min < cfg.mask_prob_max:
                mask_prob_i = float(rng.uniform(cfg.mask_prob_min, cfg.mask_prob_max))
            else:
                mask_prob_i = cfg.mask_prob
            n_ctx = max(cfg.min_context_points, int(round((1.0 - mask_prob_i) * n)))
        else:
            rng = np.random.default_rng(_stable_seed(cfg.seed, sample["oid"], "val"))
            n_ctx = max(
                cfg.min_context_points, int(round(cfg.deterministic_val_fraction * n))
            )
        ctx_idx, tgt_idx = _split_indices(
            n, n_ctx=n_ctx, min_target=cfg.min_target_points, rng=rng
        )

        ctx_x.append(torch.as_tensor(sample["t_norm"][ctx_idx], dtype=torch.float32))
        ctx_y.append(torch.as_tensor(sample["flux_norm"][ctx_idx], dtype=torch.float32))
        ctx_yerr.append(
            torch.as_tensor(sample["ferr_norm"][ctx_idx], dtype=torch.float32)
        )
        ctx_band.append(torch.as_tensor(sample["band_idx"][ctx_idx], dtype=torch.long))
        ctx_mask.append(torch.ones(len(ctx_idx), dtype=torch.float32))

        tgt_x.append(torch.as_tensor(sample["t_norm"][tgt_idx], dtype=torch.float32))
        tgt_y.append(torch.as_tensor(sample["flux_norm"][tgt_idx], dtype=torch.float32))
        tgt_yerr.append(
            torch.as_tensor(sample["ferr_norm"][tgt_idx], dtype=torch.float32)
        )
        tgt_band.append(torch.as_tensor(sample["band_idx"][tgt_idx], dtype=torch.long))
        tgt_mask.append(torch.ones(len(tgt_idx), dtype=torch.float32))

        labels.append(float(sample["target"]))
        redshifts.append(float(sample["z"]))
        ctx_t = sample["t_norm"][ctx_idx]
        ctx_span_frac = float(ctx_t.max() - ctx_t.min()) if len(ctx_idx) > 1 else 0.0
        ctx_t_span = ctx_span_frac * float(sample.get("t_series_span", 1.0))
        t_span_logs.append(float(np.log(ctx_t_span + 1.0)))
        n_obs_logs.append(float(np.log(len(ctx_idx) + 1.0)))
        meta_values.append(
            torch.as_tensor(
                sample.get("meta_values", np.zeros(0, dtype=np.float32)),
                dtype=torch.float32,
            )
        )
        meta_masks.append(
            torch.as_tensor(
                sample.get("meta_mask", np.zeros(0, dtype=np.float32)),
                dtype=torch.float32,
            )
        )
        object_ids.append(str(sample["oid"]))

    return BaselineBatch(
        context_x=_pad_1d(ctx_x),
        context_y=_pad_1d(ctx_y),
        context_yerr=_pad_1d(ctx_yerr, pad_value=1.0),
        context_band=_pad_1d(ctx_band, pad_value=0, dtype=torch.long).long(),
        context_mask=_pad_1d(ctx_mask),
        target_x=_pad_1d(tgt_x),
        target_y=_pad_1d(tgt_y),
        target_yerr=_pad_1d(tgt_yerr, pad_value=1.0),
        target_band=_pad_1d(tgt_band, pad_value=0, dtype=torch.long).long(),
        target_mask=_pad_1d(tgt_mask),
        labels=torch.as_tensor(labels, dtype=torch.float32),
        redshift=normalize_redshift(
            torch.as_tensor(redshifts, dtype=torch.float32), cfg.z_min, cfg.z_max
        ),
        t_span_log=torch.as_tensor(t_span_logs, dtype=torch.float32),
        n_obs_log=torch.as_tensor(n_obs_logs, dtype=torch.float32),
        meta_values=_pad_1d(meta_values),
        meta_mask=_pad_1d(meta_masks),
        object_ids=object_ids,
    )


def make_baseline_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    training: bool,
    cfg: BaselineCollateConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=lambda samples: collate_baseline_batch(
            samples, training=training, cfg=cfg
        ),
    )


def make_full_context_batch(
    items: list[dict], *, z_min: float, z_max: float
) -> BaselineBatch:
    context_x = [torch.as_tensor(item["t_norm"], dtype=torch.float32) for item in items]
    context_y = [
        torch.as_tensor(item["flux_norm"], dtype=torch.float32) for item in items
    ]
    context_yerr = [
        torch.as_tensor(item["ferr_norm"], dtype=torch.float32) for item in items
    ]
    context_band = [
        torch.as_tensor(item["band_idx"], dtype=torch.long) for item in items
    ]
    context_mask = [
        torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items
    ]
    target_x = [torch.as_tensor(item["t_norm"], dtype=torch.float32) for item in items]
    target_y = [torch.zeros(len(item["t_norm"]), dtype=torch.float32) for item in items]
    target_yerr = [
        torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items
    ]
    target_band = [
        torch.as_tensor(item["band_idx"], dtype=torch.long) for item in items
    ]
    target_mask = [
        torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items
    ]
    labels = [float(item.get("target", 0.0)) for item in items]
    redshifts = [float(item.get("z", np.nan)) for item in items]
    t_span_logs = [
        float(np.log(float(item.get("t_series_span", 1.0)) + 1.0)) for item in items
    ]
    n_obs_logs = [float(np.log(float(item.get("n_obs", 1)) + 1.0)) for item in items]
    meta_values = [
        torch.as_tensor(
            item.get("meta_values", np.zeros(0, dtype=np.float32)), dtype=torch.float32
        )
        for item in items
    ]
    meta_masks = [
        torch.as_tensor(
            item.get("meta_mask", np.zeros(0, dtype=np.float32)), dtype=torch.float32
        )
        for item in items
    ]
    object_ids = [str(item["oid"]) for item in items]

    return BaselineBatch(
        context_x=_pad_1d(context_x),
        context_y=_pad_1d(context_y),
        context_yerr=_pad_1d(context_yerr, pad_value=1.0),
        context_band=_pad_1d(context_band, pad_value=0, dtype=torch.long).long(),
        context_mask=_pad_1d(context_mask),
        target_x=_pad_1d(target_x),
        target_y=_pad_1d(target_y),
        target_yerr=_pad_1d(target_yerr, pad_value=1.0),
        target_band=_pad_1d(target_band, pad_value=0, dtype=torch.long).long(),
        target_mask=_pad_1d(target_mask),
        labels=torch.as_tensor(labels, dtype=torch.float32),
        redshift=normalize_redshift(
            torch.as_tensor(redshifts, dtype=torch.float32), z_min, z_max
        ),
        t_span_log=torch.as_tensor(t_span_logs, dtype=torch.float32),
        n_obs_log=torch.as_tensor(n_obs_logs, dtype=torch.float32),
        meta_values=_pad_1d(meta_values),
        meta_mask=_pad_1d(meta_masks),
        object_ids=object_ids,
    )


def make_prefix_context_batch(
    items: list[dict],
    *,
    z_min: float,
    z_max: float,
    context_fraction: float,
    min_context_points: int = 3,
    min_target_points: int = 1,
) -> BaselineBatch:
    context_fraction = float(context_fraction)
    if context_fraction <= 0.0 or context_fraction > 1.0:
        raise ValueError(f"context_fraction must be in (0, 1], got {context_fraction}")

    ctx_x, ctx_y, ctx_yerr, ctx_band, ctx_mask = [], [], [], [], []
    tgt_x, tgt_y, tgt_yerr, tgt_band, tgt_mask = [], [], [], [], []
    labels, redshifts, t_span_logs, n_obs_logs, object_ids = [], [], [], [], []
    meta_values, meta_masks = [], []

    for item in items:
        n = len(item["t_norm"])
        if context_fraction >= 1.0:
            n_ctx = n
        else:
            n_ctx = max(min_context_points, int(round(context_fraction * n)))
            n_ctx = max(1, min(n_ctx, n - min_target_points))
        ctx_idx = np.arange(n_ctx, dtype=int)
        tgt_idx = np.arange(n_ctx, n, dtype=int)

        ctx_x.append(torch.as_tensor(item["t_norm"][ctx_idx], dtype=torch.float32))
        ctx_y.append(torch.as_tensor(item["flux_norm"][ctx_idx], dtype=torch.float32))
        ctx_yerr.append(
            torch.as_tensor(item["ferr_norm"][ctx_idx], dtype=torch.float32)
        )
        ctx_band.append(torch.as_tensor(item["band_idx"][ctx_idx], dtype=torch.long))
        ctx_mask.append(torch.ones(len(ctx_idx), dtype=torch.float32))

        tgt_x.append(torch.as_tensor(item["t_norm"][tgt_idx], dtype=torch.float32))
        tgt_y.append(torch.as_tensor(item["flux_norm"][tgt_idx], dtype=torch.float32))
        tgt_yerr.append(
            torch.as_tensor(item["ferr_norm"][tgt_idx], dtype=torch.float32)
        )
        tgt_band.append(torch.as_tensor(item["band_idx"][tgt_idx], dtype=torch.long))
        tgt_mask.append(torch.ones(len(tgt_idx), dtype=torch.float32))

        labels.append(float(item.get("target", 0.0)))
        redshifts.append(float(item.get("z", np.nan)))
        ctx_t = item["t_norm"][ctx_idx]
        ctx_span_frac = float(ctx_t.max() - ctx_t.min()) if len(ctx_idx) > 1 else 0.0
        ctx_t_span = ctx_span_frac * float(item.get("t_series_span", 1.0))
        t_span_logs.append(float(np.log(ctx_t_span + 1.0)))
        n_obs_logs.append(float(np.log(len(ctx_idx) + 1.0)))
        meta_values.append(
            torch.as_tensor(
                item.get("meta_values", np.zeros(0, dtype=np.float32)),
                dtype=torch.float32,
            )
        )
        meta_masks.append(
            torch.as_tensor(
                item.get("meta_mask", np.zeros(0, dtype=np.float32)),
                dtype=torch.float32,
            )
        )
        object_ids.append(str(item["oid"]))

    return BaselineBatch(
        context_x=_pad_1d(ctx_x),
        context_y=_pad_1d(ctx_y),
        context_yerr=_pad_1d(ctx_yerr, pad_value=1.0),
        context_band=_pad_1d(ctx_band, pad_value=0, dtype=torch.long).long(),
        context_mask=_pad_1d(ctx_mask),
        target_x=_pad_1d(tgt_x),
        target_y=_pad_1d(tgt_y),
        target_yerr=_pad_1d(tgt_yerr, pad_value=1.0),
        target_band=_pad_1d(tgt_band, pad_value=0, dtype=torch.long).long(),
        target_mask=_pad_1d(tgt_mask),
        labels=torch.as_tensor(labels, dtype=torch.float32),
        redshift=normalize_redshift(
            torch.as_tensor(redshifts, dtype=torch.float32), z_min, z_max
        ),
        t_span_log=torch.as_tensor(t_span_logs, dtype=torch.float32),
        n_obs_log=torch.as_tensor(n_obs_logs, dtype=torch.float32),
        meta_values=_pad_1d(meta_values),
        meta_mask=_pad_1d(meta_masks),
        object_ids=object_ids,
    )


class ConvGNPBaseline(nn.Module):
    def __init__(self, cfg: ConvGNPBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = FourierTimeEmbedding(cfg.time_fourier_dim)
        self.band_embed = nn.Embedding(cfg.num_bands, cfg.band_emb_dim)
        point_in = (
            1
            + 1
            + cfg.band_emb_dim
            + 2 * cfg.time_fourier_dim
            + (1 if cfg.use_redshift else 0)
        )
        self.point_encoder = MLP(
            point_in, cfg.point_feat_dim, cfg.point_feat_dim, depth=3
        )
        self.setconv = GaussianSetConv1D(tuple(cfg.setconv_sigmas))
        n_scales = len(cfg.setconv_sigmas)
        self.grid_proj = nn.Conv1d(
            n_scales * (cfg.point_feat_dim + 1), cfg.grid_feat_dim, kernel_size=1
        )
        self.grid_backbone = ConvBackbone1D(
            cfg.grid_feat_dim,
            layers=cfg.conv_layers,
            kernel_size=cfg.conv_kernel_size,
            dropout=cfg.conv_dropout,
        )
        self.latent_encoder: GlobalLatentEncoder | None = None
        self.cross_attn_proj: nn.Linear | None = None
        latent_dim = 0
        if cfg.use_latent:
            self.latent_encoder = GlobalLatentEncoder(
                cfg.point_feat_dim, cfg.latent_hidden_dim, cfg.latent_dim
            )
            self.cross_attn_proj = nn.Linear(cfg.latent_dim, cfg.grid_feat_dim)
            latent_dim = cfg.latent_dim
        decoder_in = (
            cfg.grid_feat_dim
            + cfg.band_emb_dim
            + 2 * cfg.time_fourier_dim
            + (1 if cfg.use_redshift else 0)
            + latent_dim
        )
        self.decoder = MLP(decoder_in, cfg.decoder_hidden_dim, 2, depth=3)
        self.meta_encoder = None
        meta_head = 0
        if cfg.use_metadata and cfg.metadata_dim > 0:
            self.meta_encoder = MLP(
                2 * cfg.metadata_dim,
                cfg.metadata_hidden_dim,
                cfg.metadata_embed_dim,
                depth=3,
                dropout=0.1,
            )
            meta_head = cfg.metadata_embed_dim
        cross_attn_dim = cfg.grid_feat_dim if cfg.use_latent else 0
        head_in = (
            2 * cfg.grid_feat_dim
            + cross_attn_dim
            + (1 if cfg.use_redshift else 0)
            + meta_head
            + 2
            + 2 * latent_dim
        )  # mean+max+attended; +2: t_span_log, n_obs_log; +2*latent_dim: mu+logvar
        self.classifier = MLP(
            head_in, cfg.classifier_hidden_dim, cfg.num_classes, depth=3, dropout=0.1
        )
        self.register_buffer(
            "grid_x", torch.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_size)
        )

    def _encode_points(self, batch: BaselineBatch) -> torch.Tensor:
        tfeat = self.time_embed(batch.context_x)
        bfeat = self.band_embed(batch.context_band)
        pieces = [
            batch.context_y.unsqueeze(-1),
            torch.log(batch.context_yerr.clamp_min(1e-6)).unsqueeze(-1),
            bfeat,
            tfeat,
        ]
        if self.cfg.use_redshift:
            pieces.append(
                batch.redshift.unsqueeze(-1).expand_as(batch.context_x).unsqueeze(-1)
            )
        feat = torch.cat(pieces, dim=-1)
        return self.point_encoder(feat) * batch.context_mask.unsqueeze(-1)

    def _interpolate_grid(
        self, grid_features: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        channels = grid_features.shape[1]
        gsz = grid_features.shape[2]
        pos = (
            (xq.clamp(self.cfg.grid_min, self.cfg.grid_max) - self.cfg.grid_min)
            / (self.cfg.grid_max - self.cfg.grid_min)
            * (gsz - 1)
        )
        left = pos.floor().long().clamp(0, gsz - 1)
        right = (left + 1).clamp(0, gsz - 1)
        alpha = (pos - left.float()).unsqueeze(-1)
        gf = grid_features.transpose(1, 2)
        left_feat = torch.gather(gf, 1, left.unsqueeze(-1).expand(-1, -1, channels))
        right_feat = torch.gather(gf, 1, right.unsqueeze(-1).expand(-1, -1, channels))
        return left_feat * (1.0 - alpha) + right_feat * alpha

    def forward(self, batch: BaselineBatch) -> ConvGNPBaselineOutput:
        point_feat = self._encode_points(batch)
        agg, density = self.setconv(
            batch.context_x, point_feat, batch.context_mask, self.grid_x
        )
        # agg: (B, G, S*C), density: (B, G, S) — concatenate then transpose to (B, S*(C+1), G)
        grid_in = torch.cat([agg, density], dim=-1).transpose(1, 2)
        grid_features = self.grid_backbone(self.grid_proj(grid_in))

        latent_mu: torch.Tensor | None = None
        latent_logvar: torch.Tensor | None = None
        z: torch.Tensor | None = None

        if self.latent_encoder is not None:
            latent_mu, latent_logvar = self.latent_encoder(
                point_feat, batch.context_mask
            )
            if self.training:
                eps = torch.randn_like(latent_mu)
                z = latent_mu + eps * (0.5 * latent_logvar).exp()
            else:
                z = latent_mu

        query_feat = self._interpolate_grid(grid_features, batch.target_x)
        tfeat = self.time_embed(batch.target_x)
        bfeat = self.band_embed(batch.target_band)
        dec_parts = [query_feat, bfeat, tfeat]

        if self.cfg.use_redshift:
            dec_parts.append(
                batch.redshift.unsqueeze(-1).expand_as(batch.target_x).unsqueeze(-1)
            )

        if z is not None:
            dec_parts.append(z.unsqueeze(1).expand(-1, batch.target_x.shape[1], -1))

        dec = self.decoder(torch.cat(dec_parts, dim=-1))
        pred_mean = dec[..., 0]
        pred_var = (F.softplus(dec[..., 1]) + self.cfg.min_std).square()

        gf = grid_features.transpose(1, 2)  # (B, G, C)
        pooled = [gf.mean(dim=1), gf.max(dim=1).values]

        if self.cross_attn_proj is not None and latent_mu is not None:
            q = self.cross_attn_proj(latent_mu)  # (B, C)
            scores = torch.einsum("bc,bgc->bg", q, gf) / (gf.shape[-1] ** 0.5)
            attended = torch.einsum(
                "bg,bgc->bc", torch.softmax(scores, dim=-1), gf
            )  # (B, C)
            pooled.append(attended)

        if self.cfg.use_redshift:
            pooled.append(batch.redshift.unsqueeze(-1))

        pooled.extend([batch.t_span_log.unsqueeze(-1), batch.n_obs_log.unsqueeze(-1)])

        if latent_mu is not None and latent_logvar is not None:
            pooled.extend([latent_mu, latent_logvar])

        if self.meta_encoder is not None:
            pooled.append(
                self.meta_encoder(
                    torch.cat([batch.meta_values, batch.meta_mask], dim=-1)
                )
            )
        class_logits = self.classifier(torch.cat(pooled, dim=-1))

        if self.cfg.num_classes == 1:
            class_logits = class_logits.squeeze(-1)

        return ConvGNPBaselineOutput(
            pred_mean=pred_mean,
            pred_var=pred_var,
            class_logits=class_logits,
            grid_features=grid_features,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
        )


def baseline_loss(
    output: ConvGNPBaselineOutput, batch: BaselineBatch, cfg: BaselineLossConfig
) -> BaselineLosses:
    mask = batch.target_mask
    var = (output.pred_var + batch.target_yerr.square()).clamp_min(1e-6)
    sq = (batch.target_y - output.pred_mean).square()
    recon_terms = 0.5 * (sq / var + torch.log(var))
    recon = (recon_terms * mask).sum() / mask.sum().clamp_min(1.0)
    is_binary = output.class_logits.dim() == 1 or output.class_logits.shape[-1] == 1
    if is_binary:
        logits_1d = (
            output.class_logits.squeeze(-1)
            if output.class_logits.dim() > 1
            else output.class_logits
        )
        pos_weight = (
            None
            if cfg.pos_weight is None
            else torch.tensor(cfg.pos_weight, device=batch.labels.device)
        )
        cls = F.binary_cross_entropy_with_logits(
            logits_1d, batch.labels, pos_weight=pos_weight
        )
    else:
        weight = None
        if cfg.class_weights is not None:
            weight = torch.tensor(
                cfg.class_weights, dtype=torch.float32, device=batch.labels.device
            )
        cls = F.cross_entropy(output.class_logits, batch.labels.long(), weight=weight)
    kl: torch.Tensor | None = None
    total = cfg.lambda_recon * recon + cfg.lambda_cls * cls
    if output.latent_mu is not None and output.latent_logvar is not None:
        latent_dim = output.latent_mu.shape[-1]
        kl = (
            -0.5
            * (
                1.0
                + output.latent_logvar
                - output.latent_mu.square()
                - output.latent_logvar.exp()
            )
            .sum(-1)
            .mean()
            / latent_dim
        )
        total = total + cfg.beta_kl * kl
    return BaselineLosses(total=total, recon=recon, cls=cls, kl=kl)


def evaluate_binary_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    pred05 = (probs >= 0.5).astype(int)
    metrics = {
        "f1@0.5": float(f1_score(y, pred05, zero_division=0)),
        "precision@0.5": float(precision_score(y, pred05, zero_division=0)),
        "recall@0.5": float(recall_score(y, pred05, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y, probs))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["ap"] = float(average_precision_score(y, probs))
    except Exception:
        metrics["ap"] = float("nan")
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 197):
        pred = (probs >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    metrics["best_f1"] = best_f1
    metrics["best_threshold"] = best_threshold
    return metrics


def evaluate_multiclass_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    preds = probs.argmax(axis=-1)
    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
    }
    try:
        n_cls = probs.shape[-1]
        metrics["macro_auroc"] = float(
            roc_auc_score(
                y, probs, multi_class="ovr", average="macro", labels=list(range(n_cls))
            )
        )
    except Exception:
        metrics["macro_auroc"] = float("nan")
    try:
        from sklearn.metrics import top_k_accuracy_score

        metrics["top1_acc"] = float(top_k_accuracy_score(y, probs, k=1))
    except Exception:
        metrics["top1_acc"] = float((preds == y).mean())
    return metrics


def fit_epoch(
    model: ConvGNPBaseline,
    loader,
    optimizer,
    loss_cfg: BaselineLossConfig,
    device: str | torch.device,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    nan_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = baseline_loss(output, batch, loss_cfg)
        if not torch.isfinite(losses.total):
            nan_batches += 1
            if nan_batches == 1:
                print(
                    f"  [debug] first training NaN batch: {_summarize_bad_batch(batch, output, losses)}"
                )
            continue
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, float] = {
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
        }
        if losses.kl is not None:
            row["kl"] = float(losses.kl.item())
        rows.append(row)
    result = (
        {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} if rows else {}
    )
    if nan_batches > 0:
        result["nan_batches"] = float(nan_batches)
    return result


def _batch_is_finite(output: ConvGNPBaselineOutput, losses: BaselineLosses) -> bool:
    tensors = [
        output.pred_mean,
        output.pred_var,
        output.class_logits,
        losses.total,
        losses.recon,
        losses.cls,
    ]
    if output.latent_mu is not None:
        tensors.append(output.latent_mu)
    if output.latent_logvar is not None:
        tensors.append(output.latent_logvar)
    if losses.kl is not None:
        tensors.append(losses.kl)
    return all(torch.isfinite(tensor).all() for tensor in tensors)


def _tensor_range(x: torch.Tensor) -> str:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return "all_nonfinite"
    return f"{finite.min().item():.3g}..{finite.max().item():.3g}"


def _summarize_bad_batch(
    batch: BaselineBatch, output: ConvGNPBaselineOutput, losses: BaselineLosses
) -> str:
    logits = (
        output.class_logits
        if output.class_logits.dim() > 0
        else output.class_logits.unsqueeze(0)
    )
    parts = [
        f"oids={batch.object_ids[:4]}{'...' if len(batch.object_ids) > 4 else ''}",
        f"labels={batch.labels.detach().cpu().numpy().astype(int).tolist()[:8]}",
        f"ctx_x={_tensor_range(batch.context_x)}",
        f"ctx_y={_tensor_range(batch.context_y)}",
        f"ctx_yerr={_tensor_range(batch.context_yerr)}",
        f"tgt_y={_tensor_range(batch.target_y)}",
        f"meta={_tensor_range(batch.meta_values)}",
        f"meta_mask_sum={batch.meta_mask.sum().item():.1f}",
        f"pred_mean={_tensor_range(output.pred_mean)}",
        f"pred_var={_tensor_range(output.pred_var)}",
        f"logits={_tensor_range(logits)}",
        f"loss_total={losses.total.detach().item() if torch.isfinite(losses.total) else 'nonfinite'}",
        f"loss_recon={losses.recon.detach().item() if torch.isfinite(losses.recon) else 'nonfinite'}",
        f"loss_cls={losses.cls.detach().item() if torch.isfinite(losses.cls) else 'nonfinite'}",
    ]
    if losses.kl is not None:
        parts.append(
            f"loss_kl={losses.kl.detach().item() if torch.isfinite(losses.kl) else 'nonfinite'}"
        )
    return " | ".join(parts)


@torch.no_grad()
def evaluate_epoch(
    model: ConvGNPBaseline,
    loader,
    loss_cfg: BaselineLossConfig,
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    loss_rows: list[dict[str, float]] = []
    logits, labels = [], []
    nan_batches = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        losses = baseline_loss(output, batch, loss_cfg)
        if not _batch_is_finite(output, losses):
            nan_batches += 1
            if nan_batches == 1:
                print(
                    f"  [debug] first validation NaN batch: {_summarize_bad_batch(batch, output, losses)}"
                )
            continue
        loss_row: dict[str, float] = {
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
        }
        if losses.kl is not None:
            loss_row["kl"] = float(losses.kl.item())
        loss_rows.append(loss_row)
        logits.append(output.class_logits.detach().cpu())
        labels.append(batch.labels.detach().cpu())
    metrics = (
        {k: float(np.mean([row[k] for row in loss_rows])) for k in loss_rows[0]}
        if loss_rows
        else {}
    )
    if nan_batches > 0:
        metrics["nan_batches"] = float(nan_batches)
    if logits:
        all_logits = torch.cat(logits)
        all_labels = torch.cat(labels)
        if all_logits.dim() == 1 or all_logits.shape[-1] == 1:
            metrics.update(evaluate_binary_predictions(all_logits, all_labels))
        else:
            metrics.update(evaluate_multiclass_predictions(all_logits, all_labels))
    return metrics


@torch.no_grad()
def collect_epoch_predictions(
    model: ConvGNPBaseline,
    loader,
    loss_cfg: BaselineLossConfig,
    device: str | torch.device,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    model.eval()
    loss_rows: list[dict[str, float]] = []
    logits, labels = [], []
    rows: list[dict[str, float | str | int]] = []
    nan_batches = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        losses = baseline_loss(output, batch, loss_cfg)
        if not _batch_is_finite(output, losses):
            nan_batches += 1
            if nan_batches == 1:
                print(
                    f"  [debug] first prediction NaN batch: {_summarize_bad_batch(batch, output, losses)}"
                )
            continue
        ep_row: dict[str, float] = {
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
        }
        if losses.kl is not None:
            ep_row["kl"] = float(losses.kl.item())
        loss_rows.append(ep_row)
        batch_logits = output.class_logits.detach().cpu()
        batch_labels = batch.labels.detach().cpu()
        logits.append(batch_logits)
        labels.append(batch_labels)
        is_binary = batch_logits.dim() == 1 or batch_logits.shape[-1] == 1
        if is_binary:
            batch_probs = torch.sigmoid(
                batch_logits.squeeze(-1) if batch_logits.dim() > 1 else batch_logits
            ).numpy()
            for oid, label, prob in zip(
                batch.object_ids, batch_labels.numpy().astype(int), batch_probs
            ):
                rows.append(
                    {
                        "object_id": str(oid),
                        "target": int(label),
                        "prob_tde": float(prob),
                    }
                )
        else:
            batch_probs = torch.softmax(batch_logits, dim=-1).numpy()
            for oid, label, probs_row in zip(
                batch.object_ids, batch_labels.numpy().astype(int), batch_probs
            ):
                row: dict[str, float | str | int] = {
                    "object_id": str(oid),
                    "target": int(label),
                }
                for k, p in enumerate(probs_row):
                    row[f"prob_class_{k}"] = float(p)
                rows.append(row)
    metrics = (
        {k: float(np.mean([row[k] for row in loss_rows])) for k in loss_rows[0]}
        if loss_rows
        else {}
    )
    if nan_batches > 0:
        metrics["nan_batches"] = float(nan_batches)
    if logits:
        all_logits = torch.cat(logits)
        all_labels = torch.cat(labels)
        if all_logits.dim() == 1 or all_logits.shape[-1] == 1:
            metrics.update(evaluate_binary_predictions(all_logits, all_labels))
        else:
            metrics.update(evaluate_multiclass_predictions(all_logits, all_labels))
    return metrics, rows


@torch.no_grad()
def collect_full_context_predictions(
    model: ConvGNPBaseline,
    dataset: Dataset,
    *,
    batch_size: int,
    z_min: float,
    z_max: float,
    device: str | torch.device,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    model.eval()
    logits, labels = [], []
    rows: list[dict[str, float | str | int]] = []
    nan_batches = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            items = [dataset[i] for i in range(start, stop)]
            batch = make_full_context_batch(items, z_min=z_min, z_max=z_max).to(device)
            output = model(batch)
            if not (
                torch.isfinite(output.pred_mean).all()
                and torch.isfinite(output.pred_var).all()
                and torch.isfinite(output.class_logits).all()
            ):
                nan_batches += 1
                continue
            batch_logits = output.class_logits.detach().cpu()
            batch_labels = batch.labels.detach().cpu()
            logits.append(batch_logits)
            labels.append(batch_labels)
            is_binary = batch_logits.dim() == 1 or batch_logits.shape[-1] == 1
            if is_binary:
                batch_probs = torch.sigmoid(
                    batch_logits.squeeze(-1) if batch_logits.dim() > 1 else batch_logits
                ).numpy()
                for oid, label, prob in zip(
                    batch.object_ids, batch_labels.numpy().astype(int), batch_probs
                ):
                    rows.append(
                        {
                            "object_id": str(oid),
                            "target": int(label),
                            "prob_tde": float(prob),
                        }
                    )
            else:
                batch_probs = torch.softmax(batch_logits, dim=-1).numpy()
                for oid, label, probs_row in zip(
                    batch.object_ids, batch_labels.numpy().astype(int), batch_probs
                ):
                    row: dict[str, float | str | int] = {
                        "object_id": str(oid),
                        "target": int(label),
                    }
                    for k, p in enumerate(probs_row):
                        row[f"prob_class_{k}"] = float(p)
                    rows.append(row)
    metrics: dict[str, float] = {}
    if nan_batches > 0:
        metrics["nan_batches"] = float(nan_batches)
    if logits:
        all_logits = torch.cat(logits)
        all_labels = torch.cat(labels)
        if all_logits.dim() == 1 or all_logits.shape[-1] == 1:
            metrics.update(evaluate_binary_predictions(all_logits, all_labels))
        else:
            metrics.update(evaluate_multiclass_predictions(all_logits, all_labels))
    return metrics, rows


def save_baseline_checkpoint(
    path: str | Path,
    *,
    model: ConvGNPBaseline,
    model_cfg: ConvGNPBaselineConfig,
    loss_cfg: BaselineLossConfig,
    epoch: int,
    metrics: dict[str, float],
    history: list[dict],
    z_min: float,
    z_max: float,
    flux_center_by_band: np.ndarray,
    flux_scale_by_band: np.ndarray,
    extra_payload: dict | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "loss_cfg": asdict(loss_cfg),
        "metrics": metrics,
        "history": history,
        "z_min": z_min,
        "z_max": z_max,
        "flux_center_by_band": np.asarray(
            flux_center_by_band, dtype=np.float32
        ).tolist(),
        "flux_scale_by_band": np.asarray(flux_scale_by_band, dtype=np.float32).tolist(),
    }
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, path)


def load_baseline_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ConvGNPBaseline, dict]:
    ckpt = torch.load(path, map_location=device)
    model = ConvGNPBaseline(ConvGNPBaselineConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_full_context(
    model: ConvGNPBaseline,
    dataset: MallornBaselineDataset,
    *,
    batch_size: int,
    z_min: float,
    z_max: float,
    device: str | torch.device,
) -> list[dict[str, float | str]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    object_ids = list(dataset.object_ids)
    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start : start + batch_size]
        items = []
        for oid in batch_ids:
            item = dataset.data[oid].copy()
            item["oid"] = oid
            items.append(item)
        batch = make_full_context_batch(items, z_min=z_min, z_max=z_max).to(device)
        output = model(batch)
        probs = torch.sigmoid(output.class_logits).detach().cpu().numpy()
        for oid, prob in zip(batch.object_ids, probs):
            rows.append({"object_id": oid, "prob_tde": float(prob)})
    return rows
