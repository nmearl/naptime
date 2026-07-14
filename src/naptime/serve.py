"""FastAPI inference service exposing /classify and /health endpoints."""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from .baseline import (
    MAX_ABS_FLUX_NORM,
    MAX_FERR_NORM,
    MIN_FLUX_SCALE,
    make_full_context_batch,
    load_baseline_checkpoint,
)
from .data import BAND2IDX, BANDS


def _resolve_checkpoint_path() -> Path:
    configured = os.environ.get("NAPTIME_CHECKPOINT")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("/models/latest.pt"))
    candidates.append(Path(__file__).resolve().parent / "models" / "latest.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in candidates)
    raise RuntimeError(
        "No NAPTIME checkpoint found. Set NAPTIME_CHECKPOINT or mount a model at "
        f"/models/latest.pt. Checked: {tried}"
    )


def _select_device() -> torch.device:
    requested = os.environ.get("NAPTIME_DEVICE", "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NAPTIME_DEVICE=cuda requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("NAPTIME_DEVICE=mps requested, but MPS is unavailable")
    return device


def _checkpoint_meta_stats(
    ckpt: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "meta_center" in ckpt and "meta_scale" in ckpt:
        return (
            np.asarray(ckpt["meta_center"], dtype=np.float32),
            np.asarray(ckpt["meta_scale"], dtype=np.float32),
        )
    metadata_dim = int(ckpt.get("model_cfg", {}).get("metadata_dim", 0))
    if metadata_dim <= 0:
        return None, None
    return (
        np.zeros(metadata_dim, dtype=np.float32),
        np.ones(metadata_dim, dtype=np.float32),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = NaptimeInferenceService.from_environment()
    yield
    app.state.inference = None


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

VALID_BANDS = set(BANDS)


class Detection(BaseModel):
    mjd: float
    flux: float
    flux_err: float
    band: str = Field(..., description="One of: u, g, r, i, z, y")

    @model_validator(mode="after")
    def check_band(self) -> "Detection":
        if self.band not in VALID_BANDS:
            raise ValueError(f"band must be one of {sorted(VALID_BANDS)}")
        return self


class HostMetadata(BaseModel):
    mwebv: float | None = Field(None, description="Milky Way E(B-V)")
    host_snsep: float | None = Field(None, description="Host separation (arcsec)")
    host_ddlr: float | None = Field(None, description="Directional light radius offset")
    host_logmass: float | None = Field(None, description="Host log stellar mass")
    host_ellipticity: float | None = Field(None, description="Host ellipticity")
    host_mag_g: float | None = Field(None, description="Host magnitude g-band")
    host_mag_r: float | None = Field(None, description="Host magnitude r-band")
    host_mag_i: float | None = Field(None, description="Host magnitude i-band")
    host_mag_z: float | None = Field(None, description="Host magnitude z-band")
    host_mag_y: float | None = Field(None, description="Host magnitude Y-band")


class ClassifyRequest(BaseModel):
    object_id: str = Field(default="unnamed")
    detections: list[Detection] = Field(..., min_length=4)
    redshift: float | None = Field(None, description="Host photometric redshift")
    metadata: HostMetadata | None = None


class ClassifyResponse(BaseModel):
    object_id: str
    class_probs: dict[str, float]
    tde_score: float | None
    pred_class: str | None
    n_detections: int
    taxonomy: str
    model_version: str


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Order must match elasticc.META_FIELDS
_META_FIELD_ORDER = [
    "mwebv",
    "host_snsep",
    "host_ddlr",
    "host_logmass",
    "host_ellipticity",
    "host_mag_g",
    "host_mag_r",
    "host_mag_i",
    "host_mag_z",
    "host_mag_y",
]


@dataclass(slots=True)
class NaptimeInferenceService:
    model: Any
    ckpt: dict[str, Any]
    device: torch.device
    checkpoint_path: Path
    flux_center: np.ndarray
    flux_scale: np.ndarray
    meta_center: np.ndarray | None
    meta_scale: np.ndarray | None
    z_min: float
    z_max: float
    class_names: list[str]
    taxonomy: str
    model_version: str

    @classmethod
    def from_environment(cls) -> "NaptimeInferenceService":
        ckpt_path = _resolve_checkpoint_path()
        device = _select_device()
        model, ckpt = load_baseline_checkpoint(ckpt_path, device=device)
        model.eval()
        meta_center, meta_scale = _checkpoint_meta_stats(ckpt)
        return cls(
            model=model,
            ckpt=ckpt,
            device=device,
            checkpoint_path=ckpt_path,
            flux_center=np.asarray(ckpt["flux_center_by_band"], dtype=np.float32),
            flux_scale=np.asarray(ckpt["flux_scale_by_band"], dtype=np.float32),
            meta_center=meta_center,
            meta_scale=meta_scale,
            z_min=float(ckpt["z_min"]),
            z_max=float(ckpt["z_max"]),
            class_names=list(ckpt.get("class_names", [])),
            taxonomy=str(ckpt.get("elasticc_taxonomy", "unknown")),
            model_version=str(ckpt.get("model_version", ckpt_path.stem)),
        )

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "taxonomy": self.taxonomy,
            "model_version": self.model_version,
        }

    def _preprocess(self, req: ClassifyRequest) -> dict:
        dets = sorted(req.detections, key=lambda d: d.mjd)
        t_raw = np.array([d.mjd for d in dets], dtype=np.float32)
        flux_raw = np.array([d.flux for d in dets], dtype=np.float32)
        ferr_raw = np.array([d.flux_err for d in dets], dtype=np.float32)
        band_idx = np.array([BAND2IDX[d.band] for d in dets], dtype=np.int64)

        valid = np.isfinite(flux_raw) & np.isfinite(ferr_raw) & (ferr_raw > 0)
        if valid.sum() < 4:
            raise HTTPException(
                status_code=422,
                detail="Fewer than 4 valid detections after removing non-finite values.",
            )
        t_raw, flux_raw, ferr_raw, band_idx = (
            t_raw[valid],
            flux_raw[valid],
            ferr_raw[valid],
            band_idx[valid],
        )

        t_span = max(float(t_raw[-1] - t_raw[0]), 1.0)
        t_norm = (t_raw - t_raw[0]) / t_span

        centers = self.flux_center[band_idx]
        scales = np.maximum(self.flux_scale[band_idx], MIN_FLUX_SCALE)
        flux_norm = np.clip(
            (flux_raw - centers) / scales, -MAX_ABS_FLUX_NORM, MAX_ABS_FLUX_NORM
        )
        ferr_norm = np.clip(ferr_raw / scales, 1e-3, MAX_FERR_NORM)

        meta_values, meta_mask = self._preprocess_metadata(req.metadata)

        return {
            "oid": req.object_id,
            "t_norm": t_norm.astype(np.float32),
            "flux_norm": flux_norm.astype(np.float32),
            "ferr_norm": ferr_norm.astype(np.float32),
            "band_idx": band_idx.astype(np.int64),
            "z": float(req.redshift) if req.redshift is not None else float("nan"),
            "t_series_span": float(t_span),
            "n_obs": int(len(t_raw)),
            "meta_values": meta_values,
            "meta_mask": meta_mask,
        }

    def _preprocess_metadata(
        self, metadata: HostMetadata | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.meta_center is None or metadata is None:
            n_fields = len(self.meta_center) if self.meta_center is not None else 0
            return (
                np.zeros(n_fields, dtype=np.float32),
                np.zeros(n_fields, dtype=np.float32),
            )
        raw_meta = metadata.model_dump()
        vals = np.array(
            [raw_meta.get(field, np.nan) for field in _META_FIELD_ORDER],
            dtype=np.float32,
        )
        mask = np.isfinite(vals).astype(np.float32)
        vals = np.where(mask > 0, vals, 0.0)
        safe_scale = np.where(self.meta_scale > 1e-6, self.meta_scale, 1.0)
        vals = np.clip((vals - self.meta_center) / safe_scale, -10.0, 10.0)
        return vals.astype(np.float32), mask

    @torch.no_grad()
    def classify(self, req: ClassifyRequest) -> ClassifyResponse:
        item = self._preprocess(req)
        batch = make_full_context_batch(
            [item], z_min=self.z_min, z_max=self.z_max
        ).to(self.device)
        output = self.model(batch)
        logits = output.class_logits[0]

        if logits.dim() == 0 or logits.shape[-1] == 1:
            score = float(torch.sigmoid(logits.squeeze()).item())
            class_probs = {"TDE": score, "non-TDE": 1.0 - score}
            tde_score: float | None = score
            pred_class: str | None = "TDE" if score >= 0.5 else "non-TDE"
        else:
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            if self.class_names:
                class_probs = {
                    name: float(prob)
                    for name, prob in zip(self.class_names, probs, strict=False)
                }
                tde_idx = next(
                    (
                        i
                        for i, name in enumerate(self.class_names)
                        if "tde" in name.lower()
                    ),
                    None,
                )
                tde_score = float(probs[tde_idx]) if tde_idx is not None else None
                pred_class = self.class_names[int(np.argmax(probs))]
            else:
                class_probs = {str(i): float(prob) for i, prob in enumerate(probs)}
                tde_score = None
                pred_class = str(int(np.argmax(probs)))

        return ClassifyResponse(
            object_id=req.object_id,
            class_probs=class_probs,
            tde_score=tde_score,
            pred_class=pred_class,
            n_detections=item["n_obs"],
            taxonomy=self.taxonomy,
            model_version=self.model_version,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="NAPTIME", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    service: NaptimeInferenceService | None = request.app.state.inference
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not loaded")
    return service.health()


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest, request: Request) -> ClassifyResponse:
    service: NaptimeInferenceService | None = request.app.state.inference
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not loaded")
    return service.classify(req)
