from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
import time

from astropy.io import fits
import numpy as np

from .data import BAND2IDX

# Focused confuser taxonomy used throughout the current paper.
FOCUSED_CLASS_NAMES: list[str] = ["TDE", "AGN-like", "SLSN", "PISN", "SN-Ia", "SN-II", "SN-Ibc"]
FOCUSED_RELEASE_TO_CLASS: dict[str, int] = {
    # ELAsTiCC1
    "ELASTICC_TRAIN_TDE": 0,
    "ELASTICC_TRAIN_AGN": 1,
    "ELASTICC_TRAIN_SLSN-I+host": 2,
    "ELASTICC_TRAIN_SLSN-I_no_host": 2,
    "ELASTICC_TRAIN_PISN": 3,
    "ELASTICC_TRAIN_SNIa-SALT2": 4,
    "ELASTICC_TRAIN_SNIa-91bg": 4,
    "ELASTICC_TRAIN_SNIax": 4,
    "ELASTICC_TRAIN_SNII+HostXT_V19": 5,
    "ELASTICC_TRAIN_SNII-NMF": 5,
    "ELASTICC_TRAIN_SNII-Templates": 5,
    "ELASTICC_TRAIN_SNIIn+HostXT_V19": 5,
    "ELASTICC_TRAIN_SNIIn-MOSFIT": 5,
    "ELASTICC_TRAIN_SNIIb+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIb+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIb-Templates": 6,
    "ELASTICC_TRAIN_SNIc+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIc-Templates": 6,
    "ELASTICC_TRAIN_SNIcBL+HostXT_V19": 6,
    # ELAsTiCC2
    "ELASTICC2_TRAIN_02_TDE": 0,
    "ELASTICC2_TRAIN_02_CLAGN": 1,
    "ELASTICC2_TRAIN_02_SLSN-I+host": 2,
    "ELASTICC2_TRAIN_02_SLSN-I_no_host": 2,
    "ELASTICC2_TRAIN_02_PISN": 3,
    "ELASTICC2_TRAIN_02_SNIa-SALT3": 4,
    "ELASTICC2_TRAIN_02_SNIa-91bg": 4,
    "ELASTICC2_TRAIN_02_SNIax": 4,
    "ELASTICC2_TRAIN_02_SNII+HostXT_V19": 5,
    "ELASTICC2_TRAIN_02_SNII-NMF": 5,
    "ELASTICC2_TRAIN_02_SNII-Templates": 5,
    "ELASTICC2_TRAIN_02_SNIIn+HostXT_V19": 5,
    "ELASTICC2_TRAIN_02_SNIIn-MOSFIT": 5,
    "ELASTICC2_TRAIN_02_SNIIb+HostXT_V19": 6,
    "ELASTICC2_TRAIN_02_SNIb+HostXT_V19": 6,
    "ELASTICC2_TRAIN_02_SNIb-Templates": 6,
    "ELASTICC2_TRAIN_02_SNIc+HostXT_V19": 6,
    "ELASTICC2_TRAIN_02_SNIc-Templates": 6,
    "ELASTICC2_TRAIN_02_SNIcBL+HostXT_V19": 6,
}

# Broader family-level taxonomy for broker-style ELAsTiCC2 tests.
FAMILY_CLASS_NAMES: list[str] = [
    "TDE",
    "AGN-like",
    "Ca-rich/fast",
    "ILOT",
    "Kilonova",
    "SLSN",
    "PISN",
    "SN-Ia",
    "SN-II",
    "SN-Ibc",
    "M-dwarf flare",
    "Pulsator",
    "Eclipsing binary",
    "Dwarf nova",
    "Microlensing",
]
FAMILY_RELEASE_TO_CLASS: dict[str, int] = {
    "ELASTICC2_TRAIN_02_TDE": 0,
    "ELASTICC2_TRAIN_02_CLAGN": 1,
    "ELASTICC2_TRAIN_02_CART": 2,
    "ELASTICC2_TRAIN_02_ILOT": 3,
    "ELASTICC2_TRAIN_02_KN_B19": 4,
    "ELASTICC2_TRAIN_02_KN_K17": 4,
    "ELASTICC2_TRAIN_02_SLSN-I+host": 5,
    "ELASTICC2_TRAIN_02_SLSN-I_no_host": 5,
    "ELASTICC2_TRAIN_02_PISN": 6,
    "ELASTICC2_TRAIN_02_SNIa-SALT3": 7,
    "ELASTICC2_TRAIN_02_SNIa-91bg": 7,
    "ELASTICC2_TRAIN_02_SNIax": 7,
    "ELASTICC2_TRAIN_02_SNII+HostXT_V19": 8,
    "ELASTICC2_TRAIN_02_SNII-NMF": 8,
    "ELASTICC2_TRAIN_02_SNII-Templates": 8,
    "ELASTICC2_TRAIN_02_SNIIn+HostXT_V19": 8,
    "ELASTICC2_TRAIN_02_SNIIn-MOSFIT": 8,
    "ELASTICC2_TRAIN_02_SNIIb+HostXT_V19": 9,
    "ELASTICC2_TRAIN_02_SNIb+HostXT_V19": 9,
    "ELASTICC2_TRAIN_02_SNIb-Templates": 9,
    "ELASTICC2_TRAIN_02_SNIc+HostXT_V19": 9,
    "ELASTICC2_TRAIN_02_SNIc-Templates": 9,
    "ELASTICC2_TRAIN_02_SNIcBL+HostXT_V19": 9,
    "ELASTICC2_TRAIN_02_Mdwarf-flare": 10,
    "ELASTICC2_TRAIN_02_Cepheid": 11,
    "ELASTICC2_TRAIN_02_RRL": 11,
    "ELASTICC2_TRAIN_02_d-Sct": 11,
    "ELASTICC2_TRAIN_02_EB": 12,
    "ELASTICC2_TRAIN_02_dwarf-nova": 13,
    "ELASTICC2_TRAIN_02_uLens-Binary": 14,
    "ELASTICC2_TRAIN_02_uLens-Single-GenLens": 14,
    "ELASTICC2_TRAIN_02_uLens-Single_PyLIMA": 14,
}

ELASTICC_TAXONOMIES: dict[str, tuple[list[str], dict[str, int]]] = {
    "focused": (FOCUSED_CLASS_NAMES, FOCUSED_RELEASE_TO_CLASS),
    "families": (FAMILY_CLASS_NAMES, FAMILY_RELEASE_TO_CLASS),
}
ELASTICC_TAXONOMY_NAMES: tuple[str, ...] = tuple(ELASTICC_TAXONOMIES.keys())

# Backwards-compatible aliases for the focused configuration.
CLASS_NAMES: list[str] = FOCUSED_CLASS_NAMES
NUM_ELASTICC_CLASSES: int = len(CLASS_NAMES)
RELEASE_TO_CLASS: dict[str, int] = FOCUSED_RELEASE_TO_CLASS

ELASTICC_REDSHIFT_SOURCES: tuple[str, ...] = ("final", "photoz", "none")
CACHE_VERSION = 1
LEGACY_META_FIELDS = [
    "MWEBV",
    "REDSHIFT_FINAL",
    "HOSTGAL_PHOTOZ",
    "HOSTGAL_SNSEP",
    "HOSTGAL_DDLR",
    "HOSTGAL_LOGMASS",
    "HOSTGAL_ELLIPTICITY",
    "HOSTGAL_MAG_g",
    "HOSTGAL_MAG_r",
    "HOSTGAL_MAG_i",
    "HOSTGAL_MAG_z",
    "HOSTGAL_MAG_Y",
]
META_FIELDS = [
    "MWEBV",
    "HOSTGAL_SNSEP",
    "HOSTGAL_DDLR",
    "HOSTGAL_LOGMASS",
    "HOSTGAL_ELLIPTICITY",
    "HOSTGAL_MAG_g",
    "HOSTGAL_MAG_r",
    "HOSTGAL_MAG_i",
    "HOSTGAL_MAG_z",
    "HOSTGAL_MAG_Y",
]


def get_elasticc_taxonomy(name: str | None = None) -> tuple[str, list[str], dict[str, int]]:
    taxonomy_name = name or "focused"
    if taxonomy_name not in ELASTICC_TAXONOMIES:
        raise ValueError(f"unknown ELAsTiCC taxonomy: {taxonomy_name}")
    class_names, release_to_class = ELASTICC_TAXONOMIES[taxonomy_name]
    return taxonomy_name, class_names, release_to_class


def _decode(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, np.bytes_):
        return value.decode().strip()
    return value


def _is_missing_meta_value(field: str, value: float) -> bool:
    if not np.isfinite(value):
        return True
    if value <= -9.0:
        return True
    if value >= 90.0:
        return True
    if field.startswith("HOSTGAL_MAG_") and value >= 40.0:
        return True
    return False


def _table_to_native(path: str | Path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        cols = {}
        for name in data.names:
            arr = np.asarray(data[name])
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
            cols[name] = arr
    return cols


def _file_signature(path: str | Path) -> tuple[str, int, int]:
    path = Path(path)
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def _meta_from_head(cols: dict[str, np.ndarray], idx: int, metadata_fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(metadata_fields), dtype=np.float32)
    mask = np.zeros(len(metadata_fields), dtype=np.float32)
    for j, field in enumerate(metadata_fields):
        if field not in cols:
            continue
        value = cols[field][idx]
        value = float(value) if np.isfinite(value) else np.nan
        if not _is_missing_meta_value(field, value):
            values[j] = value
            mask[j] = 1.0
    return values, mask


def _extract_redshift(cols: dict[str, np.ndarray], idx: int, source: str) -> float:
    if source == "none":
        return np.nan
    field = {
        "final": "REDSHIFT_FINAL",
        "photoz": "HOSTGAL_PHOTOZ",
    }.get(source)
    if field is None:
        raise ValueError(f"unknown ELAsTiCC redshift source: {source}")
    if field not in cols:
        return np.nan
    value = cols[field][idx]
    value = float(value) if np.isfinite(value) else np.nan
    return value if not _is_missing_meta_value(field, value) else np.nan


def _default_cache_root() -> Path:
    env = os.environ.get("NAPTIME_ELASTICC_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "output" / "cache" / "elasticc"


def _cache_path_for_shard(
    head_path: Path,
    *,
    taxonomy: str,
    redshift_source: str,
    metadata_fields: list[str],
) -> Path:
    key = {
        "cache_version": CACHE_VERSION,
        "head_path": str(head_path.resolve()),
        "taxonomy": taxonomy,
        "redshift_source": redshift_source,
        "metadata_fields": tuple(metadata_fields),
    }
    digest = hashlib.blake2b(repr(sorted(key.items())).encode(), digest_size=12).hexdigest()
    return _default_cache_root() / f"{head_path.stem}.{digest}.pkl"


def _load_cached_shard(cache_path: Path, *, head_path: Path, phot_path: Path) -> list[dict] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("head_signature") != _file_signature(head_path):
        return None
    if payload.get("phot_signature") != _file_signature(phot_path):
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    return records


def _write_cached_shard(cache_path: Path, *, head_path: Path, phot_path: Path, records: list[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "head_signature": _file_signature(head_path),
        "phot_signature": _file_signature(phot_path),
        "records": records,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _records_from_shard(
    head_path: Path,
    phot_path: Path,
    *,
    target: int,
    release_name: str,
    redshift_source: str,
    metadata_fields: list[str],
) -> list[dict]:
    records: list[dict] = []
    head = _table_to_native(head_path)
    phot = _table_to_native(phot_path)
    n_rows = len(head["SNID"])
    for i in range(n_rows):
        start = int(head["PTROBS_MIN"][i]) - 1
        end = int(head["PTROBS_MAX"][i])
        bands = np.asarray([str(_decode(v)).strip().lower() for v in phot["BAND"][start:end]])
        keep = np.array([b in BAND2IDX for b in bands], dtype=bool)
        if not np.any(keep):
            continue
        flux = np.asarray(phot["FLUXCAL"][start:end], dtype=np.float32)[keep]
        ferr = np.asarray(phot["FLUXCALERR"][start:end], dtype=np.float32)[keep]
        mjd = np.asarray(phot["MJD"][start:end], dtype=np.float32)[keep]
        bands = bands[keep]
        pos_err = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
        if pos_err.sum() < 4:
            continue
        flux = flux[pos_err]
        ferr = ferr[pos_err]
        mjd = mjd[pos_err]
        bands = bands[pos_err]
        band_idx = np.array([BAND2IDX[b] for b in bands], dtype=np.int64)
        meta_values, meta_mask = _meta_from_head(head, i, metadata_fields)
        z = _extract_redshift(head, i, redshift_source)
        oid = f"{release_name}:{_decode(head['SNID'][i])}"
        records.append(
            {
                "oid": oid,
                "t_raw": mjd,
                "flux_raw": flux,
                "ferr_raw": ferr,
                "band_idx": band_idx,
                "target": target,
                "z": z,
                "meta_values": meta_values,
                "meta_mask": meta_mask,
                "release_name": release_name,
            }
        )
    return records


def load_elasticc_records(
    data_dir: str | Path,
    *,
    taxonomy: str = "focused",
    redshift_source: str = "final",
    metadata_fields: list[str] | None = None,
    max_release_dirs: int | None = None,
    max_shards_per_release: int | None = None,
    max_objects_per_release: int | None = None,
) -> tuple[list[dict], list[str], str]:
    data_dir = Path(data_dir)
    if redshift_source not in ELASTICC_REDSHIFT_SOURCES:
        raise ValueError(f"unknown ELAsTiCC redshift source: {redshift_source}")
    metadata_fields = list(META_FIELDS if metadata_fields is None else metadata_fields)
    taxonomy_name, class_names, release_to_class = get_elasticc_taxonomy(taxonomy)
    release_names = [name for name in release_to_class if (data_dir / name).exists()]
    if max_release_dirs is not None:
        release_names = release_names[:max_release_dirs]

    records: list[dict] = []
    t0 = time.perf_counter()
    print(
        f"[elasticc-load] taxonomy={taxonomy_name} redshift_source={redshift_source} "
        f"metadata_fields={len(metadata_fields)} releases={len(release_names)}",
        flush=True,
    )
    for release_name in release_names:
        release_t0 = time.perf_counter()
        release_dir = data_dir / release_name
        head_paths = sorted(release_dir.glob("*_HEAD.FITS.gz"))
        if max_shards_per_release is not None:
            head_paths = head_paths[:max_shards_per_release]
        release_count = 0
        target = release_to_class.get(release_name, -1)
        print(
            f"[elasticc-load] release={release_name} shards={len(head_paths)} target={target}",
            flush=True,
        )

        for shard_idx, head_path in enumerate(head_paths, start=1):
            phot_path = Path(str(head_path).replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"))
            if not phot_path.exists():
                continue
            shard_t0 = time.perf_counter()
            cache_path = _cache_path_for_shard(
                head_path,
                taxonomy=taxonomy_name,
                redshift_source=redshift_source,
                metadata_fields=metadata_fields,
            )
            shard_records = _load_cached_shard(cache_path, head_path=head_path, phot_path=phot_path)
            cache_hit = shard_records is not None
            if shard_records is None:
                shard_records = _records_from_shard(
                    head_path,
                    phot_path,
                    target=target,
                    release_name=release_name,
                    redshift_source=redshift_source,
                    metadata_fields=metadata_fields,
                )
                _write_cached_shard(cache_path, head_path=head_path, phot_path=phot_path, records=shard_records)
            for rec in shard_records:
                rec_out = dict(rec)
                rec_out["taxonomy"] = taxonomy_name
                records.append(rec_out)
                release_count += 1
                if max_objects_per_release is not None and release_count >= max_objects_per_release:
                    break
            if (
                shard_idx == 1
                or shard_idx == len(head_paths)
                or shard_idx % 10 == 0
                or (max_objects_per_release is not None and release_count >= max_objects_per_release)
            ):
                print(
                    f"[elasticc-load]   shard={shard_idx}/{len(head_paths)} "
                    f"records={len(shard_records)} cache={'hit' if cache_hit else 'write'} "
                    f"elapsed={time.perf_counter() - shard_t0:.1f}s cumulative_release={release_count}",
                    flush=True,
                )
            if max_objects_per_release is not None and release_count >= max_objects_per_release:
                break
        print(
            f"[elasticc-load] release_done={release_name} kept={release_count} "
            f"elapsed={time.perf_counter() - release_t0:.1f}s total_records={len(records)}",
            flush=True,
        )
    print(f"[elasticc-load] done records={len(records)} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return records, class_names, taxonomy_name


def load_elasticc_focus_records(
    data_dir: str | Path,
    *,
    redshift_source: str = "final",
    metadata_fields: list[str] | None = None,
    max_release_dirs: int | None = None,
    max_shards_per_release: int | None = None,
    max_objects_per_release: int | None = None,
) -> list[dict]:
    records, _, _ = load_elasticc_records(
        data_dir,
        taxonomy="focused",
        redshift_source=redshift_source,
        metadata_fields=metadata_fields,
        max_release_dirs=max_release_dirs,
        max_shards_per_release=max_shards_per_release,
        max_objects_per_release=max_objects_per_release,
    )
    return records
