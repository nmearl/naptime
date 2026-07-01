import json
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .baseline import (
    BaselineCollateConfig,
    BaselineLossConfig,
    ConvGNPBaseline,
    ConvGNPBaselineConfig,
    MALLORN_CLASS_NAMES,
    baseline_loss,
    build_lazy_elasticc_datasets,
    build_mallorn_baseline_datasets,
    build_precomputed_baseline_datasets,
    collect_full_context_predictions,
    collect_epoch_predictions,
    compute_flux_norm_stats_from_records,
    compute_z_bounds,
    evaluate_epoch,
    evaluate_multiclass_predictions,
    fit_epoch,
    load_mallorn_training_tables,
    load_baseline_checkpoint,
    build_mallorn_task_log,
    make_baseline_loader,
    make_prefix_context_batch,
    map_mallorn_spectype_to_class,
    predict_full_context,
    prepare_mallorn_baseline_datasets,
    prepare_mallorn_baseline_test_dataset,
    save_baseline_checkpoint,
)
from .elasticc import (
    CLASS_NAMES,
    LEGACY_META_FIELDS,
    META_FIELDS,
    NUM_ELASTICC_CLASSES,
    load_elasticc_focus_records,
)
from .elasticc import (
    ELASTICC_REDSHIFT_SOURCES,
    ELASTICC_TAXONOMY_NAMES,
    extract_elasticc_object_from_ref,
    get_elasticc_taxonomy,
    load_elasticc_records,
    scan_elasticc_index,
)


def _print_elasticc_diagnostics(
    train_records: list[dict], val_records: list[dict], class_names: list[str]
) -> None:
    click.echo("[diagnostics] class counts (train | val):")
    train_labels = np.array([r["target"] for r in train_records])
    val_labels = np.array([r["target"] for r in val_records])
    for c, name in enumerate(class_names):
        click.echo(
            f"  class {c} ({name:10s}): {(train_labels==c).sum():7d} | {(val_labels==c).sum():6d}"
        )

    flux_vals = np.concatenate([r["flux_raw"] for r in train_records[:2000]])
    ferr_vals = np.concatenate([r["ferr_raw"] for r in train_records[:2000]])
    nan_flux = (~np.isfinite(flux_vals)).sum()
    nan_ferr = (~np.isfinite(ferr_vals)).sum()
    click.echo(
        f"[diagnostics] flux non-finite: {nan_flux} / {len(flux_vals)}  ferr non-finite: {nan_ferr} / {len(ferr_vals)}"
    )
    click.echo(
        f"[diagnostics] flux p1/p50/p99: {np.nanpercentile(flux_vals, 1):.2f} / {np.nanpercentile(flux_vals, 50):.2f} / {np.nanpercentile(flux_vals, 99):.2f}"
    )

    meta_sample = [r for r in train_records[:2000] if len(r.get("meta_values", [])) > 0]
    if meta_sample:
        all_meta = np.stack([r["meta_values"] for r in meta_sample])
        all_mask = np.stack([r["meta_mask"] for r in meta_sample])
        for j, field in enumerate(META_FIELDS):
            valid = all_meta[:, j][all_mask[:, j] > 0]
            if len(valid) == 0:
                click.echo(f"  meta {field}: all missing")
            else:
                click.echo(
                    f"  meta {field:30s}: present={len(valid):5d}  min={valid.min():.3g}  max={valid.max():.3g}  median={np.median(valid):.3g}"
                )


def _print_elasticc_ref_diagnostics(
    train_refs: list[dict], val_refs: list[dict], class_names: list[str]
) -> None:
    click.echo("[diagnostics] class counts (train | val):")
    train_labels = np.array([int(ref["target"]) for ref in train_refs], dtype=np.int64)
    val_labels = np.array([int(ref["target"]) for ref in val_refs], dtype=np.int64)
    for c, name in enumerate(class_names):
        click.echo(
            f"  class {c} ({name:10s}): {(train_labels==c).sum():7d} | {(val_labels==c).sum():6d}"
        )

    sample_refs = train_refs[: min(len(train_refs), 512)]
    if not sample_refs:
        return
    sample_records = [extract_elasticc_object_from_ref(ref) for ref in sample_refs]
    flux_vals = np.concatenate([r["flux_raw"] for r in sample_records])
    ferr_vals = np.concatenate([r["ferr_raw"] for r in sample_records])
    nan_flux = (~np.isfinite(flux_vals)).sum()
    nan_ferr = (~np.isfinite(ferr_vals)).sum()
    click.echo(
        f"[diagnostics] flux non-finite: {nan_flux} / {len(flux_vals)}  ferr non-finite: {nan_ferr} / {len(ferr_vals)}"
    )
    click.echo(
        f"[diagnostics] flux p1/p50/p99: {np.nanpercentile(flux_vals, 1):.2f} / {np.nanpercentile(flux_vals, 50):.2f} / {np.nanpercentile(flux_vals, 99):.2f}"
    )

    meta_sample = [r for r in sample_records if len(r.get("meta_values", [])) > 0]
    if meta_sample:
        all_meta = np.stack([r["meta_values"] for r in meta_sample])
        all_mask = np.stack([r["meta_mask"] for r in meta_sample])
        field_names = list(META_FIELDS[: all_meta.shape[1]])
        for j, field in enumerate(field_names):
            valid = all_meta[:, j][all_mask[:, j] > 0]
            if len(valid) == 0:
                click.echo(f"  meta {field}: all missing")
            else:
                click.echo(
                    f"  meta {field:30s}: present={len(valid):5d}  min={valid.min():.3g}  max={valid.max():.3g}  median={np.median(valid):.3g}"
                )


@click.group()
def main():
    """Minimal CLI for the ConvGNP Mallorn baseline."""


def _save_json(path: Path, payload: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _elasticc_per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    num_classes = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    rows: list[dict[str, float | int | str]] = []
    for i in range(num_classes):
        rows.append(
            {
                "class_id": i,
                "class_name": class_names[i] if i < len(class_names) else str(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )
    return rows


def _train_single_split(
    *,
    train_ds,
    val_ds,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    grid_size: int,
    band_emb_dim: int,
    time_fourier_dim: int,
    point_feat_dim: int,
    grid_feat_dim: int,
    conv_layers: int,
    conv_dropout: float,
    classifier_hidden_dim: int,
    decoder_hidden_dim: int,
    setconv_sigmas: tuple[float, ...],
    use_redshift: bool,
    use_rest_frame_time: bool,
    use_metadata: bool,
    metadata_hidden_dim: int,
    metadata_embed_dim: int,
    lambda_recon: float,
    lambda_cls: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    use_latent: bool,
    latent_dim: int,
    latent_hidden_dim: int,
    checkpoint_metric: str,
    device: str,
    full_context_eval: bool = False,
    num_classes: int = 1,
    class_weights: tuple[float, ...] | None = None,
):
    z_min, z_max = compute_z_bounds(train_ds)
    train_cfg = BaselineCollateConfig(
        seed=seed,
        z_min=z_min,
        z_max=z_max,
        mask_prob=mask_prob,
        mask_prob_min=mask_prob_min,
        mask_prob_max=mask_prob_max,
        min_context_points=min_context_points,
        min_target_points=min_target_points,
    )
    val_cfg = BaselineCollateConfig(
        seed=seed,
        z_min=z_min,
        z_max=z_max,
        mask_prob=mask_prob,
        mask_prob_min=mask_prob_min,
        mask_prob_max=mask_prob_max,
        min_context_points=min_context_points,
        min_target_points=min_target_points,
    )
    train_loader = make_baseline_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        training=True,
        cfg=train_cfg,
    )
    val_loader = make_baseline_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        training=False,
        cfg=val_cfg,
    )

    if num_classes == 1:
        pos_count = float(
            sum(int(train_ds.data[oid]["target"]) == 1 for oid in train_ds.object_ids)
        )
        neg_count = float(
            sum(int(train_ds.data[oid]["target"]) == 0 for oid in train_ds.object_ids)
        )
        _pos_weight: float | None = neg_count / max(pos_count, 1.0)
    else:
        _pos_weight = None
    model_cfg = ConvGNPBaselineConfig(
        band_emb_dim=band_emb_dim,
        time_fourier_dim=time_fourier_dim,
        point_feat_dim=point_feat_dim,
        grid_size=grid_size,
        grid_feat_dim=grid_feat_dim,
        conv_layers=conv_layers,
        conv_dropout=conv_dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        setconv_sigmas=tuple(setconv_sigmas),
        use_redshift=use_redshift,
        use_rest_frame_time=use_rest_frame_time,
        use_metadata=use_metadata,
        metadata_dim=(len(META_FIELDS) if use_metadata else 0),
        metadata_hidden_dim=metadata_hidden_dim,
        metadata_embed_dim=metadata_embed_dim,
        use_latent=use_latent,
        latent_dim=latent_dim,
        latent_hidden_dim=latent_hidden_dim,
        num_classes=num_classes,
    )
    loss_cfg = BaselineLossConfig(
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        pos_weight=_pos_weight,
        beta_kl=beta_kl,
        kl_warmup_epochs=kl_warmup_epochs,
        class_weights=class_weights,
    )
    model = ConvGNPBaseline(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(10, max(1, epochs // 10))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    best_metrics = None
    best_state = None
    best_epoch = 0
    patience_count = 0
    metric_name = "best_f1" if num_classes == 1 else "macro_f1"
    aux_name = "ap" if num_classes == 1 else "macro_auroc"

    for epoch in range(1, epochs + 1):
        train_cfg.epoch = epoch
        if kl_warmup_epochs > 0:
            loss_cfg.beta_kl = beta_kl * min(1.0, (epoch - 1) / kl_warmup_epochs)
        train_metrics = fit_epoch(model, train_loader, optimizer, loss_cfg, device)
        if train_metrics.get("nan_batches", 0) > 0:
            click.echo(
                f"  [warn] epoch {epoch}: {int(train_metrics['nan_batches'])} NaN batches skipped"
            )
        if full_context_eval:
            val_metrics, _ = collect_full_context_predictions(
                model,
                val_ds,
                batch_size=batch_size,
                z_min=z_min,
                z_max=z_max,
                device=device,
            )
            val_metrics["recon"] = float("nan")
            val_metrics["cls"] = float("nan")
            val_metrics["total"] = float("nan")
        else:
            val_metrics = evaluate_epoch(model, val_loader, loss_cfg, device)
        if val_metrics.get("nan_batches", 0) > 0:
            click.echo(
                f"  [warn] epoch {epoch}: {int(val_metrics['nan_batches'])} validation NaN batches skipped"
            )
        if not val_metrics:
            click.echo(f"  [warn] epoch {epoch}: no finite validation batches")
        scheduler.step()

        click.echo(
            f"  epoch {epoch:3d}"
            f" train_total={train_metrics.get('total', float('nan')):.4f}"
            f" train_recon={train_metrics.get('recon', float('nan')):.4f}"
            f" train_cls={train_metrics.get('cls', float('nan')):.4f}"
            f" val_total={val_metrics.get('total', float('nan')):.4f}"
            f" val_recon={val_metrics.get('recon', float('nan')):.4f}"
            f" {metric_name}={val_metrics.get(metric_name, float('nan')):.4f}"
            f" {aux_name}={val_metrics.get(aux_name, float('nan')):.4f}"
        )

        current = val_metrics.get(checkpoint_metric, float("-inf"))
        best = float("inf") if checkpoint_metric == "recon" else float("-inf")
        if best_metrics is not None:
            best = best_metrics.get(checkpoint_metric, best)
        improved = current < best if checkpoint_metric == "recon" else current > best
        if improved or best_metrics is None:
            best_metrics = dict(val_metrics)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            patience_count = 0
            click.echo(f"    [best] epoch {epoch}: {checkpoint_metric}={current:.4f}")
        else:
            patience_count += 1
            click.echo(f"    [patience] {patience_count}/{patience}")
            if patience_count >= patience:
                click.echo(
                    f"    [early-stop] no improvement on {checkpoint_metric} for {patience} epochs"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, model_cfg, loss_cfg, best_metrics or {}, best_epoch, z_min, z_max


@main.command("train-mallorn-baseline")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=200, show_default=True)
@click.option("--patience", type=int, default=40, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--mask-prob", type=float, default=0.4, show_default=True)
@click.option("--mask-prob-min", type=float, default=0.0, show_default=True)
@click.option("--mask-prob-max", type=float, default=0.8, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--band-emb-dim", type=int, default=8, show_default=True)
@click.option("--time-fourier-dim", type=int, default=8, show_default=True)
@click.option("--point-feat-dim", type=int, default=64, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--conv-dropout", type=float, default=0.0, show_default=True)
@click.option("--classifier-hidden-dim", type=int, default=128, show_default=True)
@click.option("--decoder-hidden-dim", type=int, default=128, show_default=True)
@click.option(
    "--setconv-sigma",
    "setconv_sigmas",
    type=float,
    multiple=True,
    default=(0.015, 0.03, 0.06),
    show_default=True,
)
@click.option("--use-redshift/--no-use-redshift", default=True, show_default=True)
@click.option(
    "--use-rest-frame-time/--no-use-rest-frame-time", default=False, show_default=True
)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--kl-warmup-epochs", type=int, default=20, show_default=True)
@click.option("--use-latent/--no-latent", default=True, show_default=True)
@click.option("--latent-dim", type=int, default=8, show_default=True)
@click.option("--latent-hidden-dim", type=int, default=64, show_default=True)
@click.option("--num-classes", type=int, default=1, show_default=True)
@click.option(
    "--checkpoint-metric",
    type=click.Choice(
        ["best_f1", "ap", "recon", "macro_f1", "weighted_f1", "macro_auroc"]
    ),
    default="ap",
    show_default=True,
)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
@click.option(
    "--full-context-eval/--masked-context-eval", default=False, show_default=True
)
def train_mallorn_baseline(
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    val_frac: float,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    grid_size: int,
    band_emb_dim: int,
    time_fourier_dim: int,
    point_feat_dim: int,
    grid_feat_dim: int,
    conv_layers: int,
    conv_dropout: float,
    classifier_hidden_dim: int,
    decoder_hidden_dim: int,
    setconv_sigmas: tuple[float, ...],
    use_redshift: bool,
    use_rest_frame_time: bool,
    lambda_recon: float,
    lambda_cls: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    use_latent: bool,
    latent_dim: int,
    latent_hidden_dim: int,
    num_classes: int,
    checkpoint_metric: str,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
    full_context_eval: bool,
):
    if num_classes > 1 and checkpoint_metric in {"best_f1", "ap"}:
        raise click.ClickException(
            "--checkpoint-metric must be one of macro_f1, macro_auroc, or recon when --num-classes > 1"
        )
    if full_context_eval and checkpoint_metric == "recon":
        raise click.ClickException(
            "--full-context-eval cannot be combined with --checkpoint-metric recon"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _, _, train_ds, val_ds, _, _ = prepare_mallorn_baseline_datasets(
        data_dir,
        seed=seed,
        val_frac=val_frac,
        max_obs=max_obs,
        keep_all_snr_gt=keep_all_snr_gt,
        use_rest_frame_time=use_rest_frame_time,
        num_classes=num_classes,
    )
    z_min, z_max = compute_z_bounds(train_ds)
    train_cfg = BaselineCollateConfig(
        seed=seed,
        z_min=z_min,
        z_max=z_max,
        mask_prob=mask_prob,
        mask_prob_min=mask_prob_min,
        mask_prob_max=mask_prob_max,
        min_context_points=min_context_points,
        min_target_points=min_target_points,
    )
    val_cfg = BaselineCollateConfig(
        seed=seed,
        z_min=z_min,
        z_max=z_max,
        mask_prob=mask_prob,
        min_context_points=min_context_points,
        min_target_points=min_target_points,
    )
    train_loader = make_baseline_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        training=True,
        cfg=train_cfg,
    )
    val_loader = make_baseline_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        training=False,
        cfg=val_cfg,
    )

    pos_count = float(
        sum(int(train_ds.data[oid]["target"]) == 1 for oid in train_ds.object_ids)
    )
    neg_count = float(
        sum(int(train_ds.data[oid]["target"]) == 0 for oid in train_ds.object_ids)
    )
    pos_weight = neg_count / max(pos_count, 1.0)
    class_weights = None
    if num_classes > 1:
        train_labels = np.array(
            [int(train_ds.data[oid]["target"]) for oid in train_ds.object_ids]
        )
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        inv_freq = 1.0 / counts
        inv_freq = inv_freq / inv_freq.sum() * num_classes
        class_weights = tuple(float(w) for w in inv_freq)
    model_cfg = ConvGNPBaselineConfig(
        band_emb_dim=band_emb_dim,
        time_fourier_dim=time_fourier_dim,
        point_feat_dim=point_feat_dim,
        grid_size=grid_size,
        grid_feat_dim=grid_feat_dim,
        conv_layers=conv_layers,
        conv_dropout=conv_dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        setconv_sigmas=tuple(setconv_sigmas),
        use_redshift=use_redshift,
        use_rest_frame_time=use_rest_frame_time,
        use_latent=use_latent,
        latent_dim=latent_dim,
        latent_hidden_dim=latent_hidden_dim,
        num_classes=num_classes,
    )
    loss_cfg = BaselineLossConfig(
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        pos_weight=(pos_weight if num_classes == 1 else None),
        beta_kl=beta_kl,
        kl_warmup_epochs=kl_warmup_epochs,
        class_weights=class_weights,
    )
    model = ConvGNPBaseline(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(10, max(1, epochs // 10))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    header = (
        f"[baseline] train={len(train_ds)} val={len(val_ds)} "
        f"z_norm=[{z_min:.3f}, {z_max:.3f}] device={device}"
    )
    if num_classes == 1:
        header += f" pos_weight={pos_weight:.3f}"
    else:
        header += f" num_classes={num_classes}"
        if num_classes == len(MALLORN_CLASS_NAMES):
            header += " classes=" + ",".join(MALLORN_CLASS_NAMES)
    click.echo(header)
    if num_classes == 1:
        click.echo(
            f"{'Ep':>4} {'train_total':>11} {'val_total':>10} {'bestF1':>8} {'AP':>8} {'recon':>8}"
        )
    else:
        click.echo(
            f"{'Ep':>4} {'train_total':>11} {'val_total':>10} {'macroF1':>8} {'mAUROC':>8} {'recon':>8}"
        )
    click.echo("-" * 64)

    history: list[dict] = []
    if num_classes == 1:
        best_scores = {
            "best_f1": float("-inf"),
            "ap": float("-inf"),
            "recon": float("inf"),
        }
        best_paths = {
            "best_f1": out_dir / "best_f1_checkpoint.pt",
            "ap": out_dir / "best_ap_checkpoint.pt",
            "recon": out_dir / "best_recon_checkpoint.pt",
        }
    else:
        best_scores = {
            "macro_f1": float("-inf"),
            "macro_auroc": float("-inf"),
            "recon": float("inf"),
        }
        best_paths = {
            "macro_f1": out_dir / "best_macro_f1_checkpoint.pt",
            "macro_auroc": out_dir / "best_macro_auroc_checkpoint.pt",
            "recon": out_dir / "best_recon_checkpoint.pt",
        }
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_cfg.epoch = epoch
        if kl_warmup_epochs > 0:
            loss_cfg.beta_kl = beta_kl * min(1.0, (epoch - 1) / kl_warmup_epochs)
        train_metrics = fit_epoch(model, train_loader, optimizer, loss_cfg, device)
        if full_context_eval:
            val_metrics, _ = collect_full_context_predictions(
                model,
                val_ds,
                batch_size=batch_size,
                z_min=z_min,
                z_max=z_max,
                device=device,
            )
            val_metrics["recon"] = float("nan")
            val_metrics["cls"] = float("nan")
            val_metrics["total"] = float("nan")
        else:
            val_metrics = evaluate_epoch(model, val_loader, loss_cfg, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        improved = []
        if num_classes == 1:
            if val_metrics.get("best_f1", float("-inf")) > best_scores["best_f1"]:
                best_scores["best_f1"] = val_metrics["best_f1"]
                improved.append("best_f1")
            if val_metrics.get("ap", float("-inf")) > best_scores["ap"]:
                best_scores["ap"] = val_metrics["ap"]
                improved.append("ap")
        else:
            if val_metrics.get("macro_f1", float("-inf")) > best_scores["macro_f1"]:
                best_scores["macro_f1"] = val_metrics["macro_f1"]
                improved.append("macro_f1")
            if (
                val_metrics.get("macro_auroc", float("-inf"))
                > best_scores["macro_auroc"]
            ):
                best_scores["macro_auroc"] = val_metrics["macro_auroc"]
                improved.append("macro_auroc")
        if val_metrics.get("recon", float("inf")) < best_scores["recon"]:
            best_scores["recon"] = val_metrics["recon"]
            improved.append("recon")

        if epoch == 1 or epoch % 10 == 0 or improved:
            mark = f" ✓[{','.join(improved)}]" if improved else ""
            if num_classes == 1:
                click.echo(
                    f"{epoch:4d} {train_metrics['total']:11.4f} {val_metrics['total']:10.4f} "
                    f"{val_metrics.get('best_f1', float('nan')):8.4f} {val_metrics.get('ap', float('nan')):8.4f} "
                    f"{val_metrics.get('recon', float('nan')):8.4f}{mark}"
                )
            else:
                click.echo(
                    f"{epoch:4d} {train_metrics['total']:11.4f} {val_metrics['total']:10.4f} "
                    f"{val_metrics.get('macro_f1', float('nan')):8.4f} {val_metrics.get('macro_auroc', float('nan')):8.4f} "
                    f"{val_metrics.get('recon', float('nan')):8.4f}{mark}"
                )

        for key in improved:
            save_baseline_checkpoint(
                best_paths[key],
                model=model,
                model_cfg=model_cfg,
                loss_cfg=loss_cfg,
                epoch=epoch,
                metrics=val_metrics,
                history=history,
                z_min=z_min,
                z_max=z_max,
                flux_center_by_band=train_ds.flux_center_by_band,
                flux_scale_by_band=train_ds.flux_scale_by_band,
            )

        if checkpoint_metric in improved:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                click.echo(
                    f"[early stop] No improvement for {patience} epochs on {checkpoint_metric}."
                )
                break

        _save_json(out_dir / "training_log.json", history)

    primary_path = out_dir / "best_primary_checkpoint.pt"
    source = best_paths[checkpoint_metric]
    if source.exists():
        primary_path.write_bytes(source.read_bytes())
    if num_classes == 1:
        click.echo(f"[done] best_f1 checkpoint: {best_paths['best_f1']}")
        click.echo(f"[done] best_ap checkpoint: {best_paths['ap']}")
    else:
        click.echo(f"[done] best_macro_f1 checkpoint: {best_paths['macro_f1']}")
        click.echo(f"[done] best_macro_auroc checkpoint: {best_paths['macro_auroc']}")
    click.echo(f"[done] best_recon checkpoint: {best_paths['recon']}")
    click.echo(f"[done] primary checkpoint: {primary_path}")


@main.command("train-elasticc-focus-baseline")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=120, show_default=True)
@click.option("--patience", type=int, default=25, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--mask-prob", type=float, default=0.4, show_default=True)
@click.option("--mask-prob-min", type=float, default=0.0, show_default=True)
@click.option("--mask-prob-max", type=float, default=0.8, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--band-emb-dim", type=int, default=8, show_default=True)
@click.option("--time-fourier-dim", type=int, default=8, show_default=True)
@click.option("--point-feat-dim", type=int, default=64, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--conv-dropout", type=float, default=0.0, show_default=True)
@click.option("--classifier-hidden-dim", type=int, default=128, show_default=True)
@click.option("--decoder-hidden-dim", type=int, default=128, show_default=True)
@click.option(
    "--setconv-sigma",
    "setconv_sigmas",
    type=float,
    multiple=True,
    default=(0.015, 0.03, 0.06),
    show_default=True,
)
@click.option("--use-redshift/--no-use-redshift", default=True, show_default=True)
@click.option(
    "--use-rest-frame-time/--no-use-rest-frame-time", default=False, show_default=True
)
@click.option("--use-metadata/--no-use-metadata", default=True, show_default=True)
@click.option("--metadata-hidden-dim", type=int, default=64, show_default=True)
@click.option("--metadata-embed-dim", type=int, default=32, show_default=True)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--kl-warmup-epochs", type=int, default=20, show_default=True)
@click.option("--use-latent/--no-latent", default=True, show_default=True)
@click.option("--latent-dim", type=int, default=8, show_default=True)
@click.option("--latent-hidden-dim", type=int, default=64, show_default=True)
@click.option("--num-classes", type=int, default=7, show_default=True)
@click.option(
    "--checkpoint-metric",
    type=click.Choice(
        ["best_f1", "ap", "recon", "macro_f1", "weighted_f1", "macro_auroc"]
    ),
    default="macro_f1",
    show_default=True,
)
@click.option(
    "--elasticc-taxonomy",
    type=click.Choice(ELASTICC_TAXONOMY_NAMES),
    default="focused",
    show_default=True,
)
@click.option(
    "--redshift-source",
    type=click.Choice(ELASTICC_REDSHIFT_SOURCES),
    default="photoz",
    show_default=True,
)
@click.option("--max-release-dirs", type=int, default=None)
@click.option("--max-shards-per-release", type=int, default=None)
@click.option("--max-objects-per-release", type=int, default=None)
@click.option("--lazy-elasticc/--eager-elasticc", default=False, show_default=True)
@click.option("--device", type=str, default=None)
def train_elasticc_focus_baseline(
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    val_frac: float,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    grid_size: int,
    band_emb_dim: int,
    time_fourier_dim: int,
    point_feat_dim: int,
    grid_feat_dim: int,
    conv_layers: int,
    conv_dropout: float,
    classifier_hidden_dim: int,
    decoder_hidden_dim: int,
    setconv_sigmas: tuple[float, ...],
    use_redshift: bool,
    use_rest_frame_time: bool,
    use_metadata: bool,
    metadata_hidden_dim: int,
    metadata_embed_dim: int,
    lambda_recon: float,
    lambda_cls: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    use_latent: bool,
    latent_dim: int,
    latent_hidden_dim: int,
    num_classes: int,
    checkpoint_metric: str,
    elasticc_taxonomy: str,
    redshift_source: str,
    max_release_dirs: int | None,
    max_shards_per_release: int | None,
    max_objects_per_release: int | None,
    lazy_elasticc: bool,
    device: str | None,
):
    taxonomy_name, class_names, _ = get_elasticc_taxonomy(elasticc_taxonomy)
    if num_classes < len(class_names):
        raise click.ClickException(
            f"--num-classes must be at least {len(class_names)} for the current ELAsTiCC taxonomy ({taxonomy_name})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if lazy_elasticc:
        refs, class_names, taxonomy_name = scan_elasticc_index(
            data_dir,
            taxonomy=elasticc_taxonomy,
            redshift_source=redshift_source,
            max_release_dirs=max_release_dirs,
            max_shards_per_release=max_shards_per_release,
            max_objects_per_release=max_objects_per_release,
        )
        if not refs:
            raise click.ClickException("No ELAsTiCC focus records were indexed.")
        all_labels = [int(ref["target"]) for ref in refs]
        train_idx, val_idx = train_test_split(
            np.arange(len(refs)),
            test_size=val_frac,
            stratify=all_labels,
            random_state=seed,
        )
        train_refs = [refs[int(i)] for i in train_idx]
        val_refs = [refs[int(i)] for i in val_idx]
        train_ds, val_ds = build_lazy_elasticc_datasets(
            train_refs=train_refs,
            val_refs=val_refs,
            use_rest_frame_time=use_rest_frame_time,
        )
        _print_elasticc_ref_diagnostics(train_refs, val_refs, class_names)
    else:
        records, class_names, taxonomy_name = load_elasticc_records(
            data_dir,
            taxonomy=elasticc_taxonomy,
            redshift_source=redshift_source,
            max_release_dirs=max_release_dirs,
            max_shards_per_release=max_shards_per_release,
            max_objects_per_release=max_objects_per_release,
        )
        if not records:
            raise click.ClickException("No ELAsTiCC focus records were loaded.")
        all_labels = [int(r["target"]) for r in records]
        train_idx, val_idx = train_test_split(
            np.arange(len(records)),
            test_size=val_frac,
            stratify=all_labels,
            random_state=seed,
        )
        train_records = [records[int(i)] for i in train_idx]
        val_records = [records[int(i)] for i in val_idx]
        train_ds, val_ds = build_precomputed_baseline_datasets(
            train_records=train_records,
            val_records=val_records,
            use_rest_frame_time=use_rest_frame_time,
        )
        _print_elasticc_diagnostics(train_records, val_records, class_names)

    # Compute inverse-frequency class weights for multi-class
    _class_weights: tuple[float, ...] | None = None
    if num_classes > 1:
        if hasattr(train_ds, "target_by_oid"):
            train_labels = np.array(
                [int(train_ds.target_by_oid[oid]) for oid in train_ds.object_ids],
                dtype=np.int64,
            )
        else:
            train_labels = np.array(
                [int(train_ds.data[oid]["target"]) for oid in train_ds.object_ids],
                dtype=np.int64,
            )
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        inv_freq = 1.0 / counts
        inv_freq = inv_freq / inv_freq.sum() * num_classes
        _class_weights = tuple(float(w) for w in inv_freq)

    model, model_cfg, loss_cfg, metrics, best_epoch, z_min, z_max = _train_single_split(
        train_ds=train_ds,
        val_ds=val_ds,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        mask_prob=mask_prob,
        mask_prob_min=mask_prob_min,
        mask_prob_max=mask_prob_max,
        min_context_points=min_context_points,
        min_target_points=min_target_points,
        grid_size=grid_size,
        band_emb_dim=band_emb_dim,
        time_fourier_dim=time_fourier_dim,
        point_feat_dim=point_feat_dim,
        grid_feat_dim=grid_feat_dim,
        conv_layers=conv_layers,
        conv_dropout=conv_dropout,
        classifier_hidden_dim=classifier_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        setconv_sigmas=setconv_sigmas,
        use_redshift=use_redshift,
        use_rest_frame_time=use_rest_frame_time,
        use_metadata=use_metadata,
        metadata_hidden_dim=metadata_hidden_dim,
        metadata_embed_dim=metadata_embed_dim,
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        beta_kl=beta_kl,
        kl_warmup_epochs=kl_warmup_epochs,
        use_latent=use_latent,
        latent_dim=latent_dim,
        latent_hidden_dim=latent_hidden_dim,
        checkpoint_metric=checkpoint_metric,
        device=device,
        num_classes=num_classes,
        class_weights=_class_weights,
    )
    if num_classes > 1:
        click.echo(
            f"[elasticc] train={len(train_ds)} val={len(val_ds)} use_metadata={str(use_metadata).lower()} "
            f"macro_f1={metrics.get('macro_f1', float('nan')):.4f} "
            f"macro_auroc={metrics.get('macro_auroc', float('nan')):.4f} "
            f"recon={metrics.get('recon', float('nan')):.4f}"
        )
    else:
        click.echo(
            f"[elasticc] train={len(train_ds)} val={len(val_ds)} use_metadata={str(use_metadata).lower()} "
            f"best_f1={metrics.get('best_f1', float('nan')):.4f} ap={metrics.get('ap', float('nan')):.4f} "
            f"recon={metrics.get('recon', float('nan')):.4f}"
        )
    save_baseline_checkpoint(
        out_dir / "best_primary_checkpoint.pt",
        model=model,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        epoch=best_epoch,
        metrics=metrics,
        history=[],
        z_min=z_min,
        z_max=z_max,
        flux_center_by_band=train_ds.flux_center_by_band,
        flux_scale_by_band=train_ds.flux_scale_by_band,
        extra_payload={
            "elasticc_taxonomy": taxonomy_name,
            "class_names": class_names,
            "elasticc_redshift_source": redshift_source,
            "elasticc_metadata_fields": list(META_FIELDS),
        },
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "use_metadata": use_metadata,
                "elasticc_taxonomy": taxonomy_name,
                "elasticc_redshift_source": redshift_source,
                "elasticc_metadata_fields": list(META_FIELDS),
                "class_names": class_names,
                "metrics": metrics,
                "epoch": best_epoch,
            },
            f,
            indent=2,
        )
    click.echo(f"[done] checkpoint -> {out_dir / 'best_primary_checkpoint.pt'}")


@main.command("crossval-mallorn-baseline")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--folds", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=120, show_default=True)
@click.option("--patience", type=int, default=25, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--mask-prob", type=float, default=0.4, show_default=True)
@click.option("--mask-prob-min", type=float, default=0.0, show_default=True)
@click.option("--mask-prob-max", type=float, default=0.8, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--band-emb-dim", type=int, default=8, show_default=True)
@click.option("--time-fourier-dim", type=int, default=8, show_default=True)
@click.option("--point-feat-dim", type=int, default=64, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--conv-dropout", type=float, default=0.0, show_default=True)
@click.option("--classifier-hidden-dim", type=int, default=128, show_default=True)
@click.option("--decoder-hidden-dim", type=int, default=128, show_default=True)
@click.option(
    "--setconv-sigma",
    "setconv_sigmas",
    type=float,
    multiple=True,
    default=(0.015, 0.03, 0.06),
    show_default=True,
)
@click.option("--use-redshift/--no-use-redshift", default=True, show_default=True)
@click.option(
    "--use-rest-frame-time/--no-use-rest-frame-time", default=False, show_default=True
)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--kl-warmup-epochs", type=int, default=20, show_default=True)
@click.option("--use-latent/--no-latent", default=True, show_default=True)
@click.option("--latent-dim", type=int, default=8, show_default=True)
@click.option("--latent-hidden-dim", type=int, default=64, show_default=True)
@click.option(
    "--checkpoint-metric",
    type=click.Choice(["best_f1", "ap", "recon"]),
    default="ap",
    show_default=True,
)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
@click.option(
    "--full-context-eval/--masked-context-eval", default=False, show_default=True
)
def crossval_mallorn_baseline(
    data_dir: Path,
    out_dir: Path,
    folds: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    grid_size: int,
    band_emb_dim: int,
    time_fourier_dim: int,
    point_feat_dim: int,
    grid_feat_dim: int,
    conv_layers: int,
    conv_dropout: float,
    classifier_hidden_dim: int,
    decoder_hidden_dim: int,
    setconv_sigmas: tuple[float, ...],
    use_redshift: bool,
    use_rest_frame_time: bool,
    lambda_recon: float,
    lambda_cls: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    use_latent: bool,
    latent_dim: int,
    latent_hidden_dim: int,
    checkpoint_metric: str,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
    full_context_eval: bool,
):
    if full_context_eval and checkpoint_metric == "recon":
        raise click.ClickException(
            "--full-context-eval cannot be combined with --checkpoint-metric recon"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lc, log = load_mallorn_training_tables(
        data_dir, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt
    )
    all_ids = log["object_id"].tolist()
    all_tgts = np.asarray(log["target"].tolist(), dtype=int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    click.echo(f"[crossval] objects={len(all_ids)} folds={folds} device={device}")

    fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_ids, all_tgts), start=1
    ):
        torch.manual_seed(seed + fold_idx)
        train_ids = [all_ids[i] for i in train_idx]
        val_ids = [all_ids[i] for i in val_idx]
        train_ds, val_ds = build_mallorn_baseline_datasets(
            lc,
            log,
            train_ids=train_ids,
            val_ids=val_ids,
            use_rest_frame_time=use_rest_frame_time,
        )
        model, model_cfg, loss_cfg, metrics, best_epoch, z_min, z_max = (
            _train_single_split(
                train_ds=train_ds,
                val_ds=val_ds,
                seed=seed + fold_idx,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                num_workers=num_workers,
                mask_prob=mask_prob,
                mask_prob_min=mask_prob_min,
                mask_prob_max=mask_prob_max,
                min_context_points=min_context_points,
                min_target_points=min_target_points,
                grid_size=grid_size,
                band_emb_dim=band_emb_dim,
                time_fourier_dim=time_fourier_dim,
                point_feat_dim=point_feat_dim,
                grid_feat_dim=grid_feat_dim,
                conv_layers=conv_layers,
                conv_dropout=conv_dropout,
                classifier_hidden_dim=classifier_hidden_dim,
                decoder_hidden_dim=decoder_hidden_dim,
                setconv_sigmas=setconv_sigmas,
                use_redshift=use_redshift,
                use_rest_frame_time=use_rest_frame_time,
                use_metadata=False,
                metadata_hidden_dim=64,
                metadata_embed_dim=32,
                lambda_recon=lambda_recon,
                lambda_cls=lambda_cls,
                beta_kl=beta_kl,
                kl_warmup_epochs=kl_warmup_epochs,
                use_latent=use_latent,
                latent_dim=latent_dim,
                latent_hidden_dim=latent_hidden_dim,
                checkpoint_metric=checkpoint_metric,
                device=device,
                full_context_eval=full_context_eval,
            )
        )
        fold_rows.append({"fold": fold_idx, "best_epoch": best_epoch, **metrics})
        click.echo(
            f"  fold={fold_idx} epoch={best_epoch} best_f1={metrics.get('best_f1', float('nan')):.4f} "
            f"ap={metrics.get('ap', float('nan')):.4f} recon={metrics.get('recon', float('nan')):.4f}"
        )
        if full_context_eval:
            _, val_probs = collect_full_context_predictions(
                model,
                val_ds,
                batch_size=batch_size,
                z_min=z_min,
                z_max=z_max,
                device=device,
            )
        else:
            val_cfg = BaselineCollateConfig(
                seed=seed + fold_idx,
                z_min=z_min,
                z_max=z_max,
                mask_prob=mask_prob,
                min_context_points=min_context_points,
                min_target_points=min_target_points,
            )
            val_loader = make_baseline_loader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                training=False,
                cfg=val_cfg,
            )
            _, val_probs = collect_epoch_predictions(
                model, val_loader, loss_cfg, device
            )
        threshold = float(metrics.get("best_threshold", 0.5))
        for row in val_probs:
            prob = float(row["prob_tde"])
            oid = str(row["object_id"])
            oof_rows.append(
                {
                    "fold": fold_idx,
                    "object_id": oid,
                    "target": int(row["target"]),
                    "prob_tde": prob,
                    "prediction": int(prob >= threshold),
                    "fold_threshold": threshold,
                }
            )
        fold_ckpt = out_dir / f"fold_{fold_idx}_checkpoint.pt"
        save_baseline_checkpoint(
            fold_ckpt,
            model=model,
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
            epoch=best_epoch,
            metrics=metrics,
            history=[],
            z_min=z_min,
            z_max=z_max,
            flux_center_by_band=train_ds.flux_center_by_band,
            flux_scale_by_band=train_ds.flux_scale_by_band,
        )

    y = np.asarray([row["target"] for row in oof_rows], dtype=int)
    probs = np.asarray([row["prob_tde"] for row in oof_rows], dtype=float)
    pred_fold = np.asarray([row["prediction"] for row in oof_rows], dtype=int)
    oof_ap = float(average_precision_score(y, probs))
    oof_f1_fold = float(f1_score(y, pred_fold, zero_division=0))
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 197):
        pred = (probs >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    summary = {
        "folds": folds,
        "oof_ap": oof_ap,
        "oof_best_f1_global_threshold": best_f1,
        "oof_best_threshold_global": best_threshold,
        "oof_f1_using_fold_thresholds": oof_f1_fold,
        "mean_fold_best_f1": float(
            np.mean([row.get("best_f1", np.nan) for row in fold_rows])
        ),
        "mean_fold_ap": float(np.mean([row.get("ap", np.nan) for row in fold_rows])),
        "mean_fold_recon": float(
            np.mean([row.get("recon", np.nan) for row in fold_rows])
        ),
    }
    click.echo(
        f"[crossval] oof_ap={summary['oof_ap']:.4f} "
        f"oof_best_f1_global={summary['oof_best_f1_global_threshold']:.4f} "
        f"oof_f1_fold_thresholds={summary['oof_f1_using_fold_thresholds']:.4f}"
    )
    _save_json(out_dir / "crossval_folds.json", fold_rows)
    _save_json(out_dir / "crossval_oof.json", oof_rows)
    with open(out_dir / "crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    click.echo(f"[done] crossval -> {out_dir}")


@main.command("evaluate-mallorn-baseline")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--mask-prob", type=float, default=0.4, show_default=True)
@click.option("--mask-prob-min", type=float, default=0.0, show_default=True)
@click.option("--mask-prob-max", type=float, default=0.8, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
@click.option(
    "--out-json", type=click.Path(dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option(
    "--full-context-eval/--masked-context-eval", default=False, show_default=True
)
def evaluate_mallorn_baseline(
    data_dir: Path,
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
    out_json: Path | None,
    out_dir: Path | None,
    full_context_eval: bool,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_baseline_checkpoint(checkpoint, device=device)
    use_rest_frame_time = bool(
        ckpt.get("model_cfg", {}).get("use_rest_frame_time", False)
    )
    num_classes = int(ckpt.get("model_cfg", {}).get("num_classes", 1))
    _, _, train_ds, val_ds, _, _ = prepare_mallorn_baseline_datasets(
        data_dir,
        seed=seed,
        val_frac=val_frac,
        max_obs=max_obs,
        keep_all_snr_gt=keep_all_snr_gt,
        use_rest_frame_time=use_rest_frame_time,
        num_classes=num_classes,
    )
    z_min = float(ckpt["z_min"])
    z_max = float(ckpt["z_max"])
    loss_cfg = BaselineLossConfig(**ckpt["loss_cfg"])
    if full_context_eval:
        metrics, rows = collect_full_context_predictions(
            model,
            val_ds,
            batch_size=batch_size,
            z_min=z_min,
            z_max=z_max,
            device=device,
        )
        metrics["recon"] = float("nan")
        metrics["cls"] = float("nan")
        metrics["total"] = float("nan")
    else:
        val_cfg = BaselineCollateConfig(
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            mask_prob=mask_prob,
            min_context_points=min_context_points,
            min_target_points=min_target_points,
        )
        val_loader = make_baseline_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            training=False,
            cfg=val_cfg,
        )
        metrics, rows = collect_epoch_predictions(model, val_loader, loss_cfg, device)
    if num_classes == 1:
        click.echo(
            f"[eval] checkpoint={checkpoint} best_f1={metrics.get('best_f1', float('nan')):.4f} "
            f"ap={metrics.get('ap', float('nan')):.4f} recon={metrics.get('recon', float('nan')):.4f} "
            f"total={metrics.get('total', float('nan')):.4f}"
        )
    else:
        click.echo(
            f"[eval] checkpoint={checkpoint} macro_f1={metrics.get('macro_f1', float('nan')):.4f} "
            f"macro_auroc={metrics.get('macro_auroc', float('nan')):.4f} recon={metrics.get('recon', float('nan')):.4f} "
            f"total={metrics.get('total', float('nan')):.4f}"
        )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"[done] metrics -> {out_json}")
    if out_dir is not None and num_classes > 1:
        out_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pred_df = pd.DataFrame(rows)
        prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
        pred_df["pred_class"] = pred_df[prob_cols].to_numpy().argmax(axis=1)
        pred_df["true_class_name"] = pred_df["target"].map(
            lambda x: (
                MALLORN_CLASS_NAMES[int(x)]
                if int(x) < len(MALLORN_CLASS_NAMES)
                else str(int(x))
            )
        )
        pred_df["pred_class_name"] = pred_df["pred_class"].map(
            lambda x: (
                MALLORN_CLASS_NAMES[int(x)]
                if int(x) < len(MALLORN_CLASS_NAMES)
                else str(int(x))
            )
        )

        y_true = pred_df["target"].to_numpy(dtype=int)
        y_pred = pred_df["pred_class"].to_numpy(dtype=int)
        per_class_rows = _elasticc_per_class_rows(
            y_true,
            y_pred,
            MALLORN_CLASS_NAMES[:num_classes],
        )

        predictions_csv = out_dir / "val_predictions.csv"
        per_class_csv = out_dir / "per_class_metrics.csv"
        metrics_json = out_dir / "metrics.json"

        pred_df.to_csv(predictions_csv, index=False)
        pd.DataFrame(per_class_rows).to_csv(per_class_csv, index=False)
        with open(metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"[done] predictions -> {predictions_csv}")
        click.echo(f"[done] per-class metrics -> {per_class_csv}")


@main.command("evaluate-elasticc-focus-baseline")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--mask-prob", type=float, default=0.4, show_default=True)
@click.option("--mask-prob-min", type=float, default=0.0, show_default=True)
@click.option("--mask-prob-max", type=float, default=0.8, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option(
    "--elasticc-taxonomy", type=click.Choice(ELASTICC_TAXONOMY_NAMES), default=None
)
@click.option(
    "--redshift-source", type=click.Choice(ELASTICC_REDSHIFT_SOURCES), default=None
)
@click.option("--max-release-dirs", type=int, default=None)
@click.option("--max-shards-per-release", type=int, default=None)
@click.option("--max-objects-per-release", type=int, default=None)
@click.option("--lazy-elasticc/--eager-elasticc", default=False, show_default=True)
@click.option("--max-cached-shards", type=int, default=1, show_default=True)
@click.option("--device", type=str, default=None)
@click.option(
    "--full-context-eval/--masked-context-eval", default=False, show_default=True
)
def evaluate_elasticc_focus_baseline(
    data_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    mask_prob: float,
    mask_prob_min: float,
    mask_prob_max: float,
    min_context_points: int,
    min_target_points: int,
    elasticc_taxonomy: str | None,
    redshift_source: str | None,
    max_release_dirs: int | None,
    max_shards_per_release: int | None,
    max_objects_per_release: int | None,
    lazy_elasticc: bool,
    max_cached_shards: int,
    device: str | None,
    full_context_eval: bool,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_baseline_checkpoint(checkpoint, device=device)
    model_cfg = ckpt["model_cfg"]
    loss_cfg = BaselineLossConfig(**ckpt["loss_cfg"])
    num_classes = int(model_cfg.get("num_classes", 1))
    use_rest_frame_time = bool(model_cfg.get("use_rest_frame_time", False))
    taxonomy_name = str(ckpt.get("elasticc_taxonomy") or elasticc_taxonomy or "focused")
    ckpt_redshift_source = str(
        ckpt.get("elasticc_redshift_source") or redshift_source or "final"
    )
    ckpt_metadata_fields = list(
        ckpt.get("elasticc_metadata_fields") or LEGACY_META_FIELDS
    )
    _, class_names, _ = get_elasticc_taxonomy(taxonomy_name)
    if elasticc_taxonomy is not None and elasticc_taxonomy != taxonomy_name:
        raise click.ClickException(
            f"checkpoint taxonomy is {taxonomy_name}, but --elasticc-taxonomy {elasticc_taxonomy} was requested"
        )
    if redshift_source is not None and redshift_source != ckpt_redshift_source:
        raise click.ClickException(
            f"checkpoint redshift source is {ckpt_redshift_source}, but --redshift-source {redshift_source} was requested"
        )

    if lazy_elasticc:
        refs, class_names, _ = scan_elasticc_index(
            data_dir,
            taxonomy=taxonomy_name,
            redshift_source=ckpt_redshift_source,
            metadata_fields=ckpt_metadata_fields,
            max_release_dirs=max_release_dirs,
            max_shards_per_release=max_shards_per_release,
            max_objects_per_release=max_objects_per_release,
        )
        if not refs:
            raise click.ClickException("No ELAsTiCC focus records were indexed.")
        all_labels = [int(ref["target"]) for ref in refs]
        train_idx, val_idx = train_test_split(
            np.arange(len(refs)),
            test_size=val_frac,
            stratify=all_labels,
            random_state=seed,
        )
        train_refs = [refs[int(i)] for i in train_idx]
        val_refs = [refs[int(i)] for i in val_idx]
        train_ds, val_ds = build_lazy_elasticc_datasets(
            train_refs=train_refs,
            val_refs=val_refs,
            use_rest_frame_time=use_rest_frame_time,
            max_cached_shards=max_cached_shards,
        )
    else:
        records, class_names, _ = load_elasticc_records(
            data_dir,
            taxonomy=taxonomy_name,
            redshift_source=ckpt_redshift_source,
            metadata_fields=ckpt_metadata_fields,
            max_release_dirs=max_release_dirs,
            max_shards_per_release=max_shards_per_release,
            max_objects_per_release=max_objects_per_release,
        )
        if not records:
            raise click.ClickException("No ELAsTiCC focus records were loaded.")
        all_labels = [int(r["target"]) for r in records]
        train_idx, val_idx = train_test_split(
            np.arange(len(records)),
            test_size=val_frac,
            stratify=all_labels,
            random_state=seed,
        )
        train_records = [records[int(i)] for i in train_idx]
        val_records = [records[int(i)] for i in val_idx]
        train_ds, val_ds = build_precomputed_baseline_datasets(
            train_records=train_records,
            val_records=val_records,
            use_rest_frame_time=use_rest_frame_time,
        )

    z_min = float(ckpt["z_min"])
    z_max = float(ckpt["z_max"])

    if full_context_eval:
        metrics, rows = collect_full_context_predictions(
            model,
            val_ds,
            batch_size=batch_size,
            z_min=z_min,
            z_max=z_max,
            device=device,
        )
        metrics["recon"] = float("nan")
        metrics["cls"] = float("nan")
        metrics["total"] = float("nan")
    else:
        val_cfg = BaselineCollateConfig(
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            mask_prob=mask_prob,
            mask_prob_min=mask_prob_min,
            mask_prob_max=mask_prob_max,
            min_context_points=min_context_points,
            min_target_points=min_target_points,
        )
        val_loader = make_baseline_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            training=False,
            cfg=val_cfg,
        )
        metrics, rows = collect_epoch_predictions(model, val_loader, loss_cfg, device)

    import pandas as pd

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise click.ClickException(
            "No predictions were collected from the validation split."
        )

    if num_classes == 1:
        raise click.ClickException(
            "evaluate-elasticc-focus-baseline expects a multiclass ELAsTiCC checkpoint."
        )
    if num_classes < len(class_names):
        raise click.ClickException(
            f"checkpoint num_classes={num_classes} is smaller than taxonomy size {len(class_names)} for {taxonomy_name}"
        )

    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    pred_df["pred_class"] = pred_df[prob_cols].to_numpy().argmax(axis=1)
    pred_df["true_class_name"] = pred_df["target"].map(
        lambda x: class_names[int(x)] if int(x) < len(class_names) else str(int(x))
    )
    pred_df["pred_class_name"] = pred_df["pred_class"].map(
        lambda x: class_names[int(x)] if int(x) < len(class_names) else str(int(x))
    )

    y_true = pred_df["target"].to_numpy(dtype=int)
    y_pred = pred_df["pred_class"].to_numpy(dtype=int)
    per_class_rows = _elasticc_per_class_rows(y_true, y_pred, class_names[:num_classes])

    predictions_csv = out_dir / "val_predictions.csv"
    per_class_csv = out_dir / "per_class_metrics.csv"
    metrics_json = out_dir / "metrics.json"

    pred_df.to_csv(predictions_csv, index=False)
    pd.DataFrame(per_class_rows).to_csv(per_class_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    click.echo(
        f"[eval] checkpoint={checkpoint} macro_f1={metrics.get('macro_f1', float('nan')):.4f} "
        f"macro_auroc={metrics.get('macro_auroc', float('nan')):.4f} "
        f"recon={metrics.get('recon', float('nan')):.4f}"
    )
    click.echo(f"[done] predictions -> {predictions_csv}")
    click.echo(f"[done] per-class metrics -> {per_class_csv}")


@main.command("evaluate-elasticc-focus-context-sweep")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option(
    "--context-fraction",
    "context_fractions",
    type=float,
    multiple=True,
    default=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    show_default=True,
)
@click.option(
    "--elasticc-taxonomy", type=click.Choice(ELASTICC_TAXONOMY_NAMES), default=None
)
@click.option(
    "--redshift-source", type=click.Choice(ELASTICC_REDSHIFT_SOURCES), default=None
)
@click.option("--max-release-dirs", type=int, default=None)
@click.option("--max-shards-per-release", type=int, default=None)
@click.option("--max-objects-per-release", type=int, default=None)
@click.option("--device", type=str, default=None)
def evaluate_elasticc_focus_context_sweep(
    data_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    min_context_points: int,
    min_target_points: int,
    context_fractions: tuple[float, ...],
    elasticc_taxonomy: str | None,
    redshift_source: str | None,
    max_release_dirs: int | None,
    max_shards_per_release: int | None,
    max_objects_per_release: int | None,
    device: str | None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_baseline_checkpoint(checkpoint, device=device)
    model_cfg = ckpt["model_cfg"]
    loss_cfg = BaselineLossConfig(**ckpt["loss_cfg"])
    num_classes = int(model_cfg.get("num_classes", 1))
    use_rest_frame_time = bool(model_cfg.get("use_rest_frame_time", False))
    taxonomy_name = str(ckpt.get("elasticc_taxonomy") or elasticc_taxonomy or "focused")
    ckpt_redshift_source = str(
        ckpt.get("elasticc_redshift_source") or redshift_source or "final"
    )
    ckpt_metadata_fields = list(
        ckpt.get("elasticc_metadata_fields") or LEGACY_META_FIELDS
    )
    if num_classes == 1:
        raise click.ClickException(
            "evaluate-elasticc-focus-context-sweep expects a multiclass ELAsTiCC checkpoint."
        )
    if elasticc_taxonomy is not None and elasticc_taxonomy != taxonomy_name:
        raise click.ClickException(
            f"checkpoint taxonomy is {taxonomy_name}, but --elasticc-taxonomy {elasticc_taxonomy} was requested"
        )
    if redshift_source is not None and redshift_source != ckpt_redshift_source:
        raise click.ClickException(
            f"checkpoint redshift source is {ckpt_redshift_source}, but --redshift-source {redshift_source} was requested"
        )

    records, _, _ = load_elasticc_records(
        data_dir,
        taxonomy=taxonomy_name,
        redshift_source=ckpt_redshift_source,
        metadata_fields=ckpt_metadata_fields,
        max_release_dirs=max_release_dirs,
        max_shards_per_release=max_shards_per_release,
        max_objects_per_release=max_objects_per_release,
    )
    if not records:
        raise click.ClickException("No ELAsTiCC focus records were loaded.")
    all_labels = [int(r["target"]) for r in records]
    train_idx, val_idx = train_test_split(
        np.arange(len(records)),
        test_size=val_frac,
        stratify=all_labels,
        random_state=seed,
    )
    train_records = [records[int(i)] for i in train_idx]
    val_records = [records[int(i)] for i in val_idx]
    _, val_ds = build_precomputed_baseline_datasets(
        train_records=train_records,
        val_records=val_records,
        use_rest_frame_time=use_rest_frame_time,
    )

    z_min = float(ckpt["z_min"])
    z_max = float(ckpt["z_max"])
    sweep_rows: list[dict[str, float]] = []

    for frac in context_fractions:
        frac = float(frac)
        if frac <= 0.0 or frac > 1.0:
            raise click.ClickException(
                f"context_fraction must be in (0, 1], got {frac}"
            )
        if np.isclose(frac, 1.0):
            metrics, _ = collect_full_context_predictions(
                model,
                val_ds,
                batch_size=batch_size,
                z_min=z_min,
                z_max=z_max,
                device=device,
            )
            metrics["recon"] = float("nan")
            metrics["cls"] = float("nan")
            metrics["total"] = float("nan")
        else:
            val_cfg = BaselineCollateConfig(
                seed=seed,
                z_min=z_min,
                z_max=z_max,
                min_context_points=min_context_points,
                min_target_points=min_target_points,
                deterministic_val_fraction=frac,
            )
            val_loader = make_baseline_loader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                training=False,
                cfg=val_cfg,
            )
            metrics, _ = collect_epoch_predictions(model, val_loader, loss_cfg, device)
        row = {
            "context_fraction": frac,
            "macro_f1": float(metrics.get("macro_f1", float("nan"))),
            "weighted_f1": float(metrics.get("weighted_f1", float("nan"))),
            "macro_auroc": float(metrics.get("macro_auroc", float("nan"))),
            "top1_acc": float(metrics.get("top1_acc", float("nan"))),
            "recon": float(metrics.get("recon", float("nan"))),
        }
        sweep_rows.append(row)
        click.echo(
            f"[context] frac={frac:.2f} macro_f1={row['macro_f1']:.4f} "
            f"macro_auroc={row['macro_auroc']:.4f} top1_acc={row['top1_acc']:.4f}"
        )

    import pandas as pd

    csv_path = out_dir / "context_sweep.csv"
    json_path = out_dir / "context_sweep.json"
    pd.DataFrame(sweep_rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(sweep_rows, f, indent=2)
    click.echo(f"[done] context sweep -> {csv_path}")


@main.command("evaluate-elasticc-focus-prefix-sweep")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--min-context-points", type=int, default=3, show_default=True)
@click.option("--min-target-points", type=int, default=1, show_default=True)
@click.option(
    "--context-fraction",
    "context_fractions",
    type=float,
    multiple=True,
    default=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    show_default=True,
)
@click.option(
    "--elasticc-taxonomy", type=click.Choice(ELASTICC_TAXONOMY_NAMES), default=None
)
@click.option(
    "--redshift-source", type=click.Choice(ELASTICC_REDSHIFT_SOURCES), default=None
)
@click.option("--max-release-dirs", type=int, default=None)
@click.option("--max-shards-per-release", type=int, default=None)
@click.option("--max-objects-per-release", type=int, default=None)
@click.option("--device", type=str, default=None)
def evaluate_elasticc_focus_prefix_sweep(
    data_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    min_context_points: int,
    min_target_points: int,
    context_fractions: tuple[float, ...],
    elasticc_taxonomy: str | None,
    redshift_source: str | None,
    max_release_dirs: int | None,
    max_shards_per_release: int | None,
    max_objects_per_release: int | None,
    device: str | None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_baseline_checkpoint(checkpoint, device=device)
    model_cfg = ckpt["model_cfg"]
    loss_cfg = BaselineLossConfig(**ckpt["loss_cfg"])
    num_classes = int(model_cfg.get("num_classes", 1))
    use_rest_frame_time = bool(model_cfg.get("use_rest_frame_time", False))
    taxonomy_name = str(ckpt.get("elasticc_taxonomy") or elasticc_taxonomy or "focused")
    ckpt_redshift_source = str(
        ckpt.get("elasticc_redshift_source") or redshift_source or "final"
    )
    ckpt_metadata_fields = list(
        ckpt.get("elasticc_metadata_fields") or LEGACY_META_FIELDS
    )
    if num_classes == 1:
        raise click.ClickException(
            "evaluate-elasticc-focus-prefix-sweep expects a multiclass ELAsTiCC checkpoint."
        )
    if elasticc_taxonomy is not None and elasticc_taxonomy != taxonomy_name:
        raise click.ClickException(
            f"checkpoint taxonomy is {taxonomy_name}, but --elasticc-taxonomy {elasticc_taxonomy} was requested"
        )
    if redshift_source is not None and redshift_source != ckpt_redshift_source:
        raise click.ClickException(
            f"checkpoint redshift source is {ckpt_redshift_source}, but --redshift-source {redshift_source} was requested"
        )

    records, _, _ = load_elasticc_records(
        data_dir,
        taxonomy=taxonomy_name,
        redshift_source=ckpt_redshift_source,
        metadata_fields=ckpt_metadata_fields,
        max_release_dirs=max_release_dirs,
        max_shards_per_release=max_shards_per_release,
        max_objects_per_release=max_objects_per_release,
    )
    if not records:
        raise click.ClickException("No ELAsTiCC focus records were loaded.")
    all_labels = [int(r["target"]) for r in records]
    train_idx, val_idx = train_test_split(
        np.arange(len(records)),
        test_size=val_frac,
        stratify=all_labels,
        random_state=seed,
    )
    train_records = [records[int(i)] for i in train_idx]
    val_records = [records[int(i)] for i in val_idx]
    _, val_ds = build_precomputed_baseline_datasets(
        train_records=train_records,
        val_records=val_records,
        use_rest_frame_time=use_rest_frame_time,
    )

    z_min = float(ckpt["z_min"])
    z_max = float(ckpt["z_max"])
    object_ids = list(val_ds.object_ids)
    sweep_rows: list[dict[str, float]] = []
    model.eval()

    for frac in context_fractions:
        frac = float(frac)
        if frac <= 0.0 or frac > 1.0:
            raise click.ClickException(
                f"context_fraction must be in (0, 1], got {frac}"
            )

        if np.isclose(frac, 1.0):
            metrics, _ = collect_full_context_predictions(
                model,
                val_ds,
                batch_size=batch_size,
                z_min=z_min,
                z_max=z_max,
                device=device,
            )
            metrics["total"] = float("nan")
            metrics["recon"] = float("nan")
            metrics["cls"] = float("nan")
        else:
            logits, labels = [], []
            loss_rows = []
            with torch.no_grad():
                for start in range(0, len(object_ids), batch_size):
                    batch_ids = object_ids[start : start + batch_size]
                    items = []
                    for oid in batch_ids:
                        item = val_ds.data[oid].copy()
                        item["oid"] = oid
                        items.append(item)
                    batch = make_prefix_context_batch(
                        items,
                        z_min=z_min,
                        z_max=z_max,
                        context_fraction=frac,
                        min_context_points=min_context_points,
                        min_target_points=min_target_points,
                    ).to(device)
                    output = model(batch)
                    losses = baseline_loss(output, batch, loss_cfg)
                    if np.isfinite(float(losses.total.detach().cpu().item())):
                        loss_rows.append(
                            {
                                "total": float(losses.total.item()),
                                "recon": float(losses.recon.item()),
                                "cls": float(losses.cls.item()),
                            }
                        )
                    logits.append(output.class_logits.detach().cpu())
                    labels.append(batch.labels.detach().cpu())

            all_logits = torch.cat(logits)
            all_labels = torch.cat(labels)
            metrics = evaluate_multiclass_predictions(all_logits, all_labels)
            if loss_rows:
                metrics["total"] = float(np.mean([row["total"] for row in loss_rows]))
                metrics["recon"] = float(np.mean([row["recon"] for row in loss_rows]))
                metrics["cls"] = float(np.mean([row["cls"] for row in loss_rows]))
            else:
                metrics["total"] = float("nan")
                metrics["recon"] = float("nan")
                metrics["cls"] = float("nan")
        row = {
            "context_fraction": frac,
            "macro_f1": float(metrics.get("macro_f1", float("nan"))),
            "weighted_f1": float(metrics.get("weighted_f1", float("nan"))),
            "macro_auroc": float(metrics.get("macro_auroc", float("nan"))),
            "top1_acc": float(metrics.get("top1_acc", float("nan"))),
            "recon": float(metrics.get("recon", float("nan"))),
        }
        sweep_rows.append(row)
        click.echo(
            f"[prefix] frac={frac:.2f} macro_f1={row['macro_f1']:.4f} "
            f"macro_auroc={row['macro_auroc']:.4f} top1_acc={row['top1_acc']:.4f}"
        )

    import pandas as pd

    csv_path = out_dir / "prefix_sweep.csv"
    json_path = out_dir / "prefix_sweep.json"
    pd.DataFrame(sweep_rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(sweep_rows, f, indent=2)
    click.echo(f"[done] prefix sweep -> {csv_path}")


@main.command("predict-mallorn-baseline-test")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-csv", type=click.Path(dir_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
def predict_mallorn_baseline_test(
    data_dir: Path,
    checkpoint: Path,
    out_csv: Path,
    batch_size: int,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_baseline_checkpoint(checkpoint, device=device)
    flux_center_by_band = ckpt["flux_center_by_band"]
    flux_scale_by_band = ckpt["flux_scale_by_band"]
    z_min = float(ckpt["z_min"])
    z_max = float(ckpt["z_max"])
    _, _, test_ds, _ = prepare_mallorn_baseline_test_dataset(
        data_dir,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
        max_obs=max_obs,
        keep_all_snr_gt=keep_all_snr_gt,
        use_rest_frame_time=bool(
            ckpt.get("model_cfg", {}).get("use_rest_frame_time", False)
        ),
    )
    rows = predict_full_context(
        model, test_ds, batch_size=batch_size, z_min=z_min, z_max=z_max, device=device
    )
    threshold = ckpt.get("metrics", {}).get("best_threshold", 0.5)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df["prediction"] = (df["prob_tde"] >= float(threshold)).astype(int)
    df[["object_id", "prediction"]].to_csv(out_csv, index=False)
    click.echo(f"[done] submission -> {out_csv} threshold={float(threshold):.4f}")


@main.command("predict-mallorn-baseline-test-ensemble")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--cv-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-csv", type=click.Path(dir_okay=False, path_type=Path), required=True
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--threshold", type=float, default=None)
@click.option("--device", type=str, default=None)
def predict_mallorn_baseline_test_ensemble(
    data_dir: Path,
    cv_dir: Path,
    out_csv: Path,
    batch_size: int,
    max_obs: int,
    keep_all_snr_gt: float,
    threshold: float | None,
    device: str | None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fold_paths = sorted(cv_dir.glob("fold_*_checkpoint.pt"))
    if not fold_paths:
        raise click.ClickException(f"No fold checkpoints found in {cv_dir}")

    summary_path = cv_dir / "crossval_summary.json"
    if threshold is None and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        threshold = float(summary.get("oof_best_threshold_global", 0.5))
    elif threshold is None:
        threshold = 0.5

    ensemble_rows = None
    for fold_path in fold_paths:
        model, ckpt = load_baseline_checkpoint(fold_path, device=device)
        flux_center_by_band = ckpt["flux_center_by_band"]
        flux_scale_by_band = ckpt["flux_scale_by_band"]
        z_min = float(ckpt["z_min"])
        z_max = float(ckpt["z_max"])
        use_rest_frame_time = bool(
            ckpt.get("model_cfg", {}).get("use_rest_frame_time", False)
        )
        _, _, test_ds, _ = prepare_mallorn_baseline_test_dataset(
            data_dir,
            flux_center_by_band=flux_center_by_band,
            flux_scale_by_band=flux_scale_by_band,
            max_obs=max_obs,
            keep_all_snr_gt=keep_all_snr_gt,
            use_rest_frame_time=use_rest_frame_time,
        )
        rows = predict_full_context(
            model,
            test_ds,
            batch_size=batch_size,
            z_min=z_min,
            z_max=z_max,
            device=device,
        )
        if ensemble_rows is None:
            ensemble_rows = [
                {"object_id": str(row["object_id"]), "prob_tde": float(row["prob_tde"])}
                for row in rows
            ]
        else:
            if len(rows) != len(ensemble_rows):
                raise click.ClickException(
                    f"Fold {fold_path.name} produced {len(rows)} rows, expected {len(ensemble_rows)}"
                )
            for acc, row in zip(ensemble_rows, rows):
                if str(acc["object_id"]) != str(row["object_id"]):
                    raise click.ClickException(
                        f"Object order mismatch in {fold_path.name}"
                    )
                acc["prob_tde"] += float(row["prob_tde"])

    assert ensemble_rows is not None
    n_folds = len(fold_paths)
    for row in ensemble_rows:
        row["prob_tde"] /= n_folds

    import pandas as pd

    df = pd.DataFrame(ensemble_rows)
    df["prediction"] = (df["prob_tde"] >= float(threshold)).astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[["object_id", "prediction"]].to_csv(out_csv, index=False)
    click.echo(
        f"[done] ensemble submission -> {out_csv} folds={n_folds} threshold={float(threshold):.4f}"
    )
