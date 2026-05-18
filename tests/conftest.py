"""Shared fixtures for the naptime test suite."""

import pytest
import torch

from naptime.baseline import BaselineBatch, ConvGNPBaselineConfig


@pytest.fixture
def small_cfg() -> ConvGNPBaselineConfig:
    """Minimal ConvGNPBaselineConfig for fast CPU tests."""
    return ConvGNPBaselineConfig(
        num_bands=6,
        grid_size=32,
        band_emb_dim=4,
        time_fourier_dim=4,
        point_feat_dim=16,
        grid_feat_dim=16,
        conv_hidden_dim=16,
        conv_layers=2,
        conv_kernel_size=3,
        classifier_hidden_dim=16,
        decoder_hidden_dim=16,
        setconv_sigmas=(0.05, 0.15),
        use_redshift=True,
        use_latent=True,
        latent_dim=4,
        latent_hidden_dim=16,
        use_metadata=False,
        metadata_dim=0,
        num_classes=3,
    )


@pytest.fixture
def small_cfg_meta(small_cfg) -> ConvGNPBaselineConfig:
    """Config with metadata enabled."""
    from dataclasses import replace

    return replace(small_cfg, use_metadata=True, metadata_dim=5, metadata_embed_dim=8)


def make_batch(
    batch_size: int = 2,
    n_ctx: int = 8,
    n_tgt: int = 6,
    num_bands: int = 6,
    metadata_dim: int = 0,
    seed: int = 0,
) -> BaselineBatch:
    """Build a synthetic BaselineBatch on CPU."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=rng)

    return BaselineBatch(
        context_x=rand(batch_size, n_ctx),
        context_y=rand(batch_size, n_ctx) * 2 - 1,
        context_yerr=rand(batch_size, n_ctx).clamp_min(0.01) * 0.5,
        context_band=torch.randint(0, num_bands, (batch_size, n_ctx), generator=rng),
        context_mask=torch.ones(batch_size, n_ctx),
        target_x=rand(batch_size, n_tgt),
        target_y=rand(batch_size, n_tgt) * 2 - 1,
        target_yerr=rand(batch_size, n_tgt).clamp_min(0.01) * 0.5,
        target_band=torch.randint(0, num_bands, (batch_size, n_tgt), generator=rng),
        target_mask=torch.ones(batch_size, n_tgt),
        labels=torch.zeros(batch_size, dtype=torch.float32),
        redshift=rand(batch_size),
        t_span_log=rand(batch_size),
        n_obs_log=rand(batch_size),
        meta_values=(
            rand(batch_size, metadata_dim)
            if metadata_dim > 0
            else torch.zeros(batch_size, 0)
        ),
        meta_mask=(
            torch.ones(batch_size, metadata_dim)
            if metadata_dim > 0
            else torch.zeros(batch_size, 0)
        ),
        object_ids=[f"obj{i}" for i in range(batch_size)],
    )


@pytest.fixture
def batch() -> BaselineBatch:
    return make_batch()


@pytest.fixture
def batch_meta() -> BaselineBatch:
    return make_batch(metadata_dim=5)
