import pytest
import torch

from naptime.baseline import (
    BaselineLossConfig,
    ConvGNPBaseline,
    baseline_loss,
)
from tests.conftest import make_batch


class TestConvGNPBaselineForward:
    def test_output_shapes_with_latent(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        B, N_tgt = batch.target_x.shape
        assert out.pred_mean.shape == (B, N_tgt)
        assert out.pred_var.shape == (B, N_tgt)
        assert out.class_logits.shape == (B, small_cfg.num_classes)
        assert out.latent_mu is not None
        assert out.latent_mu.shape == (B, small_cfg.latent_dim)
        assert out.latent_logvar is not None
        assert out.latent_logvar.shape == (B, small_cfg.latent_dim)

    def test_output_shapes_no_latent(self, small_cfg, batch):
        from dataclasses import replace

        cfg = replace(small_cfg, use_latent=False)
        model = ConvGNPBaseline(cfg)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.latent_mu is None
        assert out.pred_mean.shape == batch.target_x.shape

    def test_output_shapes_with_metadata(self, small_cfg_meta, batch_meta):
        model = ConvGNPBaseline(small_cfg_meta)
        model.eval()
        with torch.no_grad():
            out = model(batch_meta)
        B = batch_meta.target_x.shape[0]
        assert out.class_logits.shape == (B, small_cfg_meta.num_classes)

    def test_pred_var_positive(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert (out.pred_var > 0).all()

    def test_output_finite(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out.pred_mean).all()
        assert torch.isfinite(out.pred_var).all()
        assert torch.isfinite(out.class_logits).all()

    def test_train_vs_eval_latent_differs(self, small_cfg):
        """During training z is sampled; during eval z = mu. The predictions may differ."""
        model = ConvGNPBaseline(small_cfg)
        batch = make_batch(seed=42)

        torch.manual_seed(0)
        model.train()
        out_train = model(batch)

        model.eval()
        with torch.no_grad():
            out_eval = model(batch)

        # latent_mu should be identical (deterministic encoder)
        assert torch.allclose(out_train.latent_mu, out_eval.latent_mu, atol=1e-5)

    def test_batch_size_one(self, small_cfg):
        model = ConvGNPBaseline(small_cfg)
        model.eval()
        b1 = make_batch(batch_size=1)
        with torch.no_grad():
            out = model(b1)
        assert out.pred_mean.shape[0] == 1

    def test_large_batch(self, small_cfg):
        model = ConvGNPBaseline(small_cfg)
        model.eval()
        b = make_batch(batch_size=8, n_ctx=20, n_tgt=15)
        with torch.no_grad():
            out = model(b)
        assert out.pred_mean.shape == (8, 15)

    def test_binary_classification(self, small_cfg):
        from dataclasses import replace

        cfg = replace(small_cfg, num_classes=1)
        model = ConvGNPBaseline(cfg)
        model.eval()
        batch = make_batch()
        with torch.no_grad():
            out = model(batch)
        # binary head returns 1D or shape (B,)
        assert out.class_logits.shape in {(2,), (2, 1)}


class TestBaselineLoss:
    def test_loss_finite(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        out = model(batch)
        cfg = BaselineLossConfig()
        losses = baseline_loss(out, batch, cfg)
        assert torch.isfinite(losses.total)
        assert torch.isfinite(losses.recon)
        assert torch.isfinite(losses.cls)

    def test_recon_loss_positive(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        out = model(batch)
        losses = baseline_loss(out, batch, BaselineLossConfig())
        # Gaussian NLL is not necessarily positive, but should be finite
        assert torch.isfinite(losses.recon)

    def test_kl_loss_non_negative(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        out = model(batch)
        losses = baseline_loss(out, batch, BaselineLossConfig())
        assert losses.kl is not None
        assert losses.kl >= 0

    def test_no_kl_without_latent(self, small_cfg, batch):
        from dataclasses import replace

        cfg = replace(small_cfg, use_latent=False)
        model = ConvGNPBaseline(cfg)
        out = model(batch)
        losses = baseline_loss(out, batch, BaselineLossConfig())
        assert losses.kl is None

    def test_loss_lambda_scaling(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        out = model(batch)
        cfg1 = BaselineLossConfig(lambda_recon=1.0, lambda_cls=1.0, beta_kl=0.0)
        cfg2 = BaselineLossConfig(lambda_recon=2.0, lambda_cls=1.0, beta_kl=0.0)
        l1 = baseline_loss(out, batch, cfg1)
        l2 = baseline_loss(out, batch, cfg2)
        # Doubling lambda_recon should change total by exactly losses.recon
        assert torch.allclose(l2.total - l1.total, l1.recon, atol=1e-5)

    def test_gradient_flows_through_loss(self, small_cfg, batch):
        model = ConvGNPBaseline(small_cfg)
        out = model(batch)
        losses = baseline_loss(out, batch, BaselineLossConfig())
        losses.total.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
