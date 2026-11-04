# -*- coding: utf-8 -*-
"""Minimal inference wrapper for the project's xLSTM-Informer model.

This wrapper mirrors the configuration and forward input construction in
run_model_comparison_goldimpact_formal.py. It requires the ORIGINAL project
models/Informer.py and its imported layers (not included in this package).

Model output: predicted NEXT-DAY PRICE DELTA in the target's training-scaled
space, not a directly usable gold price. Use the original training y_scaler
(mean and scale) to restore price differences before adding current close.

No silent fallback to an LSTM if the xLSTM-Informer code is unavailable.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int = 42
    seq_len: int = 20
    label_len: int = 10
    pred_len: int = 1
    freq: str = "d"
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.10
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 128
    factor: int = 5
    activation: str = "gelu"
    channel_independence: int = 0
    top_k: int = 5
    num_kernels: int = 6
    distil: int = 1
    embed: str = "timeF"
    output_attention: int = 0
    task_name: str = "short_term_forecast"

    @property
    def enc_in(self):
        return self.input_dim

    @property
    def dec_in(self):
        return self.input_dim

    @property
    def c_out(self):
        return 1


class GoldXLSTMInformer(nn.Module):
    """Forward input: x [batch, seq_len, input_dim]; output: [batch].

    The wrapped project model must produce exactly input_dim values per row
    after flattening, matching the linear output head used by the experiments.
    """
    def __init__(self, input_dim=42, seq_len=20):
        super().__init__()
        if input_dim < 1 or seq_len < 2:
            raise ValueError("input_dim must be >= 1 and seq_len >= 2")
        try:
            from models import Informer
        except Exception as exc:
            raise ImportError(
                "Cannot import models.Informer. Copy the ORIGINAL project's "
                "models/Informer.py and all required local dependencies "
                "(e.g., layers/) into this repository."
            ) from exc
        self.config = ModelConfig(input_dim=input_dim, seq_len=seq_len,
                                  label_len=seq_len // 2)
        self.base_model = Informer.Model(self.config)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.config.seq_len, self.config.input_dim):
            raise ValueError(
                f"Expected x [batch, {self.config.seq_len}, {self.config.input_dim}], "
                f"received {tuple(x.shape)}"
            )
        batch = x.shape[0]
        markers_enc = x.new_zeros((batch, self.config.seq_len, 4))
        dec_len = self.config.label_len + self.config.pred_len
        dec_in = x.new_zeros((batch, dec_len, self.config.input_dim))
        dec_in[:, :self.config.label_len, :] = x[:, -self.config.label_len:, :]
        markers_dec = x.new_zeros((batch, dec_len, 4))

        out = self.base_model(x, markers_enc, dec_in, markers_dec)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Informer backbone returned {type(out).__name__}, expected Tensor")
        out = out.reshape(batch, -1)
        if out.shape[1] != self.config.input_dim:
            raise RuntimeError(
                f"Unexpected Informer output: {tuple(out.shape)}. "
                f"Original experimental output head expects "
                f"{self.config.input_dim} features per batch row. "
                "Check that models/Informer.py matches the project version."
            )
        return self.fc_out(out).squeeze(-1)


def restore_next_close(current_close, scaled_delta, target_mean, target_scale):
    """Convert standardized next-day delta into next closing price.

    All arguments may be Python floats or broadcastable PyTorch tensors.
    target_mean and target_scale MUST come from the saved training y_scaler.
    """
    return current_close + scaled_delta * target_scale + target_mean
