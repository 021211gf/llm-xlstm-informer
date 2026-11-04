# -*- coding: utf-8 -*-
"""Architecture smoke test, NOT an accuracy evaluation or trained inference."""
import torch
from gold_predictor import GoldXLSTMInformer


def main():
    torch.manual_seed(2020)
    seq_len, feature_dim, batch = 20, 42, 2
    model = GoldXLSTMInformer(input_dim=feature_dim, seq_len=seq_len)
    model.eval()
    # Example values are random standardized FEATURES, not real gold data.
    example_input = torch.randn(batch, seq_len, feature_dim)
    with torch.no_grad():
        scaled_delta = model(example_input)
    assert scaled_delta.shape == (batch,)
    print("Model forward OK; standardized delta output shape:", tuple(scaled_delta.shape))
    print("NOTE: Weights are RANDOM. These outputs are NOT forecasts.")
    print("For real inference: load a trained checkpoint, reproduce training feature")
    print("ordering/scaling, then invert y_scaler and add today's close price.")


if __name__ == "__main__":
    main()
