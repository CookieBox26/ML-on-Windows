import torch
import torch.nn as nn


class TestTorchNnTransformer:

    def test_init(self):
        model = nn.Transformer()

        for layer in model.encoder.layers:
            assert list(layer.self_attn.in_proj_weight.size()) == [1536, 512]
            assert list(layer.self_attn.in_proj_bias.size()) == [1536]
            assert list(layer.self_attn.out_proj.weight.size()) == [512, 512]
            assert list(layer.self_attn.out_proj.bias.size()) == [512]
            assert list(layer.linear1.weight.size()) == [2048, 512]
            assert list(layer.linear1.bias.size()) == [2048]
            assert list(layer.linear2.weight.size()) == [512, 2048]
            assert list(layer.linear2.bias.size()) == [512]

        src = torch.ones(6, 1, 512)
        tgt = torch.ones(5, 1, 512)
        out = model(src, tgt)

    def test_init_(self):
        model = nn.Transformer(
            d_model=16,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=32,
            dropout=0.0,
        )
