import torch 
import torch.nn as nn
import math
import pytest


@pytest.fixture
def expected_selfattn():
    """
    MultiheadAttention を 512 次元, 8 ヘッドでインスタンス化して
    5 単語のランダムベクトル列を渡したときに期待する平均セルフアテンション
    """
    return torch.tensor([
        [0.1947, 0.1986, 0.2030, 0.2056, 0.1981],
        [0.1910, 0.1970, 0.1931, 0.2125, 0.2063],
        [0.1933, 0.1920, 0.1976, 0.2081, 0.2090],
        [0.1961, 0.1938, 0.2018, 0.2071, 0.2011],
        [0.1944, 0.1993, 0.1973, 0.2054, 0.2035]
    ])


class TestTorchNnMultiheadAttention:

    def test_forward(self, expected_selfattn):
        """
        まずは MultiheadAttention でセルフアテンションを計算する
        """
        # MultiheadAttention をインスタンス化する
        # 512 次元に埋め込んだベクトルの配列を 8 ヘッドでさばく
        torch.manual_seed(1)  # シードは固定する
        self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        assert list(self_attn.in_proj_weight.size()) == [1536, 512]
        assert list(self_attn.in_proj_bias.size()) == [1536]
        assert list(self_attn.out_proj.weight.size()) == [512, 512]
        assert list(self_attn.out_proj.bias.size()) == [512]

        # 5 単語のランダムベクトル列を用意する
        torch.manual_seed(1)  # シードは固定する
        x = torch.rand(5, 1, 512)

        # セルフアテンションする
        out = self_attn(x, x, x)
        assert list(out[0].size()) == [5, 1, 512]
        assert list(out[1].size()) == [1, 5, 5]
        # 8 ヘッドあるのに1つのセルフアテンションが出てくるのが不思議だが全ヘッドの平均である
        # https://github.com/pytorch/pytorch/blob/v1.10.1/torch/nn/functional.py#L5108

        # 平均セルフアテンションが期待する値になっていることを確認する
        assert torch.allclose(out[1][0], expected_selfattn, atol=0.0001)

    def test_forward_manual(self, expected_selfattn):
        """
        次に MultiheadAttention をつかわずに同じセルフアテンションを求める
        """
        # MultiheadAttention をインスタンス化する
        torch.manual_seed(1)  # シードは固定する
        self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # 重みだけコピーする
        w_in = self_attn.in_proj_weight.detach().clone()
        b_in = self_attn.in_proj_bias.detach().clone()
        w_out = self_attn.out_proj.weight.detach().clone()
        b_out = self_attn.out_proj.bias.detach().clone()
        
        # 5 単語のランダムベクトル列を用意する
        torch.manual_seed(1)  # シードは固定する
        x = torch.rand(5, 1, 512)

        # いい方法がわからないので 1 単語ずつ 1536 次元に写像
        x = x.squeeze()
        z0 = torch.matmul(w_in, x[0]) + b_in
        z1 = torch.matmul(w_in, x[1]) + b_in
        z2 = torch.matmul(w_in, x[2]) + b_in
        z3 = torch.matmul(w_in, x[3]) + b_in
        z4 = torch.matmul(w_in, x[4]) + b_in
        z = torch.stack((z0, z1, z2, z3, z4), 1)  # [1536, 5] にたばねる
        q, k, v = z.chunk(3)  # 3分割する
        assert list(q.size()) == [512, 5]
        assert list(k.size()) == [512, 5]
        assert list(v.size()) == [512, 5]

        # さらに8ヘッドに分割（1ヘッドあたりのモデル次元 64）
        q = q.contiguous().view(8, 64, 5)
        q = torch.transpose(q, 1, 2)
        k = k.contiguous().view(8, 64, 5)
        k = torch.transpose(k, 1, 2)
        v = v.contiguous().view(8, 64, 5)
        v = torch.transpose(v, 1, 2)
        assert list(q.size()) == [8, 5, 64]
        assert list(k.size()) == [8, 5, 64]
        assert list(v.size()) == [8, 5, 64]

        # セルフアテンション行列を計算する
        q = q / math.sqrt(64)
        attn = torch.bmm(q, k.transpose(-2, -1))  # バッチごとに行列積
        attn = attn.softmax(dim=2)
        assert list(attn.size()) == [8, 5, 5]

        # 平均セルフアテンションが期待する値になっていることを確認する
        attn = attn.sum(dim=0) / 8.0
        assert list(attn.size()) == [5, 5]
        assert torch.allclose(attn, expected_selfattn, atol=0.0001)
