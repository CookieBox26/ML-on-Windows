import torch
import torch.nn as nn
from math import sqrt


class TestTorchTensorMaskedFill:

    def test_masked_fill_(self):
        # 以下のセルフアテンションをマスクする．
        scores = torch.tensor([[
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
            ]
        ]])

        # まず未来へのアテンションが True になる行列を用意する．
        mask = torch.triu(torch.ones([1, 1, 3, 3], dtype=torch.bool), diagonal=1)
        assert torch.all(torch.eq(
            mask,
            torch.tensor([[
                [False, True, True],
                [False, False, True],
                [False, False, False],
            ]])
        ))

        # 以下のようにすると未来へのアテンションを -inf にできる．
        scores.masked_fill_(mask, float('-inf'))
        assert torch.all(torch.eq(
            scores,
            torch.tensor([[
                [1., float('-inf'), float('-inf')],
                [4., 5., float('-inf')],
                [7., 8., 9.],
            ]])
        ))

        # ついでに正規化とドロップアウトまでするとこう．
        attn = torch.softmax((1./sqrt(3)) * scores, dim=-1)
        # print(attn)  # マスク箇所が 0.0 になる．
        dropout = nn.Dropout(0.5)
        attn = dropout(attn)
        # print(attn)  # 50% の成分が 0.0 になり生き残った成分が 2 倍になる．

        # ついでにセルフアテンションの適用までするとこう．
        v = torch.tensor([[
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.5, 0.6, 0.7, 0.8]],
            [[0.1, 0.1, 0.1, 0.1]],
        ]])
        v = torch.einsum('bhls,bshd->blhd', attn, v)
