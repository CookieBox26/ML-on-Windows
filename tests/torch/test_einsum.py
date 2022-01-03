import torch


class TestTorchEinsum:

    def test_einsum(self):
        """ torch.einsum の結果を確認する．
        """
        # 簡単な例．ただの行列積．
        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        b = torch.tensor([[1., 2.], [3., 4.], [5., 6,]])
        c = torch.einsum('ij,jk->ik', a, b)
        # ただの行列積に一致することを確認．
        assert torch.all(torch.eq(c, torch.matmul(a, b)))
        # 手で検算．
        c_ = torch.zeros(a.shape[0], b.shape[1])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(b.shape[1]):
                    c_[i][k] += a[i][j] * b[j][k]
        assert torch.all(torch.eq(c, c_))

        # バッチサイズ1, 系列長3, ヘッド数2, 特徴次元数4 のセルフアテンションの例．
        # 'blhe,bshe->bhls' でサイズ [1, 2, 3, 3] のテンソルになる．
        # [b, h, l, s] 成分の意味は，「バッチ内 b 番目のデータの h ヘッド目の
        # l 単語目から s 単語目へのセルフアテンション（次元のルートで割ってソフトマックスする前）．
        q = torch.tensor([
            [
                [
                    [1., 2., 3., 4.],  # 1単語目の1ヘッド目のクエリ特徴
                    [5., 6., 7., 8.],  # 1単語目の2ヘッド目のクエリ特徴
                ],
                [
                    [10., 20., 30., 40.],  # 2単語目の1ヘッド目のクエリ特徴
                    [50., 60., 70., 80.],  # 2単語目の2ヘッド目のクエリ特徴
                ],
                [
                    [100., 200., 300., 400.],  # 3単語目の1ヘッド目のクエリ特徴
                    [500., 600., 700., 800.],  # 3単語目の2ヘッド目のクエリ特徴
                ],
            ],
        ])
        k = torch.tensor([
            [
                [
                    [0.1, 0.2, 0.3, 0.4],  # 1単語目の1ヘッド目のキー特徴
                    [0.1, 0.2, 0.3, 0.4],  # 1単語目の2ヘッド目のキー特徴
                ],
                [
                    [0.1, 0.0, 0.0, 0.0],  # 2単語目の1ヘッド目のキー特徴
                    [0.0, 0.1, 0.0, 0.0],  # 2単語目の2ヘッド目のキー特徴
                ],
                [
                    [0.0, 0.0, 0.1, 0.0],  # 3単語目の1ヘッド目のキー特徴
                    [0.0, 0.0, 0.0, 0.1],  # 3単語目の2ヘッド目のキー特徴
                ],
            ],
        ])
        scores = torch.einsum('blhe,bshe->bhls', q, k)
        # バッチ内 0 番目のデータの 0 ヘッド目のセルフアテンションを転置で計算するならこう．
        assert torch.all(
            torch.eq(
                scores[0][0],
                torch.matmul(torch.transpose(q[0], 0, 1)[0], torch.transpose(torch.transpose(k[0], 0, 1)[0], 0, 1))
            )
        )
        # 手で検算．
        scores_ = torch.zeros(q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        for b in range(q.shape[0]):
            for l in range(q.shape[1]):
                for h in range(q.shape[2]):
                    for e in range(q.shape[3]):
                        for s in range(k.shape[1]):
                            scores_[b][h][l][s] += q[b][l][h][e] * k[b][s][h][e]
        assert torch.all(torch.eq(scores, scores_))
