import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class MyModel(nn.Module):
    """
    適当なモデル
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


@pytest.fixture
def expected_grad():
    """
    適当なモデルに [0.1, 0.2, 0.3] を入れたときの出力の
    [1.0] との2乗誤差の
    モデル1層目についての勾配
    """
    grad_w1 = torch.tensor([
        [ 0.0000,  0.0000,  0.0000],
        [-0.0156, -0.0312, -0.0468],
        [-0.0043, -0.0087, -0.0130],
        [-0.0112, -0.0223, -0.0335],
        [-0.0083, -0.0166, -0.0249]
    ])
    grad_b1 = torch.tensor(
        [0.0000, -0.1562, -0.0433, -0.1115, -0.0829]
    )
    return {'w1': grad_w1, 'b1': grad_b1}


class TestTorchTensorBackward:

    def test_backward(self, expected_grad):
        """
        まずは tensor.backward で勾配を計算する
        """
        # 適当なモデルをインスタンス化する
        torch.manual_seed(1)  # シードは固定する 
        model = MyModel()

        # 適当な訓練データを1点用意する
        x = torch.tensor([[0.1, 0.2, 0.3]])
        y = torch.tensor([[1.0]])

        # モデルに訓練データを入れたときの2乗誤差の勾配をとる
        a3 = model(x)
        criterion = nn.MSELoss()
        loss = criterion(a3, y)
        model.zero_grad()
        loss.backward()  # loss を大きくする方向を求めよ！

        # 結果的にモデルの1層目の勾配はこうなる
        assert torch.allclose(model.fc1.weight.grad, expected_grad['w1'],
                              atol=0.0001)
        assert torch.allclose(model.fc1.bias.grad, expected_grad['b1'],
                              atol=0.0001)

    def test_manual_backward(self, expected_grad):
        """
        次に tensor.backward をつかわずに同じ勾配を求める
        """
        # 適当なモデルをインスタンス化する
        torch.manual_seed(1)  # シードは固定する 
        model = MyModel()

        # 重みだけコピーする
        w1 = model.fc1.weight.detach().clone()
        b1 = model.fc1.bias.detach().clone()
        w2 = model.fc2.weight.detach().clone()
        b2 = model.fc2.bias.detach().clone()
        w3 = model.fc3.weight.detach().clone()
        b3 = model.fc3.bias.detach().clone()

        # 適当な訓練データを1点用意する
        x = torch.tensor([0.1, 0.2, 0.3])
        y = torch.tensor([1.0])

        # 順伝播する
        z1 = torch.matmul(w1, x) + b1
        a1 = F.relu(z1)
        z2 = torch.matmul(w2, a1) + b2
        a2 = F.relu(z2)
        z3 = torch.matmul(w3, a2) + b3
        a3 = F.relu(z3)

        # 2乗誤差をとる 
        criterion = nn.MSELoss()
        loss = criterion(a3, y)

        # 逆伝播する
        loss_a3 = 2.0 * (a3 - y)
        a3_z3 = torch.heaviside(z3, torch.tensor([0.0]))  # ReLU の微分
        loss_z3 = loss_a3 * a3_z3
        loss_a2 = loss_z3 * w3.squeeze()  # 横ベクトルの w3 を縦ベクトルに転置
        a2_z2 = torch.heaviside(z2, torch.tensor([0.0]))  # ReLU の微分
        loss_z2 = torch.mul(loss_a2, a2_z2)  # アダマール積
        loss_a1 = torch.matmul(torch.transpose(w2, 0, 1), loss_z2)
        a1_z1 = torch.heaviside(z1, torch.tensor([0.0]))  # ReLU の微分
        loss_z1 = torch.mul(loss_a1, a1_z1)  # アダマール積
        loss_b1 = loss_z1
        loss_w1 = torch.matmul(loss_z1.view(5, 1), x.view(1, 3))  # 縦ベクトルの x を横ベクトルに転置
        # torch.matmul の左側にベクトルを渡して通常の行列積をしたいとき
        # 明示的に .view(n, 1) しなければ意図通りにならない

        # モデルの1層目の勾配が正しいことを確認
        assert torch.allclose(loss_w1, expected_grad['w1'], atol=0.0001)
        assert torch.allclose(loss_b1, expected_grad['b1'], atol=0.0001)
