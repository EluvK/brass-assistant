"""教程 00 章的配套脚本:亲手跑一遍"训练神经网络"的最小闭环。

不依赖本项目的任何代码,只需要 PyTorch。包含两个各约 20 行的实验:

  A. 回归(拟合 sin 曲线) —— 对应本项目 rank 头的 MSE 损失
  B. 分类(3 选 1)      —— 对应 policy / winner 头的 softmax + 交叉熵损失

运行(仓库根目录):

    ./.venv/Scripts/python.exe docs/tutorial/toy_nn.py

读完 00 章后,建议回来做三件事:
  1. 把 A 的 lr 改成 0.05 / 0.00001,观察收敛变快还是变慢甚至发散;
  2. 把 A 的网络砍到只剩一层 Linear(1,1),观察它还能不能拟合 sin(为什么不能);
  3. 在 B 的 loss.backward() 前后各打印一次 clf 第 0 层权重,确认"参数真的动了"。
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


def part_a_regression() -> None:
    """A: 回归 —— 学 y = sin(x)。对应 rank 头:输出连续值,用 MSE 衡量误差。"""
    model = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.linspace(-3.14159, 3.14159, 256).unsqueeze(1)  # (256, 1) 输入
    y = torch.sin(x)                                         # (256, 1) 标准答案

    for step in range(2000):
        optimizer.zero_grad()                 # 1) 清掉上一轮算出的梯度
        pred = model(x)                       # 2) 前向:用当前参数做预测
        loss = ((pred - y) ** 2).mean()       # 3) MSE:这次错了多少(标量)
        loss.backward()                       # 4) 反向:自动求出每个参数的梯度
        optimizer.step()                      # 5) 更新:每个参数朝降低 loss 的方向挪一步
        if step % 500 == 0:
            print(f"A  step {step:4d}  mse = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():                     # 推理不需要梯度,不建计算图
        max_err = (model(x) - y).abs().max().item()
    print(f"A  完成:最大误差 |sin(x) - model(x)| = {max_err:.4f}\n")


def part_b_classification() -> None:
    """B: 分类 —— 3 个候选动作选 1。对应 policy/winner 头:输出概率分布,用交叉熵。"""
    n, classes = 512, 3
    features = torch.randn(n, 8)              # 假想的"状态特征"
    # 隐含规则:某个标量分数落在下/中/上三分之一 → 正确动作 0/1/2
    score = features.sum(dim=1) + 0.5 * features[:, 0] ** 2
    boundaries = torch.quantile(score, torch.tensor([1.0 / 3, 2.0 / 3]))
    labels = torch.bucketize(score, boundaries)          # 取值 0/1/2
    one_hot = nn.functional.one_hot(labels, classes).float()  # (512, 3) 标准答案

    clf = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, classes))
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for step in range(2000):
        optimizer.zero_grad()
        logits = clf(features)                            # (512, 3) 每个候选一个原始分
        log_probs = torch.log_softmax(logits, dim=1)      # 原始分 -> 归一化对数概率
        # 交叉熵 = -(标准答案分布 · 预测对数概率) 求和,再对样本取平均。
        # one-hot 时它退化成"只挑正确类别那一项的 -log p"。
        loss = -(one_hot * log_probs).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
            print(f"B  step {step:4d}  ce = {loss.item():.4f}  acc = {acc:.1%}")

    clf.eval()
    with torch.no_grad():
        acc = (clf(features).argmax(dim=1) == labels).float().mean().item()
    print(f"B  完成:accuracy = {acc:.1%}")


if __name__ == "__main__":
    print("== A 回归:拟合 sin(x)(MSE,对应 rank 头)==")
    part_a_regression()
    print("== B 分类:3 选 1(softmax + 交叉熵,对应 policy/winner 头)==")
    part_b_classification()
