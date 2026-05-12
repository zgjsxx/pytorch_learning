"""
nn.Dropout 学习 demo
模拟 GPT 中的 drop = nn.Dropout(config.dropout)
"""

import torch
import torch.nn as nn

# ===== 1. 基本概念 =====
# Dropout: 训练时随机把一部分神经元置为0，防止过拟合
# config.dropout = 0.1 表示每个神经元有10%的概率被丢弃

config_dropout = 0.5  # 用0.5更直观，50%被丢弃

drop = nn.Dropout(config_dropout)
print(f"Dropout rate: {config_dropout}")
print()

# ===== 2. 训练模式 vs 评估模式 =====
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(f"原始输入: {x.tolist()}")

# 训练模式 (默认) — 会随机丢弃
drop.train()
x_train = drop(x)
print(f"训练模式输出: {x_train.tolist()}")
print(f"  注意: 被丢弃的位置变成了0, 其余位置被放大了(乘以1/(1-p)=1/0.5=2)")
print(f"  放大原因: 保持整体期望值不变, 否则训练和推理的数值尺度就不一致了")
print()

# 评估模式 — 不丢弃，原样输出
drop.eval()
x_eval = drop(x)
print(f"评估模式输出: {x_eval.tolist()}")
print(f"  推理时不需要Dropout, 直接原样输出")
print()

# ===== 3. 数学原理: 为什么没被丢弃的值要放大 =====
# 训练时: 每个值有 p 的概率变成0, (1-p) 的概率变成 x * (1/(1-p))
# 期望值: E = p * 0 + (1-p) * x * (1/(1-p)) = x
# 所以期望值和不用Dropout时一样, 推理时直接用原始值就行

x = torch.ones(10000)  # 10000个1
drop.train()
out = drop(x)
print(f"10000个1经过Dropout(0.5)后的均值: {out.mean().item():.4f}")
print(f"  均值接近1.0, 因为放大系数补偿了丢弃带来的损失")
print()

# ===== 4. 多次运行看随机性 =====
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
drop.train()
print("同一输入, 5次Dropout输出:")
for i in range(5):
    print(f"  第{i+1}次: {drop(x).tolist()}")
print(f"  每次丢弃的位置不同, 所以输出不同")
print()

# ===== 5. 在 GPT 中的使用 =====
# Dropout 通常加在 Embedding / Attention / MLP 之后
vocab_size = 10
n_embd = 4
dropout = 0.1  # GPT常用的小dropout率

wte = nn.Embedding(vocab_size, n_embd)
drop = nn.Dropout(dropout)

input_ids = torch.tensor([1, 3, 5])
x = wte(input_ids)        # (3, 4) token embedding
x = drop(x)               # 训练时随机丢弃10%的位置

print(f"GPT流程: input_ids -> Embedding -> Dropout")
print(f"  Embedding输出: {wte(input_ids)}")
print(f"  Dropout输出:   {x}")