"""
nn.Linear 学习 demo
模拟 GPT 中的 lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
"""

import torch
import torch.nn as nn

# ===== 1. 基本概念 =====
# nn.Linear(in_features, out_features) 是一个线性变换: y = x @ W.T + b
#   - in_features:  输入维度
#   - out_features: 输出维度
#   - W: 权重矩阵, 形状 (out_features, in_features)
#   - b: 偏置向量, 形状 (out_features), bias=False时不使用

# 简单例子: 4维输入 -> 3维输出
linear = nn.Linear(4, 3)
print(f"权重 W 形状: {linear.weight.shape}")    # (3, 4)
print(f"权重 W:\n{linear.weight}")
print(f"偏置 b 形状: {linear.bias.shape}")       # (3)
print(f"偏置 b:\n{linear.bias}")
print()

# ===== 2. 手动计算验证 =====
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = linear(x)
print(f"输入 x: {x.tolist()}")         # 4维
print(f"输出 y: {y.tolist()}")          # 3维

# 手动算: y = x @ W.T + b
y_manual = x @ linear.weight.T + linear.bias
print(f"手动计算 y: {y_manual.tolist()}")
print(f"两者相等: {torch.allclose(y, y_manual)}")
print()

# ===== 3. bias=False 的情况 =====
# GPT 的 lm_head 不使用偏置
linear_no_bias = nn.Linear(4, 3, bias=False)
print(f"bias=False 时:")
print(f"  权重 W 形状: {linear_no_bias.weight.shape}")  # (3, 4)
print(f"  偏置 b: {linear_no_bias.bias}")               # None

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = linear_no_bias(x)
y_manual = x @ linear_no_bias.weight.T  # 没有 + b
print(f"  输出 y: {y.tolist()}")
print(f"  手动计算: {y_manual.tolist()}")
print(f"  两者相等: {torch.allclose(y, y_manual)}")
print()

# ===== 4. 矩阵乘法的直观理解 =====
# Linear 本质就是矩阵乘法, 把每个输入维度映射到每个输出维度
# W[i][j] 表示: 输入第j维对输出第i维的贡献(权重)

W = linear.weight
print("权重矩阵 W 的含义:")
for i in range(W.shape[0]):
    print(f"  输出第{i}维 = {W[i].tolist()} · x + b[{i}]={linear.bias[i].item():.4f}")
    # 就是输入各维度乘对应权重再求和, 再加偏置
print()

# ===== 5. 批量输入 =====
# 实际训练时输入是batch
batch_x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],   # 第1个样本
    [5.0, 6.0, 7.0, 8.0],   # 第2个样本
])
batch_y = linear(batch_x)
print(f"批量输入形状: {batch_x.shape}")   # (2, 4)
print(f"批量输出形状: {batch_y.shape}")   # (2, 3)
print(f"每个样本独立做线性变换, batch维度不受影响")
print()

# ===== 6. 在 GPT 中 lm_head 的作用 =====
# lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# 它把 embedding 向量映射回词汇表大小的分数, 用于预测下一个token

n_embd = 32
vocab_size = 65

lm_head = nn.Linear(n_embd, vocab_size, bias=False)
print(f"lm_head 权重形状: {lm_head.weight.shape}")  # (65, 32)

# 模拟一个 token 的 embedding
token_embed = torch.randn(n_embd)  # (32)
logits = lm_head(token_embed)       # (65)

print(f"embedding 形状: {token_embed.shape}")   # (32)
print(f"logits 形状:    {logits.shape}")        # (65) — 每个token一个分数
print(f"最高分token ID: {logits.argmax().item()} (模型预测下一个token)")
print()

# ===== 7. lm_head 和 wte 的特殊关系 =====
# 在 GPT 中, lm_head 的权重和 wte 的权重形状相同!
# wte.weight:     (vocab_size, n_embd) = (65, 32)
# lm_head.weight: (vocab_size, n_embd) = (65, 32)
# 有些实现中 lm_head.weight 就是 wte.weight (权重共享)

wte = nn.Embedding(vocab_size, n_embd)
print(f"wte 权重形状:     {wte.weight.shape}")       # (65, 32)
print(f"lm_head 权重形状: {lm_head.weight.shape}")   # (65, 32)
print(f"形状相同! 所以可以做权重共享: lm_head.weight = wte.weight")
print(f"这就省了一个 (65, 32) 的参数矩阵")