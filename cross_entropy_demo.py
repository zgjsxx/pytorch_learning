"""
F.cross_entropy 两种格式 demo
"""

import torch
import torch.nn.functional as F

vocab_size = 5
num_samples = 3

# 模型预测的 logits（3个样本，每个样本对5个词的分数）
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5, 0.3],  # 样本0：分数最高是词0（2.0）
    [0.5, 0.2, 3.0, 1.0, 0.8],  # 样本1：分数最高是词2（3.0）
    [1.0, 4.0, 0.5, 0.3, 0.1],  # 样本2：分数最高是词1（4.0）
])

# ========== 格式1：整数 class ID（GPT用的方式） ==========
# targets 每个元素就是一个整数，表示正确答案的词ID
targets_int = torch.tensor([0, 2, 1])  # 样本0答案是词0，样本1答案是词2，样本2答案是词1

loss_int = F.cross_entropy(logits, targets_int)
print("=== 格式1：整数 class ID ===")
print(f"logits:    {logits.shape}  → (num_samples, vocab_size)")
print(f"targets:   {targets_int.shape}  → (num_samples,) 每个是一个词ID")
print(f"  样本0: logits=[2.0, 1.0, 0.1, 0.5, 0.3], 正确答案=词0")
print(f"  样本1: logits=[0.5, 0.2, 3.0, 1.0, 0.8], 正确答案=词2")
print(f"  样本2: logits=[1.0, 4.0, 0.5, 0.3, 0.1], 正确答案=词1")
print(f"loss = {loss_int.item():.4f}")
print()

# ========== 格式2：one-hot / 概率分布 ==========
# targets 每行是一个概率分布，只有正确答案位置是1
targets_onehot = torch.tensor([
    [1, 0, 0, 0, 0],  # 样本0：答案是词0
    [0, 0, 1, 0, 0],  # 样本1：答案是词2
    [0, 1, 0, 0, 0],  # 样本2：答案是词1
], dtype=torch.float)

loss_onehot = F.cross_entropy(logits, targets_onehot)
print("=== 格式2：one-hot ===")
print(f"logits:    {logits.shape}  → (num_samples, vocab_size)")
print(f"targets:   {targets_onehot.shape}  → (num_samples, vocab_size) 每行是概率分布")
print(f"loss = {loss_onehot.item():.4f}")
print()

print("=== 两种格式结果完全一样 ===")
print(f"格式1 loss: {loss_int.item():.4f}")
print(f"格式2 loss: {loss_onehot.item():.4f}")
print(f"差值: {abs(loss_int.item() - loss_onehot.item()):.6f}")
print()

# ========== 为什么 GPT 用格式1 ==========
print("=== 为什么用整数ID不用one-hot ===")
vocab_size_gpt = 50257  # GPT-2 的词表大小
print(f"GPT-2 vocab_size = {vocab_size_gpt}")
print(f"格式1: targets 每个样本只需 1 个整数 = 4 bytes")
print(f"格式2: targets 每个样本需要 {vocab_size_gpt} 个浮点数 = {vocab_size_gpt * 4} bytes")
print(f"格式1更高效，内存占用少 ~{vocab_size_gpt}倍")