"""
logits.view(-1, vocab_size) 形状变换 demo
理解 cross_entropy 为什么要 reshape
"""

import torch

# 模拟一个小场景：batch=2, seq_len=3, vocab_size=5
b, t, vocab_size = 2, 3, 5

# logits: 每个位置对 vocab 中每个词的预测分数
logits = torch.randn(b, t, vocab_size)
print("=== logits 原始形状 ===")
print(f"logits.shape: {logits.shape}")  # (2, 3, 5)
print(f"logits:\n{logits}")
print()

# targets: 每个位置的正确答案 (token id)
targets = torch.randint(0, vocab_size, (b, t))
print("=== targets 原始形状 ===")
print(f"targets.shape: {targets.shape}")  # (2, 3)
print(f"targets:\n{targets}")
print()

# ========== view(-1, vocab_size) ==========

# -1 表示"让 PyTorch 自动推算这个维度的大小"
# 总元素数不变：2*3*5 = 30，vocab_size=5 固定，所以 -1 自动算出 2*3=6
logits_flat = logits.view(-1, vocab_size)
print("=== logits.view(-1, vocab_size) ===")
print(f"logits_flat.shape: {logits_flat.shape}")  # (6, 5)
print(f"logits_flat:\n{logits_flat}")
print()

# 对比：手动对应每一行来自原始 logits 的哪个位置
print("=== 逐行对照 ===")
for i in range(b):
    for j in range(t):
        row_idx = i * t + j  # 展平后的行号
        print(f"flat[{row_idx}] = logits[{i},{j}]  "
              f"→ batch{i} 位置{j} 的预测分数, 正确答案=targets[{i},{j}]={targets[i,j]}")
print()

# targets 也同样展平
targets_flat = targets.view(-1)
print(f"targets_flat.shape: {targets_flat.shape}")  # (6,)
print(f"targets_flat: {targets_flat}")
print()

# ========== 现在可以算 cross_entropy ==========

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"cross_entropy loss: {loss.item():.4f}")
print()
print("本质：把 (b,t,vocab_size) 和 (b,t) 展平成 (b*t, vocab_size) 和 (b*t,)")
print("     → 每行是一个分类问题：预测分数 vs 正确答案")
print("     → 总共 b*t 个分类问题，取平均 loss")