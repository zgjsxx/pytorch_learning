"""
cross_entropy 手动计算过程 demo
逐步拆解 cross_entropy 内部到底做了什么
"""

import torch
import torch.nn.functional as F

# 3个样本，5个词的 vocab
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5, 0.3],  # 样本0
    [0.5, 0.2, 3.0, 1.0, 0.8],  # 样本1
    [1.0, 4.0, 0.5, 0.3, 0.1],  # 样本2
])

targets = torch.tensor([0, 2, 1])  # 正确答案：样本0→词0，样本1→词2，样本2→词1

# ========== PyTorch 自带计算 ==========

loss_pytorch = F.cross_entropy(logits, targets)
print(f"PyTorch cross_entropy loss: {loss_pytorch.item():.4f}")
print()

# ========== 手动逐步计算 ==========

print("=== 手动逐步计算 cross_entropy ===")
print()

# 第1步：softmax — 把 logits 变成概率分布
softmax = F.softmax(logits, dim=-1)
print("第1步：softmax(logits) → 每行变成概率分布，总和=1")
print(f"  样本0: {softmax[0].tolist()}  总和={softmax[0].sum().item():.4f}")
print(f"  样本1: {softmax[1].tolist()}  总和={softmax[1].sum().item():.4f}")
print(f"  样本2: {softmax[2].tolist()}  总和={softmax[2].sum().item():.4f}")
print()

# 第2步：取出正确答案位置的概率
p_correct = softmax[range(len(targets)), targets]
print("第2步：取出正确答案位置的概率 p_correct")
print(f"  样本0: 正确答案是词0 → p = {p_correct[0].item():.4f} (分数最高2.0，概率也最高)")
print(f"  样本1: 正确答案是词2 → p = {p_correct[1].item():.4f} (分数最高3.0，概率也最高)")
print(f"  样本2: 正确答案是词1 → p = {p_correct[2].item():.4f} (分数最高4.0，概率也最高)")
print()

# 第3步：-log(p) → 每个样本的 loss
neg_log_p = -torch.log(p_correct)
print("第3步：-log(p_correct) → 每个样本的 loss")
print(f"  样本0: -log({p_correct[0].item():.4f}) = {neg_log_p[0].item():.4f}  (概率高，loss低)")
print(f"  样本1: -log({p_correct[1].item():.4f}) = {neg_log_p[1].item():.4f}  (概率高，loss低)")
print(f"  样本2: -log({p_correct[2].item():.4f}) = {neg_log_p[2].item():.4f}  (概率高，loss低)")
print()

# 第4步：求平均
loss_manual = neg_log_p.mean()
print("第4步：求平均")
print(f"  ({neg_log_p[0].item():.4f} + {neg_log_p[1].item():.4f} + {neg_log_p[2].item():.4f}) / 3 = {loss_manual.item():.4f}")
print()

# ========== 对比 ==========

print("=== 对比结果 ===")
print(f"PyTorch 计算: {loss_pytorch.item():.4f}")
print(f"手动计算:     {loss_manual.item():.4f}")
print(f"完全一致!")
print()

# ========== 直觉理解 ==========

print("=== 直觉理解 ===")
print("预测对了（概率高） → -log(p) 小 → loss 低 → 模型开心")
print("预测错了（概率低） → -log(p) 大 → loss 高 → 模型痛苦")
print()

# 预测错的情况演示
bad_logits = torch.tensor([
    [0.1, 0.2, 2.0, 0.5, 3.0],  # 正确答案是词0，但分数最低！
])
bad_targets = torch.tensor([0])
bad_loss = F.cross_entropy(bad_logits, bad_targets)
bad_softmax = F.softmax(bad_logits, dim=-1)
bad_p = bad_softmax[0, 0].item()
print(f"预测错的例子: 正确答案是词0，但模型给词0的分数只有0.1")
print(f"  softmax后词0的概率 p = {bad_p:.4f}")
print(f"  loss = -log({bad_p:.4f}) = {bad_loss.item():.4f}  → loss很高，模型很痛苦")
print()

good_logits = torch.tensor([
    [10.0, 0.1, 0.2, 0.5, 0.3],  # 正确答案是词0，分数远高于其他
])
good_targets = torch.tensor([0])
good_loss = F.cross_entropy(good_logits, good_targets)
good_softmax = F.softmax(good_logits, dim=-1)
good_p = good_softmax[0, 0].item()
print(f"预测对的例子: 正确答案是词0，模型给词0的分数10.0远高于其他")
print(f"  softmax后词0的概率 p = {good_p:.4f}")
print(f"  loss = -log({good_p:.4f}) = {good_loss.item():.4f}  → loss很低，模型很开心")