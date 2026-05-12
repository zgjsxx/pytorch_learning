"""
nn.Embedding 学习 demo
模拟 GPT 中的 wte (word token embedding) 层
"""

import torch
import torch.nn as nn

# ===== 1. 基本概念 =====
# nn.Embedding(vocab_size, embedding_dim) 是一个查找表：
#   - vocab_size: 词汇表大小（有多少个不同的 token）
#   - embedding_dim: 每个 token 映射到的向量维度
# 它把一个整数 token ID 映射成一个 embedding_dim 维的向量

# 模拟一个小型 GPT config
class Config:
    vocab_size = 65      # 词汇表大小（比如 Shakespeare 数据集有 65 个唯一字符）
    n_embd = 32          # embedding 维度

config = Config()

# ===== 2. 创建 Embedding 层 =====
wte = nn.Embedding(config.vocab_size, config.n_embd)
print(f"Embedding 权重形状: {wte.weight.shape}")  # (vocab_size, n_embd)
print(f"Embedding 权重:\n{wte.weight[:3]}")        # 看前3个token的向量
print()

# ===== 3. 单个 token 的 embedding =====
token_id = torch.tensor([5])  # token ID = 5
embedding = wte(token_id)
print(f"token_id={token_id.item()} -> embedding 形状: {embedding.shape}")
print(f"embedding 值:\n{embedding}")
print(f"等价于直接查表: {torch.allclose(embedding, wte.weight[5])}")
print()

# ===== 4. 一段文本（多个 token）的 embedding =====
# 模拟一段文本: "hello" 编码成 token IDs
input_ids = torch.tensor([5, 10, 15, 20, 25])  # 5个token组成的序列
embeddings = wte(input_ids)
print(f"输入 token IDs: {input_ids.tolist()}")
print(f"输出 embeddings 形状: {embeddings.shape}")  # (seq_len, n_embd) = (5, 32)
print()

# ===== 5. 批量输入（Batch） =====
# 实际训练时一次处理多个序列
batch_input_ids = torch.tensor([
    [5, 10, 15, 20, 25],   # 序列1
    [3, 7, 12, 18, 24],    # 序列2
])
batch_embeddings = wte(batch_input_ids)
print(f"批量输入形状: {batch_input_ids.shape}")       # (batch, seq_len) = (2, 5)
print(f"批量输出形状: {batch_embeddings.shape}")      # (batch, seq_len, n_embd) = (2, 5, 32)
print()

# ===== 6. 在 GPT 中的位置 =====
# GPT 中 token embedding 和 position embedding 相加得到最终输入
n_positions = 64  # 最大序列长度
wpe = nn.Embedding(n_positions, config.n_embd)  # position embedding

seq_len = 5
positions = torch.arange(seq_len)                # 位置: [0, 1, 2, 3, 4]
pos_embed = wpe(positions)                       # (seq_len, n_embd)

# 最终输入 = token embedding + position embedding
final_input = embeddings + pos_embed
print(f"token embedding 形状: {embeddings.shape}")
print(f"position embedding 形状: {pos_embed.shape}")
print(f"最终输入形状: {final_input.shape}")       # (seq_len, n_embd) = (5, 32)
print()

# ===== 7. 直观理解：Embedding 让相似 token 有相似向量 =====
# 训练前是随机初始化的，训练后语义相近的 token 向量会更接近
token_a = wte(torch.tensor([10]))
token_b = wte(torch.tensor([10]))  # 同一个 token
token_c = wte(torch.tensor([50]))  # 不同 token

dist_same = torch.norm(token_a - token_b).item()
dist_diff = torch.norm(token_a - token_c).item()
print(f"相同 token (10 vs 10) 距离: {dist_same:.4f}")
print(f"不同 token (10 vs 50) 距离: {dist_diff:.4f}")
print(f"(训练前随机初始化，训练后语义相近的token距离会变小)")