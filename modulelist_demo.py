"""
nn.ModuleList 学习 demo
模拟 GPT 中的 h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
"""

import torch
import torch.nn as nn


# ===== 1. 普通 list vs ModuleList =====

# 定义一个简单模块: Linear + ReLU
class SimpleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


# 用普通 list
class ModelWithList(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [SimpleBlock(4, 4) for _ in range(3)]  # 普通 list

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# 用 ModuleList
class ModelWithModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([SimpleBlock(4, 4) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


model_list = ModelWithList()
model_modulelist = ModelWithModuleList()

# ===== 2. 参数追踪对比 =====
print("=== 参数追踪对比 ===")
print(f"普通 list 模型参数数量: {sum(p.numel() for p in model_list.parameters())}")
print(f"ModuleList 模型参数数量: {sum(p.numel() for p in model_modulelist.parameters())}")
print()

# 每个 SimpleBlock 有: Linear(4,4) 的权重(16) + 偏置(4) = 20个参数
# 3个 Block 应该有 60 个参数
print(f"每个 Block 参数: {sum(p.numel() for p in SimpleBlock(4,4).parameters())}")
print(f"3个 Block 期望参数: 60")
print(f"普通 list 只追踪到 0 个! ModuleList 追踪到 60 个")
print()

# ===== 3. 保存/加载对比 =====
print("=== 保存模型对比 ===")
import tempfile, os
tmpdir = tempfile.gettempdir()
model_list_path = os.path.join(tmpdir, "model_list.pt")
model_modulelist_path = os.path.join(tmpdir, "model_modulelist.pt")

torch.save(model_list.state_dict(), model_list_path)
torch.save(model_modulelist.state_dict(), model_modulelist_path)

import os
print(f"普通 list state_dict 大小: {os.path.getsize(model_list_path)} bytes")
print(f"ModuleList state_dict 大小: {os.path.getsize(model_modulelist_path)} bytes")
print(f"普通 list 的 state_dict 几乎是空的, ModuleList 包含了所有3个Block的参数")
print()

# ===== 4. ModuleList 的使用方式 =====
print("=== ModuleList 使用方式 ===")
blocks = nn.ModuleList([SimpleBlock(4, 4) for _ in range(3)])

# 和普通 list 一样索引
print(f"blocks[0]: {blocks[0]}")
print(f"blocks[-1]: {blocks[-1]}")
print()

# 和普通 list 一样遍历
print("遍历 ModuleList:")
for i, block in enumerate(blocks):
    params = sum(p.numel() for p in block.parameters())
    print(f"  blocks[{i}] 参数数量: {params}")
print()

# ===== 5. 模拟 GPT 中的多层堆叠 =====
print("=== 模拟 GPT 多层堆叠 ===")

class GPTBlock(nn.Module):
    """模拟一个简化的 Transformer Block"""
    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        x = x + self.linear(self.ln(x))  # 残差连接
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.h = nn.ModuleList([GPTBlock(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.wte(input_ids)
        for block in self.h:       # 逐层通过每个 Block
            x = block(x)
        logits = self.lm_head(x)
        return logits


model = MiniGPT(vocab_size=10, n_embd=8, n_layer=3)
input_ids = torch.tensor([1, 3, 5])
logits = model(input_ids)

print(f"模型结构:")
print(model)
print()
print(f"输入 token IDs: {input_ids.tolist()}")
print(f"Embedding 输出形状: {model.wte(input_ids).shape}")  # (3, 8)
print(f"经过 3 层 Block 后形状: {model.h[0](model.wte(input_ids)).shape}")  # (3, 8)
print(f"最终 logits 形状: {logits.shape}")                  # (3, 10)
print()

# 数据流: input_ids → Embedding → Block[0] → Block[1] → Block[2] → lm_head → logits
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数数量: {total_params}")
print(f"其中 ModuleList(h) 的参数: {sum(p.numel() for n, p in model.named_parameters() if 'h.' in n)}")