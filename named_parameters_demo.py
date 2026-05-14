"""
named_parameters() 学习 demo
"""

import torch
import torch.nn as nn


# ===== 1. 基本对比：parameters() vs named_parameters() =====

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 3, bias=False)
        self.norm = nn.LayerNorm(4)

model = SimpleModel()

print("=== parameters() — 只有值，没有名字 ===")
for i, p in enumerate(model.parameters()):
    print(f"  [{i}] shape={p.shape}")

print()
print("=== named_parameters() — 有名字 + 值 ===")
for name, p in model.named_parameters():
    print(f"  {name}: shape={p.shape}")

print()

# ===== 2. 名字是怎么来的 ===
print("=== 名字来源：属性路径 ===")
# 名字 = 从外到内的属性访问路径，用 . 连接
# self.embed.weight     → "embed.weight"
# self.linear.weight    → "linear.weight"
# self.norm.weight      → "norm.weight"
# self.norm.bias        → "norm.bias"

# ===== 3. 嵌套模块的名字 ===

class Block(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(10, 4),
            h = nn.ModuleList([Block(4) for _ in range(2)]),
            ln_f = nn.LayerNorm(4),
        ))
        self.lm_head = nn.Linear(4, 10, bias=False)

gpt = MiniGPT()

print("=== MiniGPT 所有参数名字 ===")
for name, p in gpt.named_parameters():
    print(f"  {name}: shape={p.shape}")

print()

# ===== 4. 用名字筛选参数 ===
print("=== 筛选出所有 c_proj.weight ===")
for name, p in gpt.named_parameters():
    if name.endswith('c_proj.weight'):
        print(f"  {name}: shape={p.shape}")

print()

# ===== 5. weight tying 后同名参数出现两次 ===
gpt.transformer.wte.weight = gpt.lm_head.weight  # weight tying

print("=== weight tying 后的 named_parameters ===")
for name, p in gpt.named_parameters():
    print(f"  {name}: shape={p.shape}")

print()
print("注意: wte.weight 和 lm_head.weight 是同一个张量!")
print(f"  wte.weight 的 data_ptr: {gpt.transformer.wte.weight.data_ptr()}")
print(f"  lm_head.weight 的 data_ptr: {gpt.lm_head.weight.data_ptr()}")
print(f"  两者相同，说明共享同一块内存，但 named_parameters 里出现了两次")

print()

# ===== 6. 统计参数数量 ===
print("=== 参数统计 ===")
all_params = sum(p.numel() for p in gpt.parameters())
unique_params = sum(p.numel() for p in dict.fromkeys(gpt.parameters()))  # 去重
print(f"  named_parameters 总参数: {all_params}")
print(f"  去重后实际参数: {unique_params}")
print(f"  重复的参数就是 tied 的 wte/lm_head.weight")