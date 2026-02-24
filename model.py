"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
#                      TODO: 从零实现完整的 GPT-2 模型
# =============================================================================
#
# 本文件需要你从零定义所有类和方法。完成后，train.py 和 sample.py 将能正常工作。
#
# 建议实现顺序:
#   1. GPTConfig           — 模型配置（@dataclass，最简单）
#   2. LayerNorm           — 自定义 LayerNorm
#   3. MLP                 — 前馈网络
#   4. CausalSelfAttention — 多头因果自注意力（核心难点）
#   5. Block               — Transformer Block
#   6. GPT                 — 完整模型（组合所有组件）
#
# 外部依赖: train.py 会 `from model import GPTConfig, GPT`
#           sample.py 同理，请确保这两个名字可被导入
#
# =============================================================================


# -----------------------------------------------------------------------------
# TODO 1: GPTConfig
# -----------------------------------------------------------------------------
# 用 @dataclass 装饰器定义模型超参数配置类
#
# 需要定义的字段（含类型和默认值）:
#   block_size: int = 1024    — 最大序列长度
#   vocab_size: int = 50304   — 词表大小 (GPT-2 的 50257 向上取整到 64 的倍数)
#   n_layer:    int = 12      — Transformer Block 层数
#   n_head:     int = 12      — 注意力头数
#   n_embd:     int = 768     — 嵌入维度（必须能被 n_head 整除）
#   dropout:    float = 0.0   — Dropout 率
#   bias:       bool = True   — 是否在 Linear/LayerNorm 中使用 bias
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TODO 2: LayerNorm (继承 nn.Module)
# -----------------------------------------------------------------------------
# 功能: 带可选 bias 的 Layer Normalization
#       PyTorch 原生 nn.LayerNorm 不支持仅关闭 bias，需要自定义
#
# 构造函数 __init__(self, ndim, bias):
#   - weight 参数: 形状 (ndim,)，初始全 1
#   - bias 参数:   形状 (ndim,)，初始全 0；如果 bias=False 则为 None
#
# 前向传播 forward(self, input):
#   - 使用 F.layer_norm 进行归一化，eps=1e-5
#   - 输入输出形状不变
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TODO 3: MLP (继承 nn.Module)
# -----------------------------------------------------------------------------
# 功能: Transformer 中的前馈神经网络
#
# 架构: Linear(C → 4C) → GELU → Linear(4C → C) → Dropout
#   - 先升维 4 倍，经过非线性激活，再降回原维度
#   - GELU 是 GPT 系列的标准激活函数: GELU(x) = x · Φ(x)
#
# 构造函数 __init__(self, config):
#   - 需要两个 Linear 层和一个 GELU 激活、一个 Dropout
#   - Linear 的 bias 参数由 config.bias 控制
#
# 前向传播 forward(self, x):
#   - 输入输出形状: (B, T, C) → (B, T, C)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TODO 4: CausalSelfAttention (继承 nn.Module)
# -----------------------------------------------------------------------------
# 功能: 多头因果自注意力机制 —— Transformer 的核心
#
# 核心数学:
#   Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
#   其中 d_k = n_embd / n_head（每个头的维度）
#
# 构造函数 __init__(self, config):
#   - 断言 n_embd 能被 n_head 整除
#   - Q/K/V 合并投影: 一个 Linear(n_embd → 3·n_embd)
#   - 输出投影: 一个 Linear(n_embd → n_embd)
#   - 两个 Dropout（注意力权重 + 残差连接）
#   - 检测是否支持 Flash Attention (PyTorch >= 2.0)
#   - 不支持 Flash Attention 时，注册因果掩码 buffer（下三角矩阵）
#
# 前向传播 forward(self, x):
#   - 输入: (B, T, C)  输出: (B, T, C)
#   - 步骤 1: 通过合并投影得到 Q, K, V，reshape 为多头形式 (B, nh, T, hs)
#   - 步骤 2: 计算注意力（两条路径: Flash Attention / 手动实现）
#     * 手动实现需要: 缩放点积 → 因果掩码 → softmax → dropout → 加权求和
#     * 因果掩码: 每个位置只能看到自身及之前的位置（下三角矩阵）
#   - 步骤 3: 合并多头输出 → 输出投影 → 残差 Dropout
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TODO 5: Block (继承 nn.Module)
# -----------------------------------------------------------------------------
# 功能: 单个 Transformer Block，使用 Pre-Norm 架构
#
# 架构 (Pre-Norm，区别于原始 Transformer 的 Post-Norm):
#   x = x + Attention(LayerNorm(x))   ← 注意力子层 + 残差
#   x = x + MLP(LayerNorm(x))         ← 前馈子层 + 残差
#
# 构造函数 __init__(self, config):
#   - 两个 LayerNorm、一个 CausalSelfAttention、一个 MLP
#
# 前向传播 forward(self, x):
#   - 输入输出形状: (B, T, C)
#   - 残差连接使梯度可以直接回传到浅层，缓解梯度消失
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TODO 6: GPT (继承 nn.Module)
# -----------------------------------------------------------------------------
# 功能: 完整的 GPT 语言模型
#
# ═══════════ 构造函数 __init__(self, config) ═══════════
# 需要构建的组件:
#   self.transformer = nn.ModuleDict 包含:
#     - wte:  Token 嵌入层   Embedding(vocab_size, n_embd)
#     - wpe:  位置嵌入层     Embedding(block_size, n_embd)
#     - drop: Embedding Dropout
#     - h:    N 个 Block 组成的 ModuleList
#     - ln_f: 最终 LayerNorm
#   self.lm_head = Linear(n_embd, vocab_size, bias=False)
#
# 关键技巧:
#   - Weight Tying: wte.weight 和 lm_head.weight 共享（同一个参数）
#   - 权重初始化: Linear 用 N(0, 0.02)，Embedding 用 N(0, 0.02)
#   - 残差投影层的初始化需要额外缩放: std = 0.02 / √(2·n_layer)
#
# ═══════════ 核心方法 ═══════════
#
# forward(self, idx, targets=None) → (logits, loss):
#   输入: idx (b, t) — token 索引序列
#   流程: Token Emb + Pos Emb → Dropout → N × Block → LayerNorm → LM Head
#   训练模式 (targets != None):
#     - logits: (b, t, vocab_size)，loss: 交叉熵标量
#   推理模式 (targets == None):
#     - 只计算最后一个位置的 logits: (b, 1, vocab_size)，loss: None
#     - 推理优化: 用 x[:, [-1], :] 保留时间维度
#
# generate(self, idx, max_new_tokens, temperature=1.0, top_k=None) → Tensor:
#   自回归文本生成，需要 @torch.no_grad() 装饰
#   循环 max_new_tokens 次:
#     1. 裁剪上下文到 block_size
#     2. 前向传播获取最后位置的 logits
#     3. 温度缩放（控制随机性）
#     4. Top-K 过滤（可选，将低概率 token 的 logits 设为 -inf）
#     5. Softmax → multinomial 采样 → 追加到序列
#
# ═══════════ 辅助方法 ═══════════
#
# get_num_params(self, non_embedding=True) → int:
#   返回参数总数，默认减去位置嵌入参数
#
# _init_weights(self, module):
#   权重初始化回调，由 self.apply() 调用
#
# crop_block_size(self, block_size):
#   模型手术: 裁剪上下文窗口（修改 config、wpe 权重、注意力掩码）
#
# ═══════════ 预训练权重加载 ═══════════
#
# @classmethod
# from_pretrained(cls, model_type, override_args=None) → GPT:
#   从 HuggingFace 加载 GPT-2 预训练权重
#   支持: 'gpt2'(124M), 'gpt2-medium'(350M), 'gpt2-large'(774M), 'gpt2-xl'(1558M)
#   注意: OpenAI 使用 Conv1D，以下权重需要转置:
#     attn.c_attn.weight, attn.c_proj.weight, mlp.c_fc.weight, mlp.c_proj.weight
#   配置映射:
#     gpt2:        n_layer=12,  n_head=12, n_embd=768
#     gpt2-medium: n_layer=24,  n_head=16, n_embd=1024
#     gpt2-large:  n_layer=36,  n_head=20, n_embd=1280
#     gpt2-xl:     n_layer=48,  n_head=25, n_embd=1600
#
# ═══════════ 优化器配置 ═══════════
#
# configure_optimizers(self, weight_decay, learning_rate, betas, device_type) → Optimizer:
#   参数分组: 2D 参数（权重矩阵）应用 weight_decay，1D 参数（bias/LN）不应用
#   使用 AdamW 优化器，可用时启用 fused 版本
#
# ═══════════ 性能估算 ═══════════
#
# estimate_mfu(self, fwdbwd_per_iter, dt) → float:
#   估算模型 FLOPS 利用率 (MFU)
#   公式: FLOPs/token ≈ 6N + 12·L·H·Q·T（参考 PaLM 论文 Appendix B）
#   与 A100 bf16 峰值 312 TFLOPS 比较
# -----------------------------------------------------------------------------
