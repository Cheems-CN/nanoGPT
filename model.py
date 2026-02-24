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
# 📐 代码规范 (请在实现每个 TODO 时遵循以下规范)
# =============================================================================
#
# 命名规范:
#   - 类名: PascalCase (如 CausalSelfAttention, GPTConfig)
#   - 方法/函数名: snake_case (如 get_num_params, forward)
#   - 变量名: snake_case (如 block_size, n_head)
#   - 常量: UPPER_SNAKE_CASE (如有需要)
#   - 私有方法: 前缀下划线 (如 _init_weights)
#   - nn.Module 子层命名: 与 HuggingFace GPT-2 保持一致
#     (c_attn, c_proj, c_fc, ln_1, ln_2, wte, wpe, ln_f, lm_head)
#
# 类型标注:
#   - 所有公开方法的参数和返回值都应有类型标注
#   - 示例:
#     def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
#                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
#     def get_num_params(self, non_embedding: bool = True) -> int:
#     def generate(self, idx: torch.Tensor, max_new_tokens: int,
#                  temperature: float = 1.0, top_k: int | None = None
#                  ) -> torch.Tensor:
#
# 文档字符串 (Google 风格):
#   - 每个类和公开方法都需要 docstring
#   - 格式示例:
#     class MLP(nn.Module):
#         """Transformer 前馈网络。
#
#         架构: Linear(C→4C) → GELU → Linear(4C→C) → Dropout
#
#         Args:
#             config: 模型配置对象
#         """
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """前馈网络的前向传播。
#
#         Args:
#             x: 输入张量，形状 (batch_size, seq_len, n_embd)
#
#         Returns:
#             输出张量，形状与输入相同 (batch_size, seq_len, n_embd)
#         """
#
# 断言与防御性编程:
#   - 在关键位置添加 assert 验证维度/类型约束
#   - 示例: assert config.n_embd % config.n_head == 0
#   - 对不合法输入尽早报错，给出清晰的错误消息
#
# PyTorch 最佳实践:
#   - 所有可学习参数必须通过 nn.Parameter 或 nn.Module 子模块注册
#   - 非参数的持久张量（如因果掩码）使用 self.register_buffer()
#   - __init__ 中只做层定义，不做前向计算
#   - forward() 中避免原地操作 (inplace ops)，以免破坏自动求导
#   - 在不需要梯度的地方使用 @torch.no_grad() 装饰器
#
# 代码风格:
#   - 每行不超过 100 个字符（推荐；原项目约 120 字符宽）
#   - 运算符两侧加空格: x = a + b，而非 x=a+b
#   - 逗号后加空格: dict(a=1, b=2)，而非 dict(a=1,b=2)
#   - 类之间空两行，方法之间空一行
#   - import 分组: 标准库 → 第三方库 → 本地模块，组间空一行
#   - 避免魔法数字，使用命名常量或 config 属性
#     ✗ self.c_fc = nn.Linear(config.n_embd, 3072)
#     ✓ self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
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
#
# 📐 规范提示:
#   - 使用 @dataclass 装饰器 (已导入 from dataclasses import dataclass)
#   - 每个字段都需要类型标注和默认值
#   - 在字段上方或右侧添加注释说明含义
#   - 示例:
#     @dataclass
#     class GPTConfig:
#         """GPT 模型配置。"""
#         block_size: int = 1024  # 最大序列长度
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
#
# 📐 规范提示:
#   - 用 nn.Parameter 包装可学习参数: self.weight = nn.Parameter(torch.ones(ndim))
#   - bias 为 None 时用条件表达式: nn.Parameter(...) if bias else None
#   - 类 docstring 说明"为什么不直接用 nn.LayerNorm"
#   - forward 参数标注: def forward(self, input: torch.Tensor) -> torch.Tensor:
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
#
# 📐 规范提示:
#   - 子层命名与 GPT-2 一致: c_fc (升维), c_proj (降维), gelu, dropout
#   - 用 config 属性代替硬编码: 4 * config.n_embd 而非字面量
#   - forward 中每步一行，保持数据流清晰:
#     x = self.c_fc(x)
#     x = self.gelu(x)
#     x = self.c_proj(x)
#     x = self.dropout(x)
#     return x
#   - 类型标注: def forward(self, x: torch.Tensor) -> torch.Tensor:
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
#
# 📐 规范提示:
#   - 子层命名: c_attn (QKV 合并投影), c_proj (输出投影),
#               attn_dropout, resid_dropout
#   - 实例属性保存常用值: self.n_head, self.n_embd, self.dropout
#   - 用 hasattr(torch.nn.functional, 'scaled_dot_product_attention')
#     检测 Flash Attention 支持，保存为 self.flash
#   - 非参数 buffer 用 register_buffer:
#     self.register_buffer("bias",
#         torch.tril(torch.ones(block_size, block_size))
#              .view(1, 1, block_size, block_size))
#   - forward 中用注释分隔三个步骤，标明张量形状变化:
#     # (B, T, C) -> (B, T, 3C) -> 3 x (B, T, C)
#     # (B, T, C) -> (B, nh, T, hs)
#   - 用 .contiguous() 确保内存连续再 .view()
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
#
# 📐 规范提示:
#   - 子层命名: ln_1, attn, ln_2, mlp (与 GPT-2 一致)
#   - forward 极简——两行即可，保持表达力:
#     x = x + self.attn(self.ln_1(x))
#     x = x + self.mlp(self.ln_2(x))
#   - 不要引入多余的中间变量，残差连接写成一行更清晰
#   - 类型标注: def forward(self, x: torch.Tensor) -> torch.Tensor:
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
#
# 📐 规范提示:
#   - __init__ 中用 self.config = config 保存配置，便于后续方法访问
#   - nn.ModuleDict + dict() 语法构建 self.transformer，使键名可读:
#     self.transformer = nn.ModuleDict(dict(
#         wte = nn.Embedding(...),
#         wpe = nn.Embedding(...),
#         ...
#     ))
#   - Weight Tying 直接赋值: self.transformer.wte.weight = self.lm_head.weight
#   - 权重初始化用 self.apply(self._init_weights) 遍历所有子模块
#   - _init_weights 中用 isinstance() 判断模块类型，分别初始化
#   - forward 的类型标注:
#     def forward(self, idx: torch.Tensor,
#                 targets: torch.Tensor | None = None
#                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
#   - generate 中用 F.softmax(dim=-1) 而非手动实现
#   - from_pretrained 用 @classmethod 装饰，返回 cls 的实例
#   - configure_optimizers 中用字典推导式分组参数，打印统计信息
# -----------------------------------------------------------------------------
