# nanoGPT 项目架构解析

## 项目概述

nanoGPT 是 Andrej Karpathy 开发的一个**最简 GPT 训练/推理框架**。整个核心代码仅约 600 行 Python，却能完整复现 GPT-2（124M 参数）的训练过程。它的设计哲学是：**用最少的代码实现最核心的功能**，去除一切不必要的抽象。

---

## 目录结构总览

```
nanoGPT/
├── model.py                 # 🧠 核心：GPT 模型定义（约 400 行）
├── train.py                 # 🏋️ 核心：训练脚本（约 350 行）
├── sample.py                # 📝 推理：文本生成脚本（约 90 行）
├── bench.py                 # ⚡ 工具：性能基准测试
├── configurator.py          # ⚙️ 工具：配置系统
├── config/                  # 📋 预设配置文件目录
│   ├── train_gpt2.py            # GPT-2 (124M) 全量训练配置
│   ├── train_shakespeare_char.py # Shakespeare 字符级训练配置
│   ├── finetune_shakespeare.py   # Shakespeare 微调配置
│   ├── eval_gpt2.py              # GPT-2 评估配置
│   ├── eval_gpt2_medium.py       # GPT-2 Medium 评估配置
│   ├── eval_gpt2_large.py        # GPT-2 Large 评估配置
│   └── eval_gpt2_xl.py           # GPT-2 XL 评估配置
├── data/                    # 📊 数据集准备目录
│   ├── openwebtext/prepare.py    # OpenWebText 数据集下载和预处理
│   ├── shakespeare/prepare.py    # Shakespeare 数据集（BPE 分词）
│   └── shakespeare_char/prepare.py # Shakespeare 数据集（字符级分词）
├── assets/                  # 📸 文档图片资源
├── scaling_laws.ipynb       # 📓 缩放定律实验 Notebook
├── transformer_sizing.ipynb # 📓 Transformer 尺寸计算 Notebook
├── Architecture.md          # 📖 本文件：项目架构解析
├── Build_Guide.md           # 📖 手搓复现通关指南
└── README.md                # 📖 原始 README
```

---

## 核心文件详解

### 1. `model.py` — GPT 模型定义 🧠

这是整个项目的**灵魂文件**，包含了完整的 GPT-2 模型架构。它定义了以下核心类：

#### 类层级关系

```
GPT（主模型）
├── nn.Embedding (wte)           # Token 嵌入层
├── nn.Embedding (wpe)           # 位置嵌入层
├── nn.Dropout (drop)            # 嵌入层 Dropout
├── nn.ModuleList (h)            # N 个 Transformer Block
│   └── Block × N
│       ├── LayerNorm (ln_1)     # 注意力前的 LayerNorm
│       ├── CausalSelfAttention (attn)
│       │   ├── nn.Linear (c_attn)   # Q/K/V 投影（合并为一个矩阵）
│       │   ├── nn.Linear (c_proj)   # 输出投影
│       │   ├── nn.Dropout (attn_dropout)
│       │   └── nn.Dropout (resid_dropout)
│       ├── LayerNorm (ln_2)     # MLP 前的 LayerNorm
│       └── MLP (mlp)
│           ├── nn.Linear (c_fc)     # 升维：n_embd → 4*n_embd
│           ├── nn.GELU (gelu)       # GELU 激活函数
│           ├── nn.Linear (c_proj)   # 降维：4*n_embd → n_embd
│           └── nn.Dropout (dropout)
├── LayerNorm (ln_f)             # 最终 LayerNorm
└── nn.Linear (lm_head)         # 语言模型头（与 wte 共享权重）
```

#### 关键类说明

| 类名 | 作用 | 需要你实现的部分 |
|------|------|-----------------|
| `LayerNorm` | 带可选 bias 的 Layer Normalization | ✅ 已提供（无需实现） |
| `CausalSelfAttention` | 因果自注意力机制（多头 + 因果掩码） | ⬜ `forward()` 方法（TODO） |
| `MLP` | 前馈神经网络（升维→GELU→降维→Dropout） | ⬜ `forward()` 方法（TODO） |
| `Block` | 单个 Transformer Block（Pre-Norm + 残差） | ⬜ `forward()` 方法（TODO） |
| `GPTConfig` | 模型超参数配置（dataclass） | ✅ 已提供（无需实现） |
| `GPT` | GPT 主模型类 | ⬜ `forward()` 和 `generate()` 方法（TODO） |

#### 关键方法说明

| 方法 | 所属类 | 作用 | 状态 |
|------|--------|------|------|
| `forward(x)` | `CausalSelfAttention` | 计算多头因果自注意力 | ⬜ TODO |
| `forward(x)` | `MLP` | 计算前馈网络输出 | ⬜ TODO |
| `forward(x)` | `Block` | Pre-Norm + 注意力 + MLP + 残差 | ⬜ TODO |
| `forward(idx, targets)` | `GPT` | 完整前向传播 + Loss 计算 | ⬜ TODO |
| `generate(idx, ...)` | `GPT` | 自回归文本生成 | ⬜ TODO |
| `__init__(config)` | `GPT` | 构建模型架构 + 权重初始化 | ✅ 已提供 |
| `from_pretrained(model_type)` | `GPT` | 加载 HuggingFace GPT-2 预训练权重 | ✅ 已提供 |
| `configure_optimizers(...)` | `GPT` | 配置 AdamW 优化器（带权重衰减分组） | ✅ 已提供 |
| `crop_block_size(block_size)` | `GPT` | 裁剪上下文窗口大小 | ✅ 已提供 |
| `estimate_mfu(...)` | `GPT` | 估算模型 FLOPS 利用率 | ✅ 已提供 |

---

### 2. `train.py` — 训练脚本 🏋️

训练脚本负责完整的模型训练流程，包括数据加载、梯度累积、混合精度训练、分布式训练等。

#### 代码结构

```python
# ======== 第一部分：配置和初始化（已提供）========
# 1. 默认超参数定义（学习率、批大小、模型大小等）
# 2. 配置覆盖系统（命令行参数 / 配置文件）
# 3. DDP 分布式训练初始化
# 4. 随机种子、设备设置、混合精度上下文

# ======== 第二部分：数据加载（已提供）========
# get_batch(split) 函数：从内存映射文件中随机采样 batch

# ======== 第三部分：模型初始化（已提供）========
# 支持三种初始化方式：scratch / resume / gpt2*
# 包含优化器配置和模型编译

# ======== 第四部分：核心函数（TODO）========
# estimate_loss()：评估函数                    ⬜ TODO
# get_lr(it)：学习率调度器                      ⬜ TODO

# ======== 第五部分：训练主循环（TODO）========
# 前向传播 + Loss 缩放                         ⬜ TODO
# 反向传播（带梯度缩放）                        ⬜ TODO
# 梯度裁剪 + 优化器更新 + 梯度清零               ⬜ TODO
```

#### 关键概念

| 概念 | 说明 |
|------|------|
| **梯度累积** | 将大 batch 拆分为多个 micro-batch，累积梯度后再更新参数 |
| **混合精度训练** | 使用 FP16/BF16 加速计算，用 GradScaler 防止梯度下溢 |
| **余弦退火 + 线性预热** | 学习率先线性增长，再按余弦曲线衰减到最小值 |
| **DDP 分布式训练** | 多 GPU 并行训练，仅在最后一个 micro-step 同步梯度 |
| **权重衰减分组** | 2D 参数（权重矩阵）应用权重衰减，1D 参数（偏置、LayerNorm）不应用 |

---

### 3. `sample.py` — 推理脚本 📝

用于从训练好的模型生成文本。它不包含 TODO——因为它依赖 `model.py` 中的 `generate()` 方法。

#### 工作流程

```
1. 加载模型（从 checkpoint 或预训练权重）
2. 加载分词器（自定义 meta.pkl 或 GPT-2 tiktoken）
3. 编码起始文本
4. 循环生成 N 个样本，每个样本 max_new_tokens 个 token
5. 解码并打印结果
```

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.8 | 温度参数：<1.0 更确定，>1.0 更随机 |
| `top_k` | 200 | 只保留概率最高的 k 个 token |
| `num_samples` | 10 | 生成样本数量 |
| `max_new_tokens` | 500 | 每个样本的最大 token 数 |

---

### 4. `configurator.py` — 配置系统 ⚙️

一个极简的配置覆盖系统（约 48 行），通过 `exec()` 在调用者的全局命名空间中执行。

#### 使用方式

```bash
# 方式 1：通过配置文件覆盖
python train.py config/train_shakespeare_char.py

# 方式 2：通过命令行参数覆盖
python train.py --batch_size=32 --compile=False

# 方式 3：两者结合
python train.py config/train_shakespeare_char.py --batch_size=64
```

#### 工作原理

1. 遍历 `sys.argv` 中的参数
2. 没有 `=` 的参数被视为配置文件路径，直接 `exec()` 执行
3. 有 `=` 的参数（如 `--key=value`）解析为键值对，覆盖全局变量
4. 自动进行类型检查：新值的类型必须与原始默认值类型一致

---

### 5. `bench.py` — 性能基准测试 ⚡

用于测量模型的训练性能（迭代时间、MFU），不涉及分布式训练。

#### 两种模式

- **简单基准测试**：测量每次迭代的平均时间和 MFU（Model FLOPS Utilization）
- **PyTorch Profiler**：生成详细的性能分析报告，可在 TensorBoard 中查看

---

### 6. `config/` — 预设配置文件 📋

| 配置文件 | 用途 | 关键设置 |
|----------|------|----------|
| `train_gpt2.py` | GPT-2 (124M) 全量训练 | batch=12, block=1024, 600K 步 |
| `train_shakespeare_char.py` | 字符级 Shakespeare 训练 | batch=64, block=256, 6层/6头/384维, 5K 步 |
| `finetune_shakespeare.py` | GPT-2 XL 微调 | 从 gpt2-xl 初始化, lr=3e-5, 20 步 |
| `eval_gpt2*.py` | 评估各尺寸 GPT-2 | 只做评估，不训练 |

**推荐学习路径**：先用 `train_shakespeare_char.py` 验证你的实现，它只需要几分钟就能看到效果。

---

### 7. `data/` — 数据集准备 📊

每个子目录包含一个 `prepare.py` 脚本，负责下载和预处理数据：

| 数据集 | 分词方式 | 输出文件 | 说明 |
|--------|----------|----------|------|
| `shakespeare_char/` | 字符级 | `train.bin`, `val.bin`, `meta.pkl` | 最简单，推荐入门 |
| `shakespeare/` | GPT-2 BPE | `train.bin`, `val.bin` | 使用 tiktoken |
| `openwebtext/` | GPT-2 BPE | `train.bin`, `val.bin` | 大规模数据集，需要大量磁盘空间 |

**数据格式**：所有数据都被预处理为 `uint16` 类型的 NumPy 数组（`.bin` 文件），每个元素是一个 token ID。训练时通过 `np.memmap` 进行内存映射读取，避免将整个数据集加载到内存中。

---

## 核心文件间的协作关系

```
┌─────────────────────────────────────────────────────────┐
│                     用户命令行                           │
│  python train.py config/train_shakespeare_char.py       │
└─────────┬───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐    exec()    ┌──────────────────┐
│     train.py        │◄────────────►│  configurator.py │
│  （训练主控脚本）     │              │  （配置覆盖系统）  │
└──────┬──────────────┘              └──────────────────┘
       │
       │ import GPTConfig, GPT
       ▼
┌─────────────────────┐
│     model.py        │
│  （模型架构定义）     │
│                     │
│  GPTConfig ──────►  │
│  GPT:               │
│  ├── forward()      │ ◄── train.py 调用（训练）
│  └── generate()     │ ◄── sample.py 调用（推理）
└─────────────────────┘
       ▲
       │ import GPTConfig, GPT
       │
┌──────┴──────────────┐
│     sample.py       │
│  （文本生成脚本）     │
└─────────────────────┘

数据流:
data/shakespeare_char/prepare.py ──► train.bin + val.bin
                                          │
                                          ▼
                              train.py: get_batch() ──► model.forward()
```

---

## GPT-2 模型架构图

```
输入 Token IDs: idx (b, t)
         │
         ▼
┌─────────────────────────┐
│    Token Embedding      │  wte: vocab_size → n_embd
│         +               │
│  Position Embedding     │  wpe: block_size → n_embd
│         +               │
│      Dropout            │
└────────┬────────────────┘
         │  (b, t, n_embd)
         ▼
┌─────────────────────────┐
│   Transformer Block ×N  │  ◄── n_layer 个相同的 Block
│  ┌───────────────────┐  │
│  │   LayerNorm       │  │
│  │        │          │  │
│  │ CausalSelfAttn    │  │  Q, K, V 投影 → 多头注意力 → 输出投影
│  │        │          │  │
│  │   + (残差连接)     │  │
│  ├───────────────────┤  │
│  │   LayerNorm       │  │
│  │        │          │  │
│  │      MLP          │  │  升维(4x) → GELU → 降维(1x)
│  │        │          │  │
│  │   + (残差连接)     │  │
│  └───────────────────┘  │
└────────┬────────────────┘
         │  (b, t, n_embd)
         ▼
┌─────────────────────────┐
│    Final LayerNorm      │  ln_f
└────────┬────────────────┘
         │  (b, t, n_embd)
         ▼
┌─────────────────────────┐
│   Language Model Head   │  lm_head: n_embd → vocab_size
│   (权重与 wte 共享)      │  （Weight Tying 技术）
└────────┬────────────────┘
         │  (b, t, vocab_size)
         ▼
    输出 Logits / Loss
```

---

## 超参数速查表（GPT-2 124M）

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `n_layer` | 12 | Transformer Block 数量 |
| `n_head` | 12 | 注意力头数量 |
| `n_embd` | 768 | 嵌入维度 |
| `head_size` | 64 | 每个头的维度 (768/12) |
| `block_size` | 1024 | 最大上下文长度 |
| `vocab_size` | 50304 | 词表大小 (50257 向上取整到 64 的倍数) |
| `dropout` | 0.0 | Dropout 率（预训练时为 0） |
| `bias` | False | 是否使用偏置（False 稍好） |
| `learning_rate` | 6e-4 | 最大学习率 |
| `min_lr` | 6e-5 | 最小学习率 |
| `weight_decay` | 0.1 | 权重衰减系数 |
| `batch_size` | 12 | micro-batch 大小 |
| `gradient_accumulation_steps` | 40 | 梯度累积步数 |
| `max_iters` | 600,000 | 最大训练步数 |
