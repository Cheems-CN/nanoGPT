# nanoGPT 项目架构解析

## 项目概述

nanoGPT 是 Andrej Karpathy 开发的一个**最简 GPT 训练/推理框架**。本仓库在其基础上将所有核心 LLM 代码清空，只保留 TODO 提示和外围基础设施。你的目标是：**从零实现所有核心组件，走通从分词到训练到推理的完整流程**。

---

## 项目当前状态

```
🔴 待实现（核心流程，全部清空为 TODO）
🟢 已提供（外围基础设施，无需修改）
📖 文档（指导你完成实现）
```

### 核心流程（待你实现）

| 文件 | 状态 | 内容 |
|------|------|------|
| `model.py` | 🔴 **全部清空** | GPTConfig、LayerNorm、MLP、CausalSelfAttention、Block、GPT 六个类 |
| `train.py` (核心部分) | 🔴 **清空** | get_batch、estimate_loss、get_lr、训练循环核心步骤 |
| `sample.py` (核心部分) | 🔴 **清空** | 模型加载、分词器加载、文本生成 |
| `data/shakespeare_char/prepare.py` | 🔴 **清空** | 字符级分词与数据预处理 |
| `data/shakespeare/prepare.py` | 🔴 **清空** | BPE 分词与数据预处理 |

### 外围基础设施（已提供）

| 文件 | 状态 | 内容 |
|------|------|------|
| `train.py` (基础设施部分) | 🟢 已提供 | 配置系统、DDP 初始化、模型初始化调度、日志、检查点保存 |
| `configurator.py` | 🟢 已提供 | 命令行/配置文件参数覆盖系统 |
| `bench.py` | 🟢 已提供 | 性能基准测试工具 |
| `config/*.py` | 🟢 已提供 | 各场景预设配置（训练/微调/评估） |
| `data/openwebtext/prepare.py` | 🟢 已提供 | 大规模数据集处理（高级用途） |

---

## 目录结构

```
nanoGPT/
├── model.py                    # 🔴 GPT 模型定义（全部待实现）
├── train.py                    # 🔴🟢 训练脚本（核心待实现，基础设施已提供）
├── sample.py                   # 🔴 推理脚本（全部待实现）
├── configurator.py             # 🟢 配置覆盖系统
├── bench.py                    # 🟢 性能基准测试
├── config/                     # 🟢 预设配置文件
│   ├── train_gpt2.py               # GPT-2 (124M) 全量训练配置
│   ├── train_shakespeare_char.py    # Shakespeare 字符级训练配置（推荐入门）
│   ├── finetune_shakespeare.py      # GPT-2 XL 微调配置
│   └── eval_gpt2*.py               # 各尺寸 GPT-2 评估配置
├── data/                       # 数据集目录
│   ├── shakespeare_char/
│   │   └── prepare.py              # 🔴 字符级分词（待实现）
│   ├── shakespeare/
│   │   └── prepare.py              # 🔴 BPE 分词（待实现）
│   └── openwebtext/
│       └── prepare.py              # 🟢 大规模数据处理（已提供）
├── Architecture.md             # 📖 本文件：项目架构
├── Build_Guide.md              # 📖 从零复现通关指南
├── Engineering_Guide.md        # 📖 工程化文件组织指南
└── README.md                   # 📖 原始 README
```

---

## 你需要实现的完整组件清单

### 第一阶段：数据与分词

| 组件 | 文件 | 功能 |
|------|------|------|
| 字符级分词器 | `data/shakespeare_char/prepare.py` | 字符↔整数映射，生成 train.bin/val.bin/meta.pkl |
| BPE 分词器 | `data/shakespeare/prepare.py` | 使用 tiktoken GPT-2 BPE 编码，生成 train.bin/val.bin |

### 第二阶段：模型架构（全在 `model.py`）

| 组件 | 类名 | 功能 |
|------|------|------|
| 模型配置 | `GPTConfig` | @dataclass，定义所有超参数 |
| 层归一化 | `LayerNorm` | 带可选 bias 的 LayerNorm |
| 前馈网络 | `MLP` | Linear(C→4C) → GELU → Linear(4C→C) → Dropout |
| 因果自注意力 | `CausalSelfAttention` | 多头注意力 + 因果掩码 + Flash Attention |
| Transformer块 | `Block` | Pre-Norm + 残差连接，组合 LN+Attn+LN+MLP |
| GPT 主模型 | `GPT` | 组合所有组件，含 forward/generate/from_pretrained 等 |

### 第三阶段：训练流程（在 `train.py`）

| 组件 | 函数/位置 | 功能 |
|------|-----------|------|
| 数据加载器 | `get_batch()` | 从 .bin 文件随机采样 batch |
| 评估损失 | `estimate_loss()` | 在 train/val 上计算平均 loss |
| 学习率调度 | `get_lr()` | 余弦退火 + 线性预热 |
| 前向传播 | 训练循环内 | 混合精度前向传播 + loss 缩放 |
| 反向传播 | 训练循环内 | GradScaler 反向传播 |
| 参数更新 | 训练循环内 | 梯度裁剪 + AdamW 更新 + 梯度清零 |

### 第四阶段：推理与生成（在 `sample.py`）

| 组件 | 位置 | 功能 |
|------|------|------|
| 模型加载 | sample.py | 从 checkpoint 或预训练权重加载模型 |
| 分词器加载 | sample.py | 加载字符级 meta.pkl 或 GPT-2 BPE |
| 文本生成 | sample.py | 编码→生成→解码→打印 |

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

## 类层级关系（你需要构建的）

```
GPT（主模型）
├── nn.Embedding (wte)           # Token 嵌入层
├── nn.Embedding (wpe)           # 位置嵌入层
├── nn.Dropout (drop)            # 嵌入层 Dropout
├── nn.ModuleList (h)            # N 个 Transformer Block
│   └── Block × N
│       ├── LayerNorm (ln_1)
│       ├── CausalSelfAttention (attn)
│       │   ├── nn.Linear (c_attn)   # Q/K/V 合并投影
│       │   ├── nn.Linear (c_proj)   # 输出投影
│       │   ├── nn.Dropout (attn_dropout)
│       │   └── nn.Dropout (resid_dropout)
│       ├── LayerNorm (ln_2)
│       └── MLP (mlp)
│           ├── nn.Linear (c_fc)     # 升维
│           ├── nn.GELU (gelu)
│           ├── nn.Linear (c_proj)   # 降维
│           └── nn.Dropout (dropout)
├── LayerNorm (ln_f)             # 最终 LayerNorm
└── nn.Linear (lm_head)         # LM 头（与 wte 共享权重）
```

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
       │ from model import GPTConfig, GPT
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
       │ from model import GPTConfig, GPT
       │
┌──────┴──────────────┐
│     sample.py       │
│  （文本生成脚本）     │
└─────────────────────┘

数据流:
data/shakespeare_char/prepare.py ──► train.bin + val.bin + meta.pkl
                                          │
                                          ▼
                              train.py: get_batch() ──► model.forward()
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
| `bias` | False | 是否使用偏置 |
| `learning_rate` | 6e-4 | 最大学习率 |
| `min_lr` | 6e-5 | 最小学习率 |
| `weight_decay` | 0.1 | 权重衰减系数 |
| `batch_size` | 12 | micro-batch 大小 |
| `gradient_accumulation_steps` | 40 | 梯度累积步数 |
| `max_iters` | 600,000 | 最大训练步数 |

---

## 推荐实现路线

详见 [Build_Guide.md](Build_Guide.md) 获取完整的从零复现通关指南。

总体路线：**数据准备 → 模型架构 → 训练循环 → 推理生成 → 微调**

关于项目文件组织的最佳实践，参见 [Engineering_Guide.md](Engineering_Guide.md)。
