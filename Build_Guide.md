# nanoGPT 从零复现通关指南

## 前言

本指南带你从零实现一个完整的 GPT 语言模型——从数据处理到模型训练到文本生成。所有核心代码已被清空，你需要根据 TODO 提示和本指南自己编写全部实现。

### 总体路线

```
阶段 0: 环境准备                    ── 安装依赖
阶段 1: 数据与分词                  ── 理解 token 是什么
阶段 2: 模型架构 (model.py)         ── 从零搭建 GPT-2
阶段 3: 训练流程 (train.py)         ── 让模型学会语言
阶段 4: 推理生成 (sample.py)        ── 让模型说话
阶段 5: 微调 (Fine-tuning)         ── 站在巨人肩膀上
```

### 依赖关系

```
阶段 1 (数据)  ──────────────────────────────────────┐
                                                      ▼
阶段 2 (模型)  ──► 阶段 3 (训练) ──► 阶段 4 (推理) ──► 阶段 5 (微调)
```

---

## 阶段 0：环境准备

```bash
pip install torch numpy tiktoken requests
```

---

## 阶段 1：数据与分词 ★☆☆☆☆

> **文件**: `data/shakespeare_char/prepare.py` 和 `data/shakespeare/prepare.py`

### 目标

理解分词（Tokenization）是 LLM 的第一步：将人类可读的文本转换为模型可以处理的整数序列。

### 1.1 字符级分词（推荐先做）

**文件**: `data/shakespeare_char/prepare.py`

这是最简单的分词方案——每个字符就是一个 token。

你需要实现:
- 下载 Shakespeare 数据集
- 统计所有唯一字符，构建词表
- 实现 encode（字符串→整数列表）和 decode（整数列表→字符串）
- 划分训练集(90%)/验证集(10%)
- 将编码后的数据保存为 `.bin` 二进制文件（uint16 NumPy 数组）
- 将词表元信息保存为 `meta.pkl`

**验证方法**:
```bash
cd data/shakespeare_char && python prepare.py && cd ../..
# 应该生成 train.bin, val.bin, meta.pkl
# 输出应显示 vocab_size ≈ 65, train ≈ 1M tokens, val ≈ 111K tokens
```

### 1.2 BPE 分词（第二步）

**文件**: `data/shakespeare/prepare.py`

使用 GPT-2 的 BPE (Byte Pair Encoding) 分词器，需要 `tiktoken` 库。

你需要实现:
- 下载同样的数据集
- 使用 tiktoken 加载 GPT-2 分词器
- 编码文本并保存为 `.bin` 文件

**思考题**: 为什么 BPE 的 token 数量（~300K）比字符级（~1M）少得多？

### 通关检查

- [ ] `data/shakespeare_char/prepare.py` 运行成功，生成 train.bin、val.bin、meta.pkl
- [ ] `data/shakespeare/prepare.py` 运行成功，生成 train.bin、val.bin
- [ ] 理解字符级分词和 BPE 分词的区别

---

## 阶段 2：模型架构 ★★★★☆

> **文件**: `model.py`
>
> 这是整个项目的核心。你需要从零定义 6 个类，构建完整的 GPT-2 模型。

### 建议实现顺序

```
2.1 GPTConfig       ★☆☆☆☆  配置类（5 分钟）
2.2 LayerNorm       ★☆☆☆☆  层归一化（10 分钟）
2.3 MLP             ★★☆☆☆  前馈网络（15 分钟）
2.4 CausalSelfAttn  ★★★★☆  因果自注意力（核心难点，45 分钟）
2.5 Block           ★★☆☆☆  Transformer Block（10 分钟）
2.6 GPT             ★★★☆☆  完整模型（30 分钟）
```

### 2.1 GPTConfig ★☆☆☆☆

**你需要**: 用 `@dataclass` 定义模型的超参数配置类。

**关键知识点**:
- Python dataclass 的用法
- GPT-2 的标准超参数是什么

**验证**: 能被其他类的 `__init__` 正确读取。

### 2.2 LayerNorm ★☆☆☆☆

**你需要**: 实现一个支持可选 bias 的 LayerNorm。

**关键知识点**:
- Layer Normalization 的作用：稳定训练，加速收敛
- 为什么需要自定义而不用 PyTorch 原生的
- `F.layer_norm` 的使用

**验证**:
```python
from model import GPTConfig, LayerNorm
import torch
ln = LayerNorm(64, bias=True)
x = torch.randn(2, 10, 64)
y = ln(x)
assert y.shape == x.shape
print("✅ LayerNorm OK")
```

### 2.3 MLP ★★☆☆☆

**你需要**: 实现 Transformer 中的前馈网络。

**关键知识点**:
- 升维-降维结构（为什么先扩大 4 倍再压回去？）
- GELU 激活函数和 ReLU 的区别
- Dropout 正则化

**验证**:
```python
from model import GPTConfig, MLP
import torch
config = GPTConfig(n_embd=64, bias=True)
mlp = MLP(config)
x = torch.randn(2, 10, 64)
y = mlp(x)
assert y.shape == (2, 10, 64)
print("✅ MLP OK")
```

### 2.4 CausalSelfAttention ★★★★☆

**你需要**: 实现多头因果自注意力——这是 Transformer 的灵魂。

**关键知识点**:
- 注意力机制的数学公式: Attention(Q,K,V) = softmax(QK^T/√d_k)·V
- 为什么需要缩放（√d_k）：防止 softmax 饱和
- 因果掩码的作用：语言模型不能"偷看"未来
- 多头注意力：让模型同时关注不同类型的信息
- Flash Attention：PyTorch 2.0 的高效实现

**核心挑战**:
- Q, K, V 的 reshape 操作（从 `(B,T,C)` 到 `(B,nh,T,hs)`）
- 手动实现注意力时的因果掩码
- 两条代码路径：Flash Attention vs 手动实现

**验证**:
```python
from model import GPTConfig, CausalSelfAttention
import torch
config = GPTConfig(n_embd=64, n_head=4, block_size=32, bias=True, dropout=0.0)
attn = CausalSelfAttention(config)
x = torch.randn(2, 10, 64)
y = attn(x)
assert y.shape == (2, 10, 64)

# 因果性测试：改变最后一个位置的输入，前面的输出不应该变
x2 = x.clone()
x2[:, -1, :] = torch.randn(2, 64)
y2 = attn(x2)
assert torch.allclose(y[:, :-1, :], y2[:, :-1, :], atol=1e-5), "因果性检查失败！"
print("✅ CausalSelfAttention OK")
```

### 2.5 Block ★★☆☆☆

**你需要**: 组装 Pre-Norm Transformer Block。

**关键知识点**:
- Pre-Norm vs Post-Norm 架构
- 残差连接（为什么 `x = x + sublayer(x)` 如此重要？）

**验证**: 依赖 2.2-2.4 都正确实现。

### 2.6 GPT ★★★☆☆

**你需要**: 实现完整的 GPT 类，包含构造函数和所有方法。

**构造函数关键知识点**:
- nn.ModuleDict 和 nn.ModuleList 的使用
- Weight Tying（为什么 token 嵌入和 LM 头共享权重？）
- 权重初始化策略（为什么残差投影需要特殊缩放？）

**forward() 关键知识点**:
- Token 嵌入 + 位置嵌入的组合
- 训练模式 vs 推理模式的区别（推理时只计算最后位置）
- 交叉熵损失计算

**generate() 关键知识点**:
- 自回归生成循环
- 温度参数对生成多样性的控制
- Top-K 采样策略

**from_pretrained() 关键知识点**:
- 如何从 HuggingFace 加载预训练权重
- Conv1D vs Linear 的权重转置问题

**configure_optimizers() 关键知识点**:
- 权重衰减分组：哪些参数应该衰减？
- Fused AdamW 优化

**验证**:
```python
import torch
from model import GPTConfig, GPT

config = GPTConfig(block_size=32, vocab_size=100, n_layer=2,
                   n_head=4, n_embd=64, dropout=0.0, bias=True)
model = GPT(config)

# 训练模式测试
idx = torch.randint(0, 100, (2, 10))
targets = torch.randint(0, 100, (2, 10))
logits, loss = model(idx, targets)
assert logits.shape == (2, 10, 100)
assert loss is not None
import math
assert abs(loss.item() - math.log(100)) < 1.0  # 初始 loss ≈ ln(vocab_size)

# 推理模式测试
logits, loss = model(idx)
assert logits.shape == (2, 1, 100)
assert loss is None

# 生成测试
model.eval()
start = torch.zeros((1, 1), dtype=torch.long)
with torch.no_grad():
    generated = model.generate(start, max_new_tokens=20)
assert generated.shape == (1, 21)
print("✅ GPT 全部测试通过！")
```

### 阶段 2 通关检查

- [ ] `GPTConfig` 可被正确实例化
- [ ] `LayerNorm` 输入输出形状一致
- [ ] `MLP` 前馈网络正确运行
- [ ] `CausalSelfAttention` 通过因果性测试
- [ ] `Block` 正确组合子层
- [ ] `GPT.forward()` 训练/推理模式都正确
- [ ] `GPT.generate()` 能自回归生成
- [ ] `GPT.from_pretrained()` 能加载 GPT-2 权重（需要 transformers 库）
- [ ] 初始 loss ≈ ln(vocab_size)

---

## 阶段 3：训练流程 ★★★☆☆

> **文件**: `train.py` 中的核心函数和训练循环

### 3.1 数据加载器 get_batch() ★★☆☆☆

**你需要**: 从二进制文件中随机采样训练/验证 batch。

**关键知识点**:
- `np.memmap`：内存映射文件读取（避免将整个数据集加载到内存）
- 输入序列和目标序列的关系：目标 = 输入右移一位
- GPU 数据传输优化：`pin_memory()` + `non_blocking=True`

### 3.2 学习率调度器 get_lr() ★★☆☆☆

**你需要**: 实现余弦退火 + 线性预热的学习率调度。

**关键知识点**:
- 线性预热：为什么训练初期需要小学习率？
- 余弦退火：比线性衰减更平滑的学习率下降
- 最小学习率：训练末期保持微小的学习率

**验证**:
```python
import math
# 设置参数后手动验证
# get_lr(0) 应接近 0
# get_lr(warmup_iters) 应等于 learning_rate
# get_lr(lr_decay_iters) 应等于 min_lr
```

### 3.3 评估损失 estimate_loss() ★☆☆☆☆

**你需要**: 在 train/val 集上计算平均损失。

**关键知识点**:
- `model.eval()` vs `model.train()` 的区别（Dropout 行为）
- 混合精度上下文 `ctx`
- `@torch.no_grad()` 的作用

### 3.4 训练循环核心步骤 ★★★★☆

**你需要**: 实现前向传播、反向传播、参数更新三个步骤。

**关键知识点**:
- **梯度累积**: 为什么要将 loss 除以 `gradient_accumulation_steps`？
- **GradScaler**: FP16 训练中如何防止梯度下溢？
  - `scaler.scale(loss)` → 放大 loss
  - `.backward()` → 计算放大后的梯度
  - `scaler.unscale_(optimizer)` → 还原梯度
  - `scaler.step(optimizer)` → 更新参数
  - `scaler.update()` → 调整缩放因子
- **梯度裁剪**: `clip_grad_norm_` 防止梯度爆炸
- **零梯度**: `set_to_none=True` 比填零更省内存

**验证**:
```bash
# 准备数据（阶段 1 完成后）
cd data/shakespeare_char && python prepare.py && cd ../..

# 最小配置训练测试
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --eval_iters=5 \
    --max_iters=10 --eval_interval=5 --log_interval=1 \
    --batch_size=4 --block_size=32

# 预期: 初始 loss ≈ ln(65) ≈ 4.17，loss 应逐步下降
```

### 阶段 3 通关检查

- [ ] `get_batch()` 返回正确形状的 (x, y) 张量
- [ ] `get_lr()` 三个阶段的值都正确
- [ ] `estimate_loss()` 能计算 train/val 平均 loss
- [ ] 训练循环能正常运行，loss 逐步下降
- [ ] 初始 loss 接近 ln(vocab_size)

---

## 阶段 4：推理与生成 ★★☆☆☆

> **文件**: `sample.py`

### 你需要实现

1. **模型加载**: 从 checkpoint 或预训练权重加载 GPT 模型
2. **分词器加载**: 从 meta.pkl 或 tiktoken 加载 encode/decode 函数
3. **文本生成**: 编码起始文本 → 调用 model.generate() → 解码并打印

**验证**:
```bash
# 需要先完成阶段 3 的训练，生成 checkpoint
python sample.py --out_dir=out --device=cpu --compile=False \
    --num_samples=3 --max_new_tokens=200
# 应该能看到生成的类 Shakespeare 风格文本
```

### 阶段 4 通关检查

- [ ] 能从 checkpoint 加载模型
- [ ] 能正确加载字符级分词器（meta.pkl）
- [ ] 能生成文本并打印

---

## 阶段 5：微调 ★★★☆☆

> **使用**: `config/finetune_shakespeare.py` 配置

### 微调流程

微调是在预训练模型的基础上，用少量特定领域数据继续训练。

```bash
# 1. 确保 from_pretrained() 已实现
# 2. 准备 Shakespeare 数据
cd data/shakespeare_char && python prepare.py && cd ../..

# 3. 运行微调（需要 GPU 和 transformers 库）
pip install transformers
python train.py config/finetune_shakespeare.py
```

**关键理解**:
- 微调 vs 从零训练：微调从预训练权重出发，用更小的学习率
- `init_from='gpt2-xl'`：加载 15 亿参数的 GPT-2 XL 模型
- 微调只需要很少的迭代（20 步 vs 600K 步）

### 阶段 5 通关检查

- [ ] `GPT.from_pretrained()` 能正确加载 GPT-2 权重
- [ ] 微调后的模型生成质量明显提升
- [ ] 理解微调和从零训练的区别

---

## 🏆 最终通关测试

完成所有阶段后，运行以下端到端测试：

### 测试 1：完整训练 + 生成

```bash
# 准备数据
cd data/shakespeare_char && python prepare.py && cd ../..

# 训练（约 3 分钟，视硬件而定）
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --max_iters=1000 --eval_interval=250

# 生成
python sample.py --out_dir=out --device=cpu --compile=False \
    --num_samples=3 --max_new_tokens=200
```

### 测试 2：加载预训练 GPT-2（需要 GPU）

```bash
python sample.py --init_from=gpt2 --device=cuda --num_samples=3 --max_new_tokens=100
```

---

## 你学到了什么

通过完成全部阶段，你已经从零实现了:

| 阶段 | 组件 | 核心概念 |
|------|------|----------|
| **数据** | 分词器 | 字符级分词、BPE、词表构建、数据序列化 |
| **模型** | GPTConfig | dataclass、超参数设计 |
| **模型** | LayerNorm | 层归一化、可选 bias |
| **模型** | MLP | 前馈网络、GELU、升维-降维 |
| **模型** | CausalSelfAttention | 注意力机制、因果掩码、多头、Flash Attention |
| **模型** | Block | Pre-Norm、残差连接 |
| **模型** | GPT | Embedding、Weight Tying、交叉熵、自回归生成 |
| **训练** | get_batch | memmap、数据采样、GPU 传输 |
| **训练** | get_lr | 余弦退火、线性预热 |
| **训练** | 训练循环 | 梯度累积、混合精度、GradScaler、梯度裁剪 |
| **推理** | sample.py | 模型加载、温度采样、Top-K |
| **微调** | Fine-tuning | 预训练权重加载、小学习率微调 |

这些概念涵盖了现代大语言模型（GPT-4、LLaMA、Mistral 等）的**核心架构和训练技术**。
