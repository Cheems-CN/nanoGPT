# nanoGPT 手搓复现通关指南

## 前言

欢迎来到 nanoGPT 手搓复现之旅！本指南将带你**由浅入深**地填写所有 TODO，最终完整实现一个可以训练和生成文本的 GPT 模型。

### 学习路线图

```
关卡 1: MLP 前馈网络          ★☆☆☆☆  入门热身
关卡 2: Transformer Block    ★★☆☆☆  理解残差连接
关卡 3: 因果自注意力          ★★★★☆  核心中的核心
关卡 4: GPT 前向传播          ★★★☆☆  组装完整模型
关卡 5: 自回归文本生成         ★★★☆☆  让模型说话
关卡 6: 学习率调度器           ★★☆☆☆  训练技巧
关卡 7: 训练循环核心           ★★★★☆  梯度累积与混合精度
```

### 前置准备

```bash
# 1. 安装依赖
pip install torch numpy tiktoken

# 2. 准备 Shakespeare 字符级数据集（最简单的数据集，几秒即可完成）
cd data/shakespeare_char
python prepare.py
cd ../..
```

---

## 关卡 1：MLP 前馈网络 ★☆☆☆☆

> **文件**: `model.py` → `MLP.forward()`
>
> **目标**: 实现 Transformer 中的前馈神经网络

### 理论背景

MLP（Multi-Layer Perceptron）是 Transformer 中每个 Block 的第二个子层。它对每个位置的表示独立地进行非线性变换：

```
MLP(x) = Dropout(Linear_down(GELU(Linear_up(x))))
```

其中：
- `Linear_up`: 将维度从 `n_embd` 扩展到 `4 * n_embd`（升维）
- `GELU`: 激活函数，比 ReLU 更平滑
- `Linear_down`: 将维度从 `4 * n_embd` 压缩回 `n_embd`（降维）
- `Dropout`: 正则化

### 实现提示

```python
def forward(self, x):
    # x: (B, T, C) 其中 C = n_embd
    x = self.c_fc(x)      # (B, T, C) -> (B, T, 4*C)  升维
    x = self.gelu(x)       # (B, T, 4*C) -> (B, T, 4*C) 激活
    x = self.c_proj(x)     # (B, T, 4*C) -> (B, T, C)  降维
    x = self.dropout(x)    # (B, T, C) -> (B, T, C)    正则化
    return x
```

### 调试建议

```python
# 验证代码（可以在 Python REPL 中测试）
import torch
from model import GPTConfig, MLP

config = GPTConfig(n_embd=64, bias=True)
mlp = MLP(config)
x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, embd=64)
y = mlp(x)
assert y.shape == (2, 10, 64), f"期望 (2, 10, 64)，实际 {y.shape}"
print("✅ MLP 测试通过！")
```

### 通关检查

- [ ] `MLP.forward()` 输入输出形状一致：`(B, T, C) → (B, T, C)`
- [ ] 中间维度为 `4 * C`
- [ ] 使用了 GELU 激活函数
- [ ] 最后应用了 Dropout

---

## 关卡 2：Transformer Block ★★☆☆☆

> **文件**: `model.py` → `Block.forward()`
>
> **目标**: 用 Pre-Norm 架构组装注意力和 MLP 子层
>
> **前置条件**: 完成关卡 1（MLP）

### 理论背景

GPT-2 使用 **Pre-Norm**（前置归一化）架构，与原始 Transformer 论文中的 Post-Norm 不同：

```
# Post-Norm（原始 Transformer）:
x = LayerNorm(x + SubLayer(x))

# Pre-Norm（GPT-2）:
x = x + SubLayer(LayerNorm(x))
```

Pre-Norm 的优势：梯度流更稳定，不容易出现训练不稳定的问题。

**残差连接**是深度学习中最重要的技巧之一：
- 数学上：`output = input + F(input)`
- 作用：让梯度可以直接跳过中间层回传，缓解梯度消失
- 直觉：给网络一条"高速公路"，即使中间层学不到东西，至少可以做恒等映射

### 实现提示

```python
def forward(self, x):
    # x: (B, T, C)
    x = x + self.attn(self.ln_1(x))    # 自注意力子层 + 残差连接
    x = x + self.mlp(self.ln_2(x))     # MLP 子层 + 残差连接
    return x
```

### 调试建议

```python
# 注意：此时 CausalSelfAttention.forward() 还未实现
# 所以暂时无法直接测试 Block
# 可以先确认代码结构正确，等关卡 3 完成后再一起测试

# 如果你想提前测试 Block 的结构（跳过注意力），可以临时这样：
# 先在 CausalSelfAttention.forward() 中写一个临时占位：
# def forward(self, x): return torch.zeros_like(x)
```

### 通关检查

- [ ] 使用了 Pre-Norm 顺序：先 LayerNorm，再子层
- [ ] 两个子层都有残差连接
- [ ] 输入输出形状一致：`(B, T, C) → (B, T, C)`

---

## 关卡 3：因果自注意力 ★★★★☆

> **文件**: `model.py` → `CausalSelfAttention.forward()`
>
> **目标**: 实现多头因果自注意力机制——Transformer 的灵魂
>
> **前置条件**: 理解注意力机制的数学原理

### 理论背景

#### 注意力机制的核心公式

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- **Q (Query)**: 查询向量——"我在找什么？"
- **K (Key)**: 键向量——"我有什么？"
- **V (Value)**: 值向量——"我能提供什么信息？"
- **d_k**: 每个头的维度，用于缩放（防止点积值过大导致 softmax 饱和）

#### 因果掩码

在语言模型中，生成第 t 个 token 时只能看到位置 0 到 t-1 的信息（不能"偷看"未来）。这通过一个下三角矩阵实现：

```
掩码矩阵（T=4）:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

将掩码为 0 的位置填充为 `-inf`，这样 softmax 后这些位置的注意力权重就变为 0。

#### 多头注意力

将 Q, K, V 分成 `n_head` 个子空间，各自独立计算注意力，然后拼接结果。这让模型能同时关注不同类型的信息（如语法关系、语义关系等）。

### 实现步骤（分三步）

#### 步骤 1：计算 Q, K, V 并 reshape

```python
# 输入: x (B, T, C)
# 通过一个大的线性层同时计算 Q, K, V
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# 此时 q, k, v 的形状都是 (B, T, C)

# 将每个张量 reshape 为多头形式
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
# nh = n_head, hs = head_size = C // n_head
```

#### 步骤 2：计算注意力

```python
# 路径 A：Flash Attention（高效实现）
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True
    )

# 路径 B：手动实现（帮助理解原理）
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
```

#### 步骤 3：合并多头输出

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, C)
y = self.resid_dropout(self.c_proj(y))
return y
```

### 调试建议

```python
import torch
from model import GPTConfig, CausalSelfAttention

config = GPTConfig(n_embd=64, n_head=4, block_size=32, bias=True, dropout=0.0)
attn = CausalSelfAttention(config)
x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, embd=64)
y = attn(x)
assert y.shape == (2, 10, 64), f"期望 (2, 10, 64)，实际 {y.shape}"
print("✅ CausalSelfAttention 测试通过！")

# 高级验证：检查因果性
# 修改输入序列的最后一个位置，前面位置的输出不应该改变
x2 = x.clone()
x2[:, -1, :] = torch.randn(2, 64)  # 只改变最后一个位置
y2 = attn(x2)
# 前面位置的输出应该完全相同
assert torch.allclose(y[:, :-1, :], y2[:, :-1, :], atol=1e-5), "因果性检查失败！前面位置不应受后面位置影响"
print("✅ 因果性检查通过！")
```

### 通关检查

- [ ] Q, K, V 从 `c_attn` 正确切分
- [ ] 正确 reshape 为多头形式 `(B, nh, T, hs)`
- [ ] 注意力分数使用 `sqrt(d_k)` 缩放
- [ ] 因果掩码正确应用（手动路径）
- [ ] Flash Attention 路径正确调用
- [ ] 多头输出正确合并回 `(B, T, C)`
- [ ] 应用了输出投影和残差 Dropout

---

## 关卡 4：GPT 前向传播 ★★★☆☆

> **文件**: `model.py` → `GPT.forward()`
>
> **目标**: 组装完整的 GPT 前向传播管线
>
> **前置条件**: 完成关卡 1-3

### 理论背景

GPT 的前向传播将所有组件串联起来：

```
Token IDs → Token Embedding + Position Embedding → Dropout
    → Transformer Block × N → Final LayerNorm → LM Head → Logits
```

#### 训练 vs 推理的区别

- **训练时**：计算所有位置的 logits，用交叉熵损失与 targets 对比
- **推理时**：只需要最后一个位置的 logits（用于预测下一个 token），这是一个重要的优化

#### 交叉熵损失

交叉熵衡量模型预测分布与真实分布之间的差距：

$$L = -\frac{1}{N}\sum_{i} \log P(y_i | x_i)$$

在 PyTorch 中，`F.cross_entropy` 接受 logits（未归一化的分数），内部自动做 softmax。

### 实现提示

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, ...
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    # 步骤 1：嵌入
    tok_emb = self.transformer.wte(idx)      # (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)      # (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)  # (b, t, n_embd)

    # 步骤 2：Transformer Blocks
    for block in self.transformer.h:
        x = block(x)

    # 步骤 3：最终 LayerNorm
    x = self.transformer.ln_f(x)

    # 步骤 4：计算 logits 和 loss
    if targets is not None:
        logits = self.lm_head(x)  # (b, t, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (b*t, vocab_size)
            targets.view(-1),                   # (b*t,)
            ignore_index=-1
        )
    else:
        logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
        loss = None

    return logits, loss
```

### 调试建议

```python
import torch
from model import GPTConfig, GPT

# 创建一个小模型用于测试
config = GPTConfig(
    block_size=32, vocab_size=100, n_layer=2,
    n_head=4, n_embd=64, dropout=0.0, bias=True
)
model = GPT(config)

# 测试推理模式
idx = torch.randint(0, 100, (2, 10))  # (batch=2, seq_len=10)
logits, loss = model(idx)
assert logits.shape == (2, 1, 100), f"推理模式：期望 (2, 1, 100)，实际 {logits.shape}"
assert loss is None, "推理模式 loss 应为 None"
print("✅ GPT 推理模式测试通过！")

# 测试训练模式
targets = torch.randint(0, 100, (2, 10))
logits, loss = model(idx, targets)
assert logits.shape == (2, 10, 100), f"训练模式：期望 (2, 10, 100)，实际 {logits.shape}"
assert loss is not None and loss.dim() == 0, "训练模式 loss 应为标量"
print(f"✅ GPT 训练模式测试通过！Loss = {loss.item():.4f}")

# 验证 loss 的合理范围（随机初始化时，loss ≈ -ln(1/vocab_size) = ln(vocab_size)）
import math
expected_loss = math.log(100)  # ≈ 4.605
assert abs(loss.item() - expected_loss) < 1.0, f"Loss {loss.item():.2f} 偏离预期值 {expected_loss:.2f} 过多"
print(f"✅ Loss 范围合理（预期 ≈ {expected_loss:.2f}，实际 = {loss.item():.2f}）")
```

### 通关检查

- [ ] Token 嵌入和位置嵌入正确相加
- [ ] 通过了所有 Transformer Blocks
- [ ] 应用了最终 LayerNorm
- [ ] 训练模式：返回所有位置的 logits + 交叉熵 loss
- [ ] 推理模式：只返回最后一个位置的 logits + loss=None
- [ ] 随机初始化时 loss ≈ ln(vocab_size)

---

## 关卡 5：自回归文本生成 ★★★☆☆

> **文件**: `model.py` → `GPT.generate()`
>
> **目标**: 实现自回归解码——让模型"说话"
>
> **前置条件**: 完成关卡 4

### 理论背景

自回归生成的核心思想：**一次生成一个 token，然后把它追加到输入中，再预测下一个 token**。

```
输入:  "The cat"
第1步: "The cat" → 模型预测 → "sat"  → "The cat sat"
第2步: "The cat sat" → 模型预测 → "on" → "The cat sat on"
第3步: "The cat sat on" → 模型预测 → "the" → "The cat sat on the"
...
```

#### 温度（Temperature）

温度参数控制生成的随机性：
- `T → 0`: 总是选择概率最高的 token（贪婪解码）
- `T = 1`: 按原始概率分布采样
- `T → ∞`: 均匀随机采样

数学上：`P(token_i) = softmax(logit_i / T)`

#### Top-K 采样

只保留概率最高的 K 个 token，其余概率设为 0：
- 防止模型生成极低概率的怪异 token
- K 越小，生成越保守

### 实现提示

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 裁剪上下文
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # 前向传播
        logits, _ = self(idx_cond)
        # 温度缩放
        logits = logits[:, -1, :] / temperature
        # Top-K 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # 采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### 调试建议

```python
import torch
from model import GPTConfig, GPT

config = GPTConfig(
    block_size=32, vocab_size=100, n_layer=2,
    n_head=4, n_embd=64, dropout=0.0, bias=True
)
model = GPT(config)
model.eval()

# 基本生成测试
start = torch.zeros((1, 1), dtype=torch.long)  # 从 token 0 开始
with torch.no_grad():
    generated = model.generate(start, max_new_tokens=20, temperature=1.0)
assert generated.shape == (1, 21), f"期望 (1, 21)，实际 {generated.shape}"
print(f"✅ 生成测试通过！生成的 token IDs: {generated[0].tolist()}")

# 测试上下文窗口裁剪
long_start = torch.zeros((1, 50), dtype=torch.long)  # 超过 block_size=32
with torch.no_grad():
    generated = model.generate(long_start, max_new_tokens=5)
assert generated.shape == (1, 55), f"期望 (1, 55)，实际 {generated.shape}"
print("✅ 上下文裁剪测试通过！")

# 温度测试：低温度应该产生更一致的结果
with torch.no_grad():
    torch.manual_seed(42)
    gen_low_temp = model.generate(start.clone(), max_new_tokens=10, temperature=0.01)
    torch.manual_seed(42)
    gen_low_temp2 = model.generate(start.clone(), max_new_tokens=10, temperature=0.01)
assert torch.equal(gen_low_temp, gen_low_temp2), "相同种子 + 低温度应产生相同结果"
print("✅ 温度控制测试通过！")
```

### 通关检查

- [ ] 循环 `max_new_tokens` 次
- [ ] 正确处理上下文窗口超限（裁剪到 `block_size`）
- [ ] 温度缩放正确
- [ ] Top-K 采样正确
- [ ] 每步追加新 token 到序列末尾
- [ ] 返回完整序列

---

## 🎉 里程碑检查点：模型已完成！

到这里，`model.py` 中的所有 TODO 都已完成。你可以用以下方式验证整体正确性：

```python
import torch
from model import GPTConfig, GPT

# 创建完整模型
config = GPTConfig(
    block_size=64, vocab_size=256, n_layer=4,
    n_head=4, n_embd=128, dropout=0.0, bias=True
)
model = GPT(config)

# 前向传播测试
idx = torch.randint(0, 256, (4, 32))
targets = torch.randint(0, 256, (4, 32))
logits, loss = model(idx, targets)
print(f"Logits shape: {logits.shape}")  # (4, 32, 256)
print(f"Loss: {loss.item():.4f}")       # ≈ ln(256) ≈ 5.545

# 生成测试
model.eval()
start = torch.zeros((1, 1), dtype=torch.long)
with torch.no_grad():
    output = model.generate(start, max_new_tokens=50, temperature=0.8, top_k=40)
print(f"Generated: {output[0].tolist()}")

print("\n🎉 恭喜！model.py 的所有核心实现已完成！")
```

---

## 关卡 6：学习率调度器 ★★☆☆☆

> **文件**: `train.py` → `get_lr()`
>
> **目标**: 实现余弦退火 + 线性预热的学习率调度

### 理论背景

现代大模型训练通常使用"先预热再衰减"的学习率策略：

```
学习率
  ↑
  │    ╱‾‾‾╲
  │   ╱      ╲
  │  ╱        ╲
  │ ╱          ╲___________
  │╱                        ← min_lr
  └──────────────────────→ 迭代步数
   ↑预热    ↑余弦退火     ↑恒定
```

1. **线性预热**（0 ~ warmup_iters 步）：从小学习率线性增长到最大学习率
   - 目的：避免训练初期梯度不稳定导致的参数剧烈震荡

2. **余弦退火**（warmup_iters ~ lr_decay_iters 步）：按余弦函数平滑衰减
   - 公式：`coeff = 0.5 * (1 + cos(π * decay_ratio))`
   - 优点：比线性衰减更平滑，在衰减末期变化更慢

3. **恒定最小值**（lr_decay_iters 步之后）：保持最小学习率

### 实现提示

```python
def get_lr(it):
    # 阶段 1：线性预热
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 阶段 3：超过衰减步数后返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 阶段 2：余弦退火
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

### 调试建议

```python
import math

# 模拟参数
learning_rate = 6e-4
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# 手动实现进行验证
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 验证关键节点
assert get_lr(0) < learning_rate, "预热阶段初始 lr 应很小"
assert abs(get_lr(warmup_iters) - learning_rate) < 1e-6, "预热结束时 lr 应接近最大值"
assert abs(get_lr(lr_decay_iters) - min_lr) < 1e-6, "衰减结束时 lr 应等于最小值"
assert get_lr(lr_decay_iters + 100) == min_lr, "超过衰减步数后 lr 应恒定为最小值"
print("✅ 学习率调度器测试通过！")
```

### 通关检查

- [ ] 预热阶段：学习率从接近 0 线性增长
- [ ] 余弦退火阶段：学习率平滑衰减
- [ ] 恒定阶段：学习率保持 `min_lr`
- [ ] 边界条件正确

---

## 关卡 7：训练循环核心 ★★★★☆

> **文件**: `train.py` → 训练主循环中的三个 TODO
>
> **目标**: 实现前向传播、反向传播、梯度更新的完整循环
>
> **前置条件**: 完成关卡 1-6

### 理论背景

#### 梯度累积

当 GPU 内存不足以容纳大 batch 时，可以把大 batch 拆成多个 micro-batch：

```
大 batch (batch_size=128)
= micro-batch_1 (32) + micro-batch_2 (32) + micro-batch_3 (32) + micro-batch_4 (32)

每个 micro-batch 的 loss 除以 4 再 backward
累积 4 次梯度后，效果等价于一个大 batch
```

#### 混合精度训练与 GradScaler

FP16 精度范围有限，梯度可能下溢（变为 0）。GradScaler 的工作流程：

```
1. scaler.scale(loss)     → 将 loss 放大（如乘以 65536）
2. .backward()            → 在放大后的 loss 上计算梯度（梯度也被放大了）
3. scaler.unscale_(opt)   → 将梯度还原到真实尺度
4. clip_grad_norm_()      → 在真实尺度上裁剪梯度
5. scaler.step(opt)       → 用真实梯度更新参数
6. scaler.update()        → 调整缩放因子
```

### 实现提示

#### TODO 1：评估阶段的前向传播

```python
# 在 estimate_loss() 函数的内部循环中:
with ctx:
    logits, loss = model(X, Y)
losses[k] = loss.item()
```

#### TODO 2：前向传播 + Loss 缩放

```python
# 在训练循环的 micro-step 中:
with ctx:
    logits, loss = model(X, Y)
    loss = loss / gradient_accumulation_steps
```

#### TODO 3：反向传播

```python
scaler.scale(loss).backward()
```

#### TODO 4：梯度裁剪 + 优化器更新 + 梯度清零

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

### 调试建议

```bash
# 准备数据
cd data/shakespeare_char && python prepare.py && cd ../..

# 用最小配置测试训练（不编译，CPU 模式，少量迭代）
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --eval_iters=5 \
    --max_iters=10 --eval_interval=5 --log_interval=1 \
    --batch_size=4 --block_size=32
```

如果一切正常，你会看到类似输出：

```
tokens per iteration will be: ...
Initializing a new model from scratch
number of parameters: ...M
step 0: train loss 4.XXXX, val loss 4.XXXX
iter 0: loss 4.XXXX, time XXXXms, mfu -100.00%
iter 1: loss 4.XXXX, time XXXXms, mfu -100.00%
...
step 5: train loss 4.XXXX, val loss 4.XXXX
...
```

**关键指标**：初始 loss 应该接近 `ln(vocab_size)`。对于 Shakespeare 字符级数据集（vocab_size ≈ 65），初始 loss ≈ `ln(65) ≈ 4.17`。

### 通关检查

- [ ] `estimate_loss()` 能正确计算评估 loss
- [ ] 前向传播在混合精度上下文中执行
- [ ] Loss 正确除以 `gradient_accumulation_steps`
- [ ] 反向传播使用了 `scaler.scale()`
- [ ] 梯度裁剪在 `scaler.unscale_()` 之后执行
- [ ] 优化器更新和梯度清零顺序正确
- [ ] 初始 loss 接近 `ln(vocab_size)`

---

## 🏆 最终通关测试

完成所有关卡后，运行以下命令进行端到端测试：

### 测试 1：训练 Shakespeare 字符级模型

```bash
# 准备数据（如果还没有）
cd data/shakespeare_char && python prepare.py && cd ../..

# 训练（约 3 分钟，视硬件而定）
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --max_iters=1000 --eval_interval=250
```

你应该能看到 loss 逐渐下降（从 ~4.17 降到 ~2.0 以下）。

### 测试 2：文本生成

```bash
python sample.py --out_dir=out --device=cpu --compile=False \
    --num_samples=3 --max_new_tokens=200
```

你应该能看到类似 Shakespeare 风格的英文文本生成（虽然质量不高，但可读性已经有了）。

### 测试 3：加载预训练 GPT-2（可选，需要联网和 GPU）

```bash
# 安装额外依赖
pip install transformers

# 使用预训练 GPT-2 生成文本
python sample.py --init_from=gpt2 --num_samples=3 --max_new_tokens=100
```

---

## 总结：你学到了什么

通过完成这 7 个关卡，你已经亲手实现了：

| 组件 | 核心概念 |
|------|----------|
| **MLP** | 前馈网络、GELU 激活函数、升维-降维结构 |
| **Block** | Pre-Norm 架构、残差连接 |
| **CausalSelfAttention** | 注意力机制、因果掩码、多头注意力、Flash Attention |
| **GPT.forward()** | Token/Position Embedding、Weight Tying、交叉熵损失 |
| **GPT.generate()** | 自回归解码、温度采样、Top-K 采样 |
| **get_lr()** | 余弦退火、线性预热 |
| **训练循环** | 梯度累积、混合精度训练、GradScaler、梯度裁剪 |

这些概念涵盖了现代大语言模型（如 GPT-3/4、LLaMA、Mistral 等）的**核心架构**。理解了 nanoGPT，你就掌握了理解这些前沿模型的基础。

**下一步建议**：
1. 尝试修改模型架构（如添加 RoPE 位置编码、SwiGLU 激活函数）
2. 在自己的数据集上训练
3. 阅读 LLaMA、Mistral 等模型的论文和代码，看看它们在 nanoGPT 基础上做了哪些改进
