# 工程化文件组织指南

## 前言

本指南帮助你在实现 nanoGPT 的过程中，同时练习规范的软件工程实践。你不需要一开始就创建所有文件——**边实现代码边创建对应的文件结构**。

---

## 推荐的项目结构

当你逐步完成所有代码后，项目应该组织成如下结构：

```
nanoGPT/
│
├── src/                            # 🔴 源代码主目录
│   ├── __init__.py
│   ├── model/                      # 模型架构模块
│   │   ├── __init__.py                 # 导出 GPTConfig, GPT
│   │   ├── config.py                   # GPTConfig dataclass
│   │   ├── layers.py                   # LayerNorm, MLP
│   │   ├── attention.py                # CausalSelfAttention
│   │   ├── block.py                    # Block (Transformer Block)
│   │   └── gpt.py                      # GPT 主模型类
│   │
│   ├── data/                       # 数据处理模块
│   │   ├── __init__.py
│   │   ├── tokenizer.py                # 分词器抽象（字符级 + BPE）
│   │   ├── dataset.py                  # 数据集类和数据加载器
│   │   └── prepare.py                  # 数据集下载和预处理入口
│   │
│   ├── training/                   # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py                  # 训练器类（封装训练循环）
│   │   ├── scheduler.py                # 学习率调度器
│   │   └── utils.py                    # 训练工具（checkpoint、日志等）
│   │
│   └── inference/                  # 推理模块
│       ├── __init__.py
│       ├── generator.py                # 文本生成器
│       └── sampler.py                  # 采样策略（温度、Top-K、Top-P）
│
├── configs/                        # 🟢 配置文件目录
│   ├── base.yaml                       # 基础配置
│   ├── train_shakespeare_char.yaml     # Shakespeare 字符级训练
│   ├── train_gpt2.yaml                 # GPT-2 全量训练
│   ├── finetune_shakespeare.yaml       # Shakespeare 微调
│   └── eval_gpt2.yaml                  # GPT-2 评估
│
├── data/                           # 🟢 数据目录（不纳入版本控制）
│   ├── raw/                            # 原始数据
│   │   └── shakespeare/input.txt
│   └── processed/                      # 预处理后的数据
│       ├── shakespeare_char/
│       │   ├── train.bin
│       │   ├── val.bin
│       │   └── meta.pkl
│       └── shakespeare_bpe/
│           ├── train.bin
│           └── val.bin
│
├── tests/                          # 🧪 测试目录
│   ├── __init__.py
│   ├── test_model/
│   │   ├── test_layers.py              # LayerNorm、MLP 单元测试
│   │   ├── test_attention.py           # 注意力机制测试（含因果性测试）
│   │   ├── test_block.py               # Block 测试
│   │   └── test_gpt.py                 # GPT 完整模型测试
│   ├── test_data/
│   │   ├── test_tokenizer.py           # 分词器测试
│   │   └── test_dataset.py             # 数据加载测试
│   ├── test_training/
│   │   ├── test_scheduler.py           # 学习率调度器测试
│   │   └── test_trainer.py             # 训练流程集成测试
│   └── conftest.py                     # pytest 共享 fixtures
│
├── scripts/                        # 🔧 脚本目录
│   ├── train.py                        # 训练入口脚本
│   ├── sample.py                       # 推理入口脚本
│   ├── prepare_data.py                 # 数据准备入口脚本
│   └── benchmark.py                    # 性能基准测试
│
├── notebooks/                      # 📓 实验笔记本
│   ├── scaling_laws.ipynb
│   └── transformer_sizing.ipynb
│
├── docs/                           # 📖 文档目录
│   ├── architecture.md                 # 架构设计文档
│   ├── build_guide.md                  # 实现指南
│   └── engineering_guide.md            # 工程化指南（本文件）
│
├── outputs/                        # 📦 输出目录（不纳入版本控制）
│   └── checkpoints/                    # 训练 checkpoint
│
├── .gitignore                      # Git 忽略规则
├── pyproject.toml                  # 项目元数据和依赖
├── requirements.txt                # Python 依赖
├── Makefile                        # 常用命令快捷方式
└── README.md                       # 项目说明
```

---

## 核心组织原则

### 1. 模块化拆分

**原则**: 每个文件只负责一个职责（Single Responsibility Principle）

| 当前 (nanoGPT 原版) | 推荐拆分 | 理由 |
|---------------------|----------|------|
| `model.py` (所有类) | `config.py` + `layers.py` + `attention.py` + `block.py` + `gpt.py` | 每个类独立一个文件，便于单独测试和修改 |
| `train.py` (所有逻辑) | `trainer.py` + `scheduler.py` + `utils.py` | 训练循环、学习率调度、工具函数分离 |
| `sample.py` | `generator.py` + `sampler.py` | 生成逻辑和采样策略分离 |
| `data/*/prepare.py` | `tokenizer.py` + `dataset.py` + `prepare.py` | 分词器抽象、数据集类、预处理脚本分离 |

### 2. 分层架构

```
┌─────────────────────────────────────┐
│           scripts/ (入口层)          │  用户直接运行的脚本
├─────────────────────────────────────┤
│         src/training/ (训练层)       │  训练循环、调度器
│         src/inference/ (推理层)      │  生成、采样
├─────────────────────────────────────┤
│          src/model/ (模型层)         │  模型架构定义
├─────────────────────────────────────┤
│          src/data/ (数据层)          │  数据加载和处理
└─────────────────────────────────────┘
```

**依赖方向**: 上层依赖下层，下层不依赖上层。

### 3. 配置管理

**推荐**: 使用 YAML 或 TOML 替代 Python exec() 配置

```yaml
# configs/train_shakespeare_char.yaml
model:
  n_layer: 6
  n_head: 6
  n_embd: 384
  block_size: 256
  dropout: 0.2

training:
  batch_size: 64
  max_iters: 5000
  learning_rate: 1.0e-3
  weight_decay: 0.1

data:
  dataset: shakespeare_char
```

**工具选择**: `hydra`, `omegaconf`, 或标准库 `tomllib`

---

## 测试策略

### 测试金字塔

```
        /  端到端测试  \        少量：完整训练+生成
       / 集成测试       \       适量：模型前向+反向传播
      / 单元测试          \     大量：每个类/函数独立测试
     ──────────────────────
```

### 单元测试示例

```python
# tests/test_model/test_layers.py
import pytest
import torch
from src.model.layers import LayerNorm, MLP
from src.model.config import GPTConfig

class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(64, bias=True)
        x = torch.randn(2, 10, 64)
        assert ln(x).shape == x.shape

    def test_no_bias(self):
        ln = LayerNorm(64, bias=False)
        assert ln.bias is None

    def test_normalization(self):
        ln = LayerNorm(64, bias=False)
        x = torch.randn(2, 10, 64)
        y = ln(x)
        # 归一化后，每个向量的均值应接近 0，标准差应接近 1
        assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 10), atol=1e-5)

class TestMLP:
    @pytest.fixture
    def config(self):
        return GPTConfig(n_embd=64, bias=True, dropout=0.0)

    def test_output_shape(self, config):
        mlp = MLP(config)
        x = torch.randn(2, 10, 64)
        assert mlp(x).shape == (2, 10, 64)

    def test_intermediate_dimension(self, config):
        mlp = MLP(config)
        assert mlp.c_fc.out_features == 4 * 64
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_model/test_layers.py -v

# 运行特定测试类
pytest tests/test_model/test_attention.py::TestCausalSelfAttention -v

# 带覆盖率
pytest tests/ --cov=src --cov-report=html
```

---

## 代码规范

### 类型标注

```python
import torch
from torch import Tensor

def forward(self, x: Tensor) -> Tensor:
    """前向传播。

    Args:
        x: 输入张量，形状 (batch_size, seq_len, n_embd)

    Returns:
        输出张量，形状与输入相同
    """
    ...
```

### 文档字符串

每个类和公开方法都应有文档字符串，包含:
- 功能描述
- 参数说明（含形状）
- 返回值说明
- 使用示例（对于复杂方法）

### 推荐工具

| 工具 | 用途 | 命令 |
|------|------|------|
| `black` | 代码格式化 | `black src/ tests/` |
| `isort` | import 排序 | `isort src/ tests/` |
| `ruff` | 快速 linting | `ruff check src/ tests/` |
| `mypy` | 类型检查 | `mypy src/` |
| `pytest` | 测试 | `pytest tests/ -v` |
| `pytest-cov` | 覆盖率 | `pytest --cov=src` |

---

## Makefile 示例

```makefile
.PHONY: format lint test train sample clean

format:
	black src/ tests/
	isort src/ tests/

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html

prepare-data:
	python scripts/prepare_data.py --dataset=shakespeare_char

train:
	python scripts/train.py --config=configs/train_shakespeare_char.yaml

sample:
	python scripts/sample.py --checkpoint=outputs/checkpoints/latest.pt

clean:
	rm -rf outputs/ data/processed/ __pycache__ .pytest_cache htmlcov
```

---

## 依赖管理

### pyproject.toml 示例

```toml
[project]
name = "nanogpt"
version = "0.1.0"
description = "Minimal GPT implementation for learning"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy",
    "tiktoken",
    "requests",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "isort",
    "ruff",
    "mypy",
]
finetune = [
    "transformers",
]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## .gitignore 规则

```gitignore
# 数据文件（太大，不纳入版本控制）
data/raw/
data/processed/
*.bin
*.pkl

# 训练输出
outputs/
out/

# Python
__pycache__/
*.pyc
.pytest_cache/
htmlcov/
*.egg-info/

# IDE
.vscode/
.idea/

# 系统文件
.DS_Store
```

---

## 实施建议

### 渐进式重构路线

你不需要一开始就按照上面的结构来组织。建议的渐进路线：

1. **第一步**: 先在现有的 `model.py`、`train.py`、`sample.py` 中完成所有 TODO
2. **第二步**: 添加 `tests/` 目录，为每个组件写单元测试
3. **第三步**: 将 `model.py` 拆分为 `src/model/` 模块
4. **第四步**: 将 `train.py` 拆分为 `src/training/` 模块
5. **第五步**: 添加配置管理（YAML）、Makefile、pyproject.toml
6. **第六步**: 添加类型标注和文档字符串
7. **第七步**: 设置 CI/CD（GitHub Actions）

### 每一步的检查点

在每一步重构后，确保：
- [ ] 所有测试通过
- [ ] 训练脚本能正常运行
- [ ] 生成脚本能正常运行
- [ ] 代码通过 lint 检查

---

## 总结

**核心理念**: 好的工程实践不是目的，而是手段——它帮助你写出更可维护、更可靠、更容易扩展的代码。在学习阶段，先保证功能正确，再逐步引入工程化实践。
