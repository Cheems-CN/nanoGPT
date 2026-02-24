"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# TODO: ====== 模型加载 ======
# 根据 init_from 的值加载模型:
#
# (A) init_from == 'resume': 从训练 checkpoint 加载
#     1. 从 out_dir/ckpt.pt 加载 checkpoint（torch.load）
#     2. 用 checkpoint['model_args'] 创建 GPTConfig 和 GPT 模型
#     3. 处理 state_dict 中可能存在的 '_orig_mod.' 前缀（torch.compile 产生的）
#     4. 加载权重: model.load_state_dict(state_dict)
#
# (B) init_from.startswith('gpt2'): 从预训练 GPT-2 加载
#     model = GPT.from_pretrained(init_from, dict(dropout=0.0))
#
# 加载后: model.eval() → model.to(device) → 可选 torch.compile
#
# 📐 规范提示:
#   - 用 if/elif 分支处理两种加载模式，不要用 try/except 控制流程
#   - checkpoint 前缀清理用字典推导或循环，命名清晰:
#     unwanted_prefix = '_orig_mod.'
#     for k, v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#   - 调用链分行写，增加可读性:
#     model.eval()
#     model.to(device)
raise NotImplementedError("TODO: 实现模型加载")

# TODO: ====== 分词器加载 ======
# 两种分词方案:
#
# (A) 自定义分词器（字符级等）:
#     如果 checkpoint 中包含 dataset 信息，检查对应 data/ 目录下是否有 meta.pkl
#     如果有，从 meta.pkl 加载 stoi/itos 映射，构建 encode/decode 函数
#
# (B) GPT-2 BPE 分词器（默认）:
#     使用 tiktoken.get_encoding("gpt2") 加载 GPT-2 分词器
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)
#
# 📐 规范提示:
#   - 用 bool 标志变量 load_meta 控制加载路径
#   - lambda 表达式适合简单的 encode/decode 一行函数
#   - 复杂逻辑应用 def 定义具名函数
raise NotImplementedError("TODO: 实现分词器加载")

# TODO: ====== 文本生成 ======
# 1. 编码起始文本 start 为 token IDs（支持 "FILE:path" 格式从文件读取）
# 2. 创建输入张量: (1, len(start_ids))，dtype=torch.long
# 3. 在 torch.no_grad() 和混合精度上下文中:
#    循环 num_samples 次:
#      调用 model.generate(x, max_new_tokens, temperature, top_k)
#      decode 生成的 token IDs 并打印
#
# 📐 规范提示:
#   - 文件读取用 with open(..., encoding='utf-8') 确保编码正确
#   - 张量创建的惯用写法:
#     x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
#   - 嵌套上下文管理器:
#     with torch.no_grad():
#         with ctx:
#             for k in range(num_samples):
#                 ...
#   - 生成结果用 y[0].tolist() 转为 Python 列表再 decode
#   - 样本之间用分隔线: print('---------------')
raise NotImplementedError("TODO: 实现文本生成")
