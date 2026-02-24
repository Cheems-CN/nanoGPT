"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# TODO: ====== 字符级分词器与数据预处理 ======
#
# 这是最简单的分词方案：将每个字符映射为一个整数 ID。
# 完成后将生成 train.bin, val.bin 和 meta.pkl 三个文件。
#
# 实现步骤:
#   1. 下载数据集
#      - URL: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#      - 保存到当前目录下的 input.txt
#
#   2. 构建字符级词表
#      - 读取全部文本，找出所有不重复的字符并排序
#      - 创建 stoi (字符→整数) 和 itos (整数→字符) 映射
#      - vocab_size = 唯一字符数量
#
#   3. 实现 encode 和 decode 函数
#      - encode(s): 字符串 → 整数列表
#      - decode(l): 整数列表 → 字符串
#
#   4. 划分训练集和验证集
#      - 前 90% 作为训练集，后 10% 作为验证集
#
#   5. 编码并保存为二进制文件
#      - 将文本编码为整数列表
#      - 转为 np.uint16 类型的 NumPy 数组
#      - 用 .tofile() 保存为 train.bin 和 val.bin
#
#   6. 保存元信息
#      - 将 vocab_size, itos, stoi 保存到 meta.pkl (pickle 格式)
#
# 预期输出:
#   - 数据集约 1,115,394 个字符
#   - vocab_size ≈ 65 个唯一字符
#   - train: ~1,003,854 tokens, val: ~111,540 tokens
#
# 📐 规范提示:
#   - 文件路径用 os.path.join(os.path.dirname(__file__), 'input.txt')
#     确保无论从哪个目录运行脚本，路径都正确
#   - 下载前检查文件是否已存在，避免重复下载:
#     if not os.path.exists(input_file_path):
#         ...
#   - 字典推导构建映射: stoi = {ch: i for i, ch in enumerate(chars)}
#   - encode/decode 用简洁的列表推导:
#     def encode(s: str) -> list[int]:
#         return [stoi[c] for c in s]
#     def decode(l: list[int]) -> str:
#         return ''.join([itos[i] for i in l])
#   - 每步打印进度信息: print(f"train has {len(train_ids):,} tokens")
#   - pickle 保存用 'wb' 模式: with open(path, 'wb') as f: pickle.dump(meta, f)
raise NotImplementedError("TODO: 实现字符级数据预处理")
