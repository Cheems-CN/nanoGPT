"""
Prepare the Shakespeare dataset for word-level language modeling using GPT-2 BPE tokenizer.
Will save train.bin, val.bin containing the token ids.
"""
import os
import requests
import tiktoken
import numpy as np

# TODO: ====== BPE 分词器与数据预处理 ======
#
# 使用 GPT-2 的 BPE (Byte Pair Encoding) 分词器对 Shakespeare 数据集进行编码。
# 与字符级分词不同，BPE 将常见的字符组合合并为子词 token。
#
# 实现步骤:
#   1. 下载数据集
#      - URL: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#      - 保存到当前目录下的 input.txt
#
#   2. 加载 GPT-2 BPE 分词器
#      - 使用 tiktoken.get_encoding("gpt2")
#
#   3. 划分训练集和验证集
#      - 前 90% 作为训练集，后 10% 作为验证集
#
#   4. 编码文本
#      - 使用 enc.encode_ordinary() 编码（忽略特殊 token）
#
#   5. 保存为二进制文件
#      - 转为 np.uint16（GPT-2 最大 token ID = 50256 < 2^16）
#      - 用 .tofile() 保存为 train.bin 和 val.bin
#
# 预期输出:
#   - train.bin: ~301,966 tokens
#   - val.bin:   ~36,059 tokens
#
# 对比字符级: BPE 的 token 数量更少（约 1/3），因为每个 token 可以表示多个字符
raise NotImplementedError("TODO: 实现 BPE 数据预处理")
