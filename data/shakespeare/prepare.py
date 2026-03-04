"""
Prepare the Shakespeare dataset for word-level language modeling using GPT-2 BPE tokenizer.
Will save train.bin, val.bin containing the token ids.
"""
import os

import pickle
import requests
import numpy as np
from transformers import AutoTokenizer


# 使用 Hugging Face 生态下的轻量级 GPT-2 BPE 分词器对 Shakespeare 数据集进行编码。
# BPE 将高频字符组合映射为子词 Token，大幅缩短序列长度，提升模型上下文覆盖率。
#
# 实现步骤:
#   1. 数据准备
#      - URL: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#      - 检查本地是否已有 input.txt，避免重复请求。
#
#   2. 加载 BPE 分词器
#      - 弃用 tiktoken，改用 transformers.AutoTokenizer。
#      - 加载轻量级目标: "stefan-it/secret-gpt2-tokenizer" (规避下载完整 GPT-2 模型)。
#
#   3. 全量文本编码
#      - 读取文本内容 (必须指定 encoding='utf-8')。
#      - 使用 tokenizer.encode(content, add_special_tokens=False) 获取原生 Token ID 列表。
#
#   4. 划分训练集和验证集
#      - 根据【编码后的 Token 总长度】进行切分：前 90% (train)，后 10% (val)。
#
#   5. 持久化存储 (二进制文件)
#      - 将列表转为 np.array，指定 dtype=np.uint16 (GPT-2 词表 50257 < 65535，完美适配)。
#      - 用 .tofile() 保存为 train.bin 和 val.bin。
#
#   6. 导出元信息 (Meta Data)
#      - 提取 tokenizer.vocab_size 存入 meta.pkl，供下游 model.py 初始化 Embedding 层使用。
#
# 预期输出 (基于 ~417,556 Token 总量):
#   - train.bin: ~375,800 tokens
#   - val.bin:   ~41,756 tokens
#   - meta.pkl:  包含 vocab_size (50257)
#
# 📐 规范提示:
#   - 文件路径: 统一使用 os.path.join(os.path.dirname(__file__), ...) 确保相对路径鲁棒性。
#   - 编码陷阱: open(..., 'r', encoding='utf-8') 是防范 Windows 乱码切分爆炸的关键。
#   - 纯净数据: 必须显式关闭特殊符 (add_special_tokens=False)，保留最纯粹的文本流。

def main() -> None:
    """
    检查数据集是否存在，否则下载并进行分词处理
    """
    # 路径定义
    file_path = os.path.join(os.path.dirname(__file__), 'input.txt')


    # 2. 加载预训练的 BPE 分词器
    # 使用你之前找的轻量级 GPT-2 分词器
    print("正在加载 GPT-2 BPE 分词器...")
    tokenizer = AutoTokenizer.from_pretrained("stefan-it/secret-gpt2-tokenizer")

    if os.path.exists(file_path):
        print('数据集准备就绪')

    else:
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

        response = requests.get(url=url,stream=True)
        if response.status_code == 200:
            with open(file_path,'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                print('下载完成')

        else:
                raise NotImplementedError('下载失败，检查状态码')

    #生成词袋池
    with open(file_path,'r', encoding='utf-8') as f:

        #读取文件
        content = f.read()

        #统计单词
        words = content.split(' ')
        print('数据集包含单词',len(set(words)))

        # 使用 encode 转换为 ID 列表
        # add_special_tokens=False 保证我们拿到的是纯粹的数据流
        tokens = tokenizer.encode(content, add_special_tokens=False)
        print(f"编码后 Token 总数: {len(tokens):,}")

        # 划分数据集
        train = tokens[:int(len(tokens)*0.9)]
        val =  tokens[int(len(tokens)*0.9):]


        #转换数据类型
        train_en = np.array(train,dtype=np.uint16)
        val_en =np.array(val,dtype=np.uint16)

        #保存文件
        if os.path.exists('train.bin') and os.path.exists('val.bin'):
            print('数据集已生成')
        else:
            train_en.tofile('train.bin')
            val_en.tofile('val.bin')
            print("已成功保存 train.bin 和 val.bin")


    #存储词袋
    with open('meta.pkl','wb') as f:

        meta = {

            "vocab_size" : tokenizer.vocab_size,
            'tokenizer_name': 'gpt2-bpe'
        }

        pickle.dump(meta,f)
        print('词袋保存成功')












if __name__ == "__main__":

    main()








