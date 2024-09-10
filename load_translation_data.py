import os
import torch
from torch.utils import data
from text_preprocessing import Vocab

# d2l-zh/pytorch/! My Exercises/data/fra-eng/fra.txt
def read_data_nmt():
    """载入“英语－法语”数据集"""
    
    # data_dir = "data/fra-eng"
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(current_script_path)
    # 拼接data_dir的绝对路径
    data_dir = os.path.join(current_dir, '..', 'data', 'fra-eng')
    
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()
raw_text = read_data_nmt()

raw_text = raw_text[:1000]

# 用空格代替不间断空格（non-breaking space）， 使用小写字母替换大写字母，并在单词和标点符号之间插入空格
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格/
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
text = preprocess_nmt(raw_text)

# 词元化
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target
source, target = tokenize_nmt(text)
"""
source[:6], target[:6] 输出：
([['go', '.'],
  ['hi', '.'],
  ['run', '!'],
  ['run', '!'],
  ['who', '?'],
  ['wow', '!']],
 [['va', '!'],
  ['salut', '!'],
  ['cours', '!'],
  ['courez', '!'],
  ['qui', '?'],
  ['ça', 'alors', '!']])
"""

src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(list(src_vocab.token_to_idx.items())[:10]) 输出：[('<unk>', 0), ('<pad>', 1), ('<bos>', 2), ('<eos>', 3), ('.', 4), ('i', 5), ('you', 6), ('to', 7), ('the', 8), ('?', 9)]

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True): 
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
"""
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break

list(src_vocab.token_to_idx.items())[:10], list(tgt_vocab.token_to_idx.items())[:10]

输出：

X: tensor([[ 0,  0,  4,  3,  1,  1,  1,  1],
        [ 6, 69, 54,  4,  3,  1,  1,  1]], dtype=torch.int32)
X的有效长度: tensor([4, 5])
Y: tensor([[83,  0,  0,  4,  3,  1,  1,  1],
        [ 6, 57,  0,  4,  3,  1,  1,  1]], dtype=torch.int32)
Y的有效长度: tensor([5, 5])
([('<unk>', 0),
  ('<pad>', 1),
  ('<bos>', 2),
  ('<eos>', 3),
  ('.', 4),
  ('!', 5),
  ('i', 6),
  ("i'm", 7),
  ('it', 8),
  ('go', 9)],
 [('<unk>', 0),
  ('<pad>', 1),
  ('<bos>', 2),
  ('<eos>', 3),
  ('.', 4),
  ('!', 5),
  ('je', 6),
  ('suis', 7),
  ('tom', 8),
  ('?', 9)])
"""






