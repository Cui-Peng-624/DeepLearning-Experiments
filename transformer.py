import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import math
import torch
from torch import nn
from torch.nn.functional import softmax

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, embedding_size)) 
        
        numerator = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) # 定义分子 - 转变为(max_len, 1)
        denominator = torch.pow(10000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size) # 定义分母 - 输出维度为(1, embedding_size/2)
        fraction = numerator / denominator # 输出维度(max_len, embedding_size/2)
        
        self.P[:, :, 0::2] = torch.sin(fraction) # 0::2 表示从索引 0 开始，每隔两个元素选择一个元素
        self.P[:, :, 1::2] = torch.cos(fraction) # 1::2 表示从索引 1 开始，每隔两个元素选择一个元素。

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        X = self.dropout(X)
        return X

# Add & Norm
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        # 输入维度：(batch_size, seq_size, num_hiddens)
        # normalized_shape是最后一个维度的大小，LN就是对每个样本的所有特征进行归一化，在transformer中就是对每个句子，也就是batch进行LN
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        fx_add_x = self.dropout(Y) + X # 残差连接
        outputs = self.ln(fx_add_x) # Layer Normalization层归一化 - 对每个样本的所有特征进行归一化
        return outputs

# 前反馈神经网络
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, num_hiddens, ffn_num_hiddens, **kwargs):
        # 输入的维度：(batch_size, seq_size, num_hiddens)
        # ffn_num_input = ffn_num_outputs = num_hiddens
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 梯度裁剪
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm # 整体进行一个缩放

# 掩码交叉熵损失
def sequence_mask(X, valid_len, value=0): # valid_length 有效长度
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1) # 获取第二个维度的大小
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)\
                                                                        [None, :] < valid_len[:, None]
    # [None, :]：将该一维张量扩展为形状 (1, maxlen)
    # valid_len[:, None]：将有效长度张量 valid_len 从形状 (batch_size,) 扩展为 (batch_size, 1)
    X[~mask] = value
    return X

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，seq_size，#_size*num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # (batch_size，seq_size，num_heads，#_size) 
    X = X.permute(0, 2, 1, 3) # (batch_size，num_heads，seq_size, #_size)
    X = X.reshape(-1, X.shape[2], X.shape[3]) # 最终输出的形状: (batch_size*num_heads, seq_size, #_size)
    return X

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # 输出维度：(batch_size*num_heads, seq_size, value_size)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # (batch_size, num_heads, seq_size, value_size)
    X = X.permute(0, 2, 1, 3) # (batch_size, seq_size, num_heads, value_size)
    X = X.reshape(X.shape[0], X.shape[1], -1) # (batch_size, seq_size, num_heads*value_size)
    return X

# 调用 criterion(Y_hat, Y, Y_valid_len)，Y_hat的size：(batch_size,seq_size,tg_vocab_size)，Y的size：(batch_size,seq_size)，Y_valid_len的size：(1,batch_size)
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label) # 创建一个与指定张量具有相同形状和相同数据类型的新张量，并将所有元素初始化为 1
        weights = sequence_mask(weights, valid_len) # 根据valid_len将weights中的部分设置为0
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label) 
        '''
        调用父类 nn.CrossEntropyLoss 的 forward 方法计算未加权的损失。
        由于 nn.CrossEntropyLoss 期望输入形状为 (batch_size, num_classes, num_steps)，所以我们通过 pred.permute(0, 2, 1) 将 pred 的形状从 (batch_size, num_steps, vocab_size) 转换为 (batch_size, vocab_size, num_steps)
        '''
        # print(unweighted_loss.shape)：(batch_size,num_steps)
        # 将未加权的损失 unweighted_loss 与权重 weights 相乘，对每个时间步进行加权。然后对 dim=1 维度（时间步维度）进行平均，得到每个序列的加权损失。
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs): 
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.key_size = num_hiddens / num_heads
        
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, X_queries, X_keys, X_values, valid_lens=None):
        queries = self.W_q(X_queries) 
        keys = self.W_k(X_keys) 
        values = self.W_v(X_values) 

        queries = transpose_qkv(queries, self.num_heads)
        keys = transpose_qkv(keys, self.num_heads)
        values = transpose_qkv(values, self.num_heads)

        keys = keys.transpose(1, 2)
        scores = torch.bmm(queries, keys) 
        scores = scores / torch.tensor([math.sqrt(self.key_size)]).to(scores.device)
        
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
            attention_weights = self.masked_softmax(scores, valid_lens)
        else:
            attention_weights = self.masked_softmax(scores, valid_lens)

        attention_weights = self.dropout(attention_weights)
        attention_outputs = torch.bmm(attention_weights, values) 
        attention_outputs_concat = transpose_output(attention_outputs, self.num_heads) # Z_concat的维度：(batch_size, seq_size, num_heads*value_size)
        outputs = self.W_o(attention_outputs_concat) # outputs的维度：(batch_size, seq_size, num_hiddens)
        return outputs

    def masked_softmax(self, scores, valid_lens):
        shape = scores.shape # shape[-1] 为 seq_size
        if valid_lens is None:
            return nn.functional.softmax(scores, dim=-1) # dim=-1指定了延最后一个维度进行softmax，也就是行
        else: 
            if valid_lens.dim() == 1:
                mask = torch.arange(shape[-1], device=scores.device)[None, :] < valid_lens[:, None] 
            else:
                mask = torch.arange(shape[-1], device=scores.device)[None, :] < valid_lens
            scores[~mask] = float('-inf') # scores的维度：(batch_size*num_heads, seq_size, seq_size)
            return nn.functional.softmax(scores, dim=-1)

# AddNorm层
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        # 输入维度：(batch_size, seq_size, num_hiddens)
        # normalized_shape是最后一个维度的大小，LN就是对每个样本的所有特征进行归一化，在transformer中就是对每个句子，也就是batch进行LN
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        fx_add_x = self.dropout(Y) + X # 残差连接
        outputs = self.ln(fx_add_x) # Layer Normalization层归一化 - 对每个样本的所有特征进行归一化
        return outputs

# 一个transformer encoder block
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, num_hiddens, num_heads, normalized_shape, ffn_num_hiddens, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens)
        self.addnorm2 = AddNorm(normalized_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

