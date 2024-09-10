import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import re
from collections import Counter
import numpy as np
from visualization import TrainingVisualizer

def create_dataset(text_as_int, seq_size, batch_size):
    sequences = []
    targets = []
    for i in range(0, len(text_as_int) - seq_size):
        sequences.append(text_as_int[i:i + seq_size])
        targets.append(text_as_int[i + 1:i + seq_size + 1])
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(sequences, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def initialize_LSTM_Wb(embedding_size, hidden_size, vocab_size, device):
    """
    hidden_size：神经元个数
    # X's size: (batch_size, seq_size, vocab_size) -> (seq_size, batch_size, vocab_size) -> (seq_size, batch_size, embedding_size)
    但注意：这里我们需要的X的维度：(batch_size, embedding_size)
    # H = torch.normal(0, 1, (batch_size, hidden_size))
    """
    torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, device=device))
    # 输入门 I
    W_xi = torch.nn.init.xavier_uniform_(torch.empty(embedding_size, hidden_size, device=device))
    W_hi = torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, device=device))
    b_i = torch.zeros((1, hidden_size), device = device) # 注意这里的偏置项是加在神经元上，不是batch_size*hidden_size

    # 遗忘门 F
    W_xf =  torch.nn.init.xavier_uniform_(torch.empty(embedding_size, hidden_size, device=device))
    W_hf = torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, device=device))
    b_f = torch.zeros((1, hidden_size), device = device) 

    # 输出门 O
    W_xo =  torch.nn.init.xavier_uniform_(torch.empty(embedding_size, hidden_size, device=device))
    W_ho = torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, device=device))
    b_o = torch.zeros((1, hidden_size), device = device) 
    
    # 候选记忆元 tilde_C
    W_xc = torch.nn.init.xavier_uniform_(torch.empty(embedding_size, hidden_size, device=device))
    W_hc = torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, device=device))
    b_c = torch.zeros((1, hidden_size), device = device)

    # 输出层参数
    W_hq = torch.normal(0, 1, (hidden_size, vocab_size), device = device) * 0.01
    b_q = torch.zeros((1, vocab_size), device = device)

    # 附加梯度
    I_params = [W_xi, W_hi, b_i]
    F_params = [W_xf, W_hf, b_f]
    O_params = [W_xo, W_ho, b_o]
    C_params = [W_xc, W_hc, b_c]
    Output_layer_params = [W_hq, b_q]
    params = [I_params, F_params, O_params, C_params, Output_layer_params]
    for i_param in params:
        for param in i_param:
            param.requires_grad_(True)
    
    return params

def initialize_LSTM_HC(batch_size, hidden_size, device):
    H = torch.zeros((batch_size, hidden_size), device = device)
    C = torch.zeros((batch_size, hidden_size), device = device)
    return H, C

def LSTM_calculate(inputs, params, H, C, device): 
    # inputs的形状：(seq_size，batch_size, embedding_size)
    # 需要传入上一个时间步的隐状态H和记忆元C
    outputs = []
    
    I_params, F_params, O_params, C_params, Output_layer_params = params
    W_xi, W_hi, b_i = I_params
    W_xf, W_hf, b_f = F_params
    W_xo, W_ho, b_o = O_params
    W_xc, W_hc, b_c = C_params
    W_hq, b_q = Output_layer_params
    
    for X in inputs: # X的形状：(batch_size, embedding_size)
        I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
        F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
        O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
        tilde_C = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = torch.mul(F, C) + torch.mul(I, tilde_C)
        H = torch.mul(O, torch.tanh(C)) 
        O = torch.mm(H, W_hq) + b_q
        outputs.append(O) # outputs是每一个时间步的输出 - [tensor(batch_size, vocab_size), tensor(batch_size, vocab_size), ......]
        
    outputs = torch.stack(outputs)
    return outputs, H, C # outputs 的 size：(seq_size, batch_size, vocab_size)；H 的 size：(batch_size, hidden_size)

class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, initialize_LSTM_Wb, initialize_LSTM_HC, LSTM_calculate, device):
        super(LSTMModel, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.params = initialize_LSTM_Wb(embedding_size, hidden_size, vocab_size, device)
        self.initialize_LSTM_HC = initialize_LSTM_HC
        self.LSTM_calculate = LSTM_calculate

    def forward(self, X, H, C): 
        # 输入的X的形状为：(batch_size, seq_size)；H的形状为：(batch_size, hidden_size)
        X = self.embedding(X) # 此时X的形状变为：(batch_size, seq_size, embedding_size)
        X = X.permute(1, 0, 2).to(self.device) # 此时X的形状变为：(seq_size, batch_size, embedding_size)
        Y, H, C = self.LSTM_calculate(X, self.params, H, C, self.device)
        return Y, H, C # 返回Y的size：(seq_size, batch_size, vocab_size)

    def begin_HC(self, batch_size, hidden_size, device):
        H, C = self.initialize_LSTM_HC(batch_size, hidden_size, device)
        return H, C

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

def LSTM_train_epoch(train_iter, model, criterion, optimizer, device):
    """训练网络一个迭代周期"""
    H = None
    C = None
    
    for X, Y in train_iter: # X，Y的形状：(batch_size, seq_size)
        
        if H is None and C is  None: # 如果处于第一个迭代周期
            H, C = model.begin_HC(X.shape[0], hidden_size, device)
        else: 
            H = H.detach()
            C = C.detach()

        Y = Y.T # Y的size从(batch_size, seq_size) -> (seq_size, batch_size)
        Y_hat, H, C = model(X, H, C) # Y_hat的size：(seq_size, batch_size, vocab_size)
        # 为了计算交叉熵损失，我们需要调整Y_hat和Y的size，这也是课本代码中使用reshape的原因
        Y_hat = Y_hat.reshape(-1, Y_hat.shape[2]) # size变成：(seq_size*batch_size, vocab_size)
        Y = Y.reshape(-1) # 把Y变为一维的张量
        Y_hat, Y = Y_hat.to(device), Y.to(device)

        optimizer.zero_grad()
        # print(Y_hat, Y)
        loss = criterion(Y_hat, Y.long()).mean()
        loss.backward()
        grad_clipping(model, 1)
        optimizer.step()

        perplexity = math.exp(loss)
        
    return perplexity

def LSTM_train(model, train_iter, lr, num_epochs, device):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)  # 使用Adam优化器
    visualizer = TrainingVisualizer(xlabel='Epoch', ylabel='Perplexity', title='Train LSTM_scratch in Time_Machine', legend=['Perplexity'])  

    predict = lambda prefix: text_prediction(prefix, 50, model, device) # 匿名函数，输入prefix预测50个token
    
    # 训练和预测
    for epoch in range(num_epochs):
        perplexity = LSTM_train_epoch(train_iter, model, criterion, optimizer, device)
        visualizer.add(epoch, [perplexity])


