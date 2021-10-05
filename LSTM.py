'''
date: 2020/2/5
author: 流氓兔233333
content: LSTM 处理新闻文本分类任务
'''
import numpy as np
import pandas as pd
from tqdm import tqdm  
import time
import random
import os, warnings, pickle
warnings.filterwarnings('ignore')

import torch
import torch.utils.data as Data
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sklearn.metrics as ms


data_path = './data_raw/'
save_path = './temp_results/'

# load word2idx tar2idx
word2idx_dict = pickle.load(open(save_path+'word2idx_dict.pkl', 'rb'))
tar2idx_dict = pickle.load(open(save_path+'target2idx_dict.pkl', 'rb'))


# load tensor data
train_tensor_tuple = pickle.load(open(save_path+'train_tensor_tuple.pkl', 'rb'))
val_tensor_tuple = pickle.load(open(save_path+'val_tensor_tuple.pkl', 'rb'))
x_train_tensor, y_train_tensor = train_tensor_tuple[0], train_tensor_tuple[1]
x_val_tensor, y_val_tensor = val_tensor_tuple[0], val_tensor_tuple[1]

del train_tensor_tuple, val_tensor_tuple

batch_size = 128
dataset = Data.TensorDataset(x_train_tensor, y_train_tensor)
loader = Data.DataLoader(dataset, batch_size, True)

# input data [batch_size, seq_len]

class Net_LSTM(nn.Module):
    def __init__(self, embed_dim, hid_dim, vocab_size, tar_size):
        super(Net_LSTM, self).__init__()
        self.hid_dim = hid_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, tar_size)

    def forward(self, sequence):
        # nn.Embedding 要求输入的数据为 [batch_size, seq_len]

        batch_size = sequence.shape[0]
        padid = word2idx_dict['PAD']
        fun_len = lambda seq: len(np.where(seq.numpy() != padid)[0])
        sentence_lens = np.array([fun_len(seq) for seq in sequence])
        sentence_lens = torch.LongTensor(sentence_lens)

        # 排序
        sorted_seq_lengths, indices = torch.sort(sentence_lens, descending=True)
        # 恢复排序索引
        _, desorted_indices = torch.sort(indices, descending=False)
        del sentence_lens
        # 对原始输入排序
        sequence = sequence[indices]

        # 输出为 [batch_szie, seq_len, embed_dim]
        embeds = self.word_embedding(sequence)
        embed_input_x_packed = pack_padded_sequence(embeds, sorted_seq_lengths, batch_first=True)

        # nn.LSTM 要求输入数据为 [batch_size, seq_len, embed_dim]
        # hidden_0 未给出时会被默认设置为0
        # lstm_out  [batch_size, seq_len, hid_dim*num_directions] 
        # h_n  [n_layers*num_directions, batch_size, embed_dim]
        # h_c 用于生成 h_n, 不用于输出
        # 文本分类任务中， 我只需要 lstm_hidden 的最后一个时刻的值就可以了
        lstm_out, (h_n, h_c) = self.lstm(embed_input_x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
    
        # # 将 h_n 与 h_c 拼接起来输入 fc层
        # lstm_hidden = torch.cat([h_n.squeeze(0), h_c.squeeze(0)], 1)
        
        lstm_out_mean = torch.mean(lstm_out, dim=1)
        # 逆排序
        lstm_out_mean = lstm_out_mean[desorted_indices]
        output = self.fc(lstm_out_mean)

        return output



tar_size = len(tar2idx_dict)
vocab_size = len(word2idx_dict)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = Net_LSTM(embed_dim=embed_size, hid_dim=16, vocab_size=vocab_size, tar_size=tar_size)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


step, (batch_x, batch_y) = next(enumerate(loader))
batch_x = x_val_tensor
output = model(batch_x)


loss_his = []
for epoch in tqdm(range(10)):
    model = model.train()
    for step, (batch_x, batch_y) in enumerate(loader):
        output = model(batch_x)

        loss = loss_fun(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_his.append(loss.item())

    if epoch % 1 == 0:
        model = model.eval()
        output = model(x_val_tensor)

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_test = ms.accuracy_score(y_val_tensor.numpy(), pred_val)
        print('  loss:', loss.item())
        print('  test_acc:', acc_test)



# loss 曲线图
import matplotlib.pyplot as plt
plt.plot(loss_his)
plt.ylabel('loss', fontsize=20)
plt.xlabel('eopch', fontsize=20)
plt.tick_params(labelsize=20)
plt.title('EOPCH_LOSS_LSTM', fontsize=20)
plt.grid()
plt.show()


# model save
torch.save(model.state_dict(), save_path+'LSTM_params.pkl')

# model load
model_load = Net_LSTM(embed_dim=embed_size, hid_dim=16, vocab_size=vocab_size, tar_size=tar_size)
model_load.load_state_dict(torch.load(save_path+'LSTM_params.pkl'))
model_load.eval()

output = model_load(x_val_tensor)
_, prediction = torch.max(F.softmax(output, dim=1), 1)
pred_val = prediction.data.numpy().squeeze()
acc_test = ms.accuracy_score(y_val_tensor.numpy(), pred_val)
print('  test_acc:', acc_test)

