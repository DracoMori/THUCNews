'''
date: 2020/2/5
author: 流氓兔233333
content: TextCNN 处理新闻文本分类任务
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

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, tar_size):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        num_filters = 64
        filter_sizes = [2,3,4]
        self.convs = nn.ModuleList(
            # [ batch_size, channels, height_1, width_1]
            # num_filters 定义了feature map 的个数
            [nn.Conv2d(1, num_filters, (k, embed_size)) for k in filter_sizes])
                        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), tar_size)

    # out_emb = [batch, seq_len, embed_size]
    def conv_and_pool(self, out_emb, conv):
        # x = [batch_size, channel, seq_len-filter_size[0] , 1] -> [batch_size, channel, seq_len-filter_size[0]]
        x = F.relu(conv(out_emb)).squeeze(3)
        # x = [batch_size, channel, 1 , 1] -> x = [batch_size, channel]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x): 
        # 动态获取 batch_size
        batch_size = x.shape[0]
        embedding_x = self.embed(x)
        # embedding_x = [batch_size, channel=1, seq_len, embed_size]
        embedding_x = embedding_x.unsqueeze(1) # add channel(=1)
        # out = [batch_size, num_filters*len(filter_sizes)]
        out = torch.cat([self.conv_and_pool(embedding_x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# embed_size = 50
tar_size = len(tar2idx_dict)
vocab_size = len(word2idx_dict)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = TextCNN(vocab_size=vocab_size, embed_size=embed_size, tar_size=tar_size)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train
loss_his = []
# step, (batch_X, batch_y) = next(enumerate(loader))
for epoch in tqdm(range(40)):
    model = model.train()
    for step, (batch_X, batch_y) in enumerate(loader):
        batch_X, batch_y = batch_X, batch_y
        pred = model(batch_X)
        loss = loss_fun(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_his.append(loss.item())
    print('epoch: ', epoch, '   train loss:', loss.item())

    if epoch % 1 == 0:
        # print('epoch: ', epoch, '   train loss:', loss.item())

        model = model.eval()
        x_val_tensor = x_val_tensor
        # x_train_device = x_train_tensor.to(device)
        # output_train = model(x_train_device)
        output = model(x_val_tensor)

        # _, prediction_train = torch.max(F.softmax(output_train, dim=1), 1)
        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        # pred_train = prediction_train.data.numpy().squeeze()
        pred_val = prediction.data.numpy().squeeze()
        # acc_train = ms.accuracy_score(y_train_tensor.numpy(), pred_train)
        acc_test = ms.accuracy_score(y_val_tensor.numpy(), pred_val)
        print('test_acc:', acc_test)
        # print('train_acc:', acc_train, 'test_acc:', acc_test)


# model save
torch.save(model.state_dict(), save_path+'TextCNN_params.pkl')

# model load
model = TextCNN(vocab_size=vocab_size, embed_size=embed_size, tar_size=tar_size)
model.load_state_dict(torch.load(save_path+'LSTM_params.pkl'))
model.eval()

