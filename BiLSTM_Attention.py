'''
date: 2020/2/7
author: 流氓兔233333
content: 注意力机制融合双向LSTM
model架构参考  NLP_融合主题模型和注意力机制的政策文本分类模型_胡吉明
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
dataset_trn = Data.TensorDataset(x_train_tensor, y_train_tensor)
loader_trn = Data.DataLoader(dataset_trn, batch_size, True)
dataset_val = Data.TensorDataset(x_val_tensor, y_val_tensor)
loader_val = Data.DataLoader(dataset_val, len(dataset_val), True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class BiLSTM_Attention(nn.Module):
    def __init__(self, embed_dim, hid_dim, vocab_size, tar_size):
        super(BiLSTM_Attention, self).__init__()
        self.hid_dim = hid_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hid_dim, bidirectional=True, batch_first=True)
        # attention, self.attn 的 out_dim 是任意的，这里直接设成 tar_dim
        self.attn = nn.Linear(self.hid_dim, tar_size)
        self.v = nn.Linear(tar_size, 1, bias=False)
		
        self.fc = nn.Linear(self.hid_dim, tar_size)

    def forward(self, sequence):
        batch_size = sequence.shape[0]
        
        # nn.Embedding 要求输入的数据为 [batch_size, seq_len]
        # 输出为 [batch_szie, seq_len, embed_dim]
        embeds = self.word_embedding(sequence)
        seq_len = embeds.shape[1]
        
        # nn.LSTM 要求输入数据为 [batch_size, seq_len, embed_dim]
        # hidden_0 未给出时会被默认设置为0
        # lstm_out  [batch_size, seq_len, hid_dim*num_directions] 
        # h_n  [n_layers*num_directions, batch_size, embed_dim]
        # h_c 用于生成 h_n, 不用于输出
        lstm_out, (h_n, h_c) = self.lstm(embeds)
        # [bs, seq_len, hid_size]
        lstm_out = lstm_out[:, :, :self.hid_dim] + lstm_out[:, :, self.hid_dim:]

        lstm_out = F.tanh(lstm_out)

        attention = torch.zeros(seq_len, batch_size)   # [seq_len, batch_size]
        for t in range(seq_len):
            
            # attention_iuput [batch_size, hid_dim*2]
            # h_n[0, :, :] shape [batch_size, hid_dim]
            attention_iuput = lstm_out[:, t, :]
            # mt [batch_size, tar_size]
            mt = self.attn(attention_iuput)
            attention_t = self.v(mt)  # [batch_size, 1]
            attention[t] = attention_t.squeeze(1)
        
        attention = F.softmax(attention, dim=0)
        attention = attention.transpose(0,1).unsqueeze(1) # [batch_size, 1, seq_len]
        # lstm_out [batch_size, seq_len, hid_dim*2]
        c = torch.bmm(attention, lstm_out) # [batch_size, 1, hid_dim*2]
        c = c.squeeze(1) # [batch_size, hid_dim*2]
        
        output = self.fc(c) # [batch_size, tar_size]

        return output


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, './BiLSTM_Attention.pth')
    print('The best model has been saved')


def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=2):
    try:
        checkpoint = torch.load('./model_bert.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('-----Continue Training-----')
    except:
        print('No Pretrained model!')
        print('-----Training-----')

    model.to(device)
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            output = model(batch[0])
            loss = criterion(output, batch[-1])
            if((i+1) % 8)==0:
                optimizer.step()        # 反向传播，更新网络参数
                optimizer.zero_grad()   # 清空梯度

        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, val_loader)


best_score = 0
def eval(model, optimizer, val_loader):
    model.eval()
    _, batch_A = next(enumerate(val_loader))
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        output = model(batch[0])
        label_ids = batch[-1].numpy()

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_val = ms.accuracy_score(label_ids, pred_val)

    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)


tar_size = len(tar2idx_dict)
vocab_size = len(word2idx_dict)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

model = BiLSTM_Attention(embed_dim=embed_size, hid_dim=64, vocab_size=vocab_size, tar_size=tar_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


set_seed(1)
train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2)






loss_his = []
step, (batch_x, batch_y) = next(enumerate(loader))
for epoch in range(20):
    model = model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(loader)):
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
        print('EPOCH: ', epoch,   '         loss:', loss.item())
        print('  test_acc:', acc_test)


# loss 曲线图
import matplotlib.pyplot as plt
plt.plot(loss_his)
plt.ylabel('loss', fontsize=20)
plt.xlabel('eopch', fontsize=20)
plt.tick_params(labelsize=20)
plt.title('EOPCH_LOSS_BiLSTM_Attention', fontsize=20)
plt.grid()
plt.show()


# model save
torch.save(model.state_dict(), save_path+'BiLSTM_Attention.pkl')


x = [1, 2, 3, 5]

[1 if i >2 else 0 for i in x ]
