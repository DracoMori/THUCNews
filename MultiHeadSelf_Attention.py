'''
date: 2020/2/9
author: 流氓兔233333
content: MultiHead self_Attention  
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

id2word = {v: k for k, v in word2idx_dict.items()}

# load tensor data
train_tensor_tuple = pickle.load(open(save_path+'train_tensor_tuple.pkl', 'rb'))
val_tensor_tuple = pickle.load(open(save_path+'val_tensor_tuple.pkl', 'rb'))
x_train_tensor, y_train_tensor = train_tensor_tuple[0], train_tensor_tuple[1]
x_val_tensor, y_val_tensor = val_tensor_tuple[0], val_tensor_tuple[1]

del train_tensor_tuple, val_tensor_tuple


def attention_masks(input_ids):
    atten_masks = []  
    for seq in input_ids:                       
        seq_mask = [float(i != word2idx_dict['PAD']) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    atten_masks = torch.LongTensor(atten_masks)
    return atten_masks

train_masks = attention_masks(x_train_tensor)
train_masks.shape, x_train_tensor.shape

val_masks = attention_masks(x_val_tensor)
val_masks.shape, x_val_tensor.shape

train_masks[0], x_train_tensor[0]



batch_size = 128
dataset_trn = Data.TensorDataset(x_train_tensor, train_masks, y_train_tensor)
loader_trn = Data.DataLoader(dataset_trn, batch_size, True)

dataset_val = Data.TensorDataset(x_val_tensor, val_masks, y_val_tensor)
loader_val = Data.DataLoader(dataset_val, len(dataset_val), True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, atten_masks):
        '''
        Q: [batch_size, n_heads, d_q=d_k, seq_len]
        K: [batch_size, n_heads, d_k, seq_len]
        V: [batch_size, n_heads, d_v, seq_len]       
        '''
        n_heads = Q.shape[1]
        seq_len = Q.shape[-1]
        
        # transpose 相当于转置
        d_k = Q.shape[2]
        # scores [batch_size, n_heads, seq_len(行), seq_len(列)]
        scores = torch.matmul(K.transpose(-1, -2), Q) / np.sqrt(d_k)

        atten_masks = torch.cat([atten_masks.unsqueeze(1) for i in range(seq_len)], dim=1)
        atten_masks = torch.cat([atten_masks.unsqueeze(1) for i in range(n_heads)], dim=1)
        # atten masks
        if atten_masks is not None:
            scores = scores.masked_fill(atten_masks == 0, -1e9) # mask步骤，用 -1e9 代表负无穷

        del atten_masks

        # 按列进行归一化  attn [batch_size, n_heads, seq_len(行), seq_len(列)]
        attn = nn.Softmax(dim=-1)(scores)
        # context [batch_size, n_head, d_v, seq_len]
        context = torch.matmul(V, attn)
        # context [batch_size, n_head, seq_len, d_v]
        context = context.transpose(-1, -2)
        # context [batch_size, seq_len, d_v*n_heads]
        context = torch.cat([context[:, head, :, :] for head in range(context.shape[1])], dim=2)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim, d_k, d_v, n_heads, vocab_size, tar_size):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_k
        self.d_q = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.hidden_size = 64
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, self.hidden_size, batch_first=True, bidirectional=True)
        # d_q = d_k
        self.W_Q = nn.Linear(self.hidden_size*2, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(self.hidden_size*2, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(self.hidden_size*2, d_v*n_heads, bias=False)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(n_heads*d_v, tar_size)

    def forward(self, sequence, atten_masks):
       
        # sequence  [batch_size, seq_len] 
        # embeds [batch_szie, seq_len, embed_dim]
        embeds = self.word_embedding(sequence)
        # lstm_out [bs, seq_len, hidden_size*2]
        lstm_out, _ = self.lstm(embeds)
        batch_size, seq_len = embeds.shape[0], embeds.shape[1]
        
        Q_t = self.W_Q(lstm_out).transpose(1, 2)    # [batch_size, d_q*n_heads, seq_len]
        K_t = self.W_K(lstm_out).transpose(1, 2)    # [batch_size, d_q*n_heads, seq_len]
        V_t = self.W_V(lstm_out).transpose(1, 2)    # [batch_size, d_q*n_heads, seq_len]
        
        Q = torch.zeros(batch_size, self.n_heads, self.d_q, seq_len)   
        K = torch.zeros(batch_size, self.n_heads, self.d_k, seq_len) 
        V = torch.zeros(batch_size, self.n_heads, self.d_v, seq_len)
        
        for head in range(self.n_heads):
            Q[:, head, :, :] = Q_t[:, head*self.d_q: (head+1)*self.d_q, :]
            K[:, head, :, :] = K_t[:, head*self.d_k: (head+1)*self.d_k, :]
            V[:, head, :, :] = V_t[:, head*self.d_v: (head+1)*self.d_v, :]
        
        # print(Q.shape, K.shape, V.shape)
        # context [batch_size, seq_len, d_v*n_heads]
        # attn [batch_size, n_head, seq_len, seq_len]
        context, self.attn = ScaledDotProductAttention()(Q, K, V, atten_masks)    
        importance = torch.argmax(self.attn, dim=2)

        context_mean = torch.mean(context, dim=1) # [batch_size,  d_v*n_heads]
        context_mean = self.drop(context_mean)
        output = self.fc(context_mean) # [batch_szie, tar_size]

        return output, importance



def Attention_plot(batch_x, batch_y, importance, id2word, id2tar, pred_y, num_plot=None, mark=None):
    '''
    # batch_x [batch_size, seq_len]
    # importance [batch_size, n_heads, seq_len]
    num_plot: 画图个数
    mark: 需要屏蔽的词的id mark = [word2idx['<PAD>'], word2idx['<UNK>']]
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    
    batch_x, batch_y = batch_x.numpy(), batch_y.numpy()

    seq_len = batch_x.shape[-1]
    n_heads = importance.shape[1]

    # 只画前num_plot副图
    if not num_plot:
        batch_size = batch_x.shape[0]
    else:
        batch_size = num_plot

    for n in range(batch_size):
        n_id = random.choice(range(batch_x.shape[0]))
        dict_ = {i:np.array([list(importance[n_id][head].numpy()).count(i) for head \
                            in range(n_heads)]).mean() for i in range(seq_len)}
        dict_unique = {}
        for i in range(seq_len):
            dict_unique[batch_x[n_id][i]] = dict_unique.get(batch_x[n_id][i], 0) + dict_[i]
        
        dict_unique = {k:v/seq_len for k, v in dict_unique.items()}
        if mark:
            for w in mark:
                dict_unique[w] = 0
        
        dict_sort = sorted(dict_unique.items(), key=lambda item: item[1], reverse=True)
        words_imp = [[k, v] for k, v in dict_sort]
        
        data_x = [id2word[x] for x, _ in words_imp]
        data_y = [y for _, y in words_imp]
        
        ax = sns.heatmap(np.array(data_y[:10]).reshape((1, len(data_y[:10]))), xticklabels=data_x[:10], cmap='Blues')
        ax.set_xlabel('Each Word in the Sentence', fontsize=20)
        ax.set_ylabel('Words importance', fontsize=20)
        plt.title('real type: {},  pred type: {}'.format(id2tar[batch_y[n_id]], id2tar[pred_y[n_id]]), fontsize=20)
        plt.tick_params(labelsize=20)
        plt.show()

best_score = 0
def train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2):
    try:
        checkpoint = torch.load(save_path+'model_BiLSTM_Atten.pth', map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('-----Continue Training-----')
    except:
        print('No Pretrained model!')
        print('-----Training-----')

    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        # [x_train_tensor, train_masks, y_train_tensor]
        # _, batch = next(enumerate(loader_trn))
        for i, batch in enumerate(tqdm(loader_trn)):
            output, _ = model(batch[0], batch[1])
            loss = criterion(output, batch[-1])
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, loader_val)

def eval(model, optimizer, loader_val):
    model.eval()
    _, batch = next(enumerate(loader_val))
    with torch.no_grad():
        output, _ = model(batch[0], batch[1])
        label_ids = batch[-1].numpy()

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_val = ms.accuracy_score(label_ids, pred_val)

    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)



def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path+'model_BiLSTM_Atten.pth')
    print('The best model has been saved')





tar_size = len(tar2idx_dict)
vocab_size = len(word2idx_dict)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

model = MultiHeadAttention(embed_dim=embed_size, d_k=32, d_v=16, n_heads=3, vocab_size=vocab_size, tar_size=tar_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=5)








step, (batch_x, atten_masks, batch_y) = next(enumerate(loader))
output, importance = model(batch_x, atten_masks)

# 继续训练
model = MultiHeadAttention(embed_dim=embed_size, d_k=32, d_v=16, n_heads=3, vocab_size=vocab_size, tar_size=tar_size)
model.load_state_dict(torch.load(save_path+'MultiHeadAttention.pkl'))
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_his = []
EPOCH = 10
# step, (batch_x, batch_y) = next(enumerate(loader))
for epoch in tqdm(range(EPOCH)):
    model = model.train()
    for step, (batch_x, atten_masks, batch_y) in enumerate(loader):
        output, importance = model(batch_x, atten_masks)

        loss = loss_fun(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_his.append(loss.item())

    if epoch % 1 == 0:
        model = model.eval()
        output, _ = model(x_val_tensor, val_masks)

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_test = ms.accuracy_score(y_val_tensor.numpy(), pred_val)
        print('  EPOCH: (%d/%d)' % (epoch+1, EPOCH))
        print(' loss: %f' % loss.item(), '                test_acc:%f' % acc_test)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), save_path+'MultiHeadAttention.pkl')
        print('已保存')



# loss 曲线图
import matplotlib.pyplot as plt
plt.plot(loss_his)
plt.ylabel('loss', fontsize=20)
plt.xlabel('eopch', fontsize=20)
plt.tick_params(labelsize=20)
plt.title('LOSS_EPOCH_MultiHeadAttention', fontsize=20)
plt.grid()
plt.show()


# model save
torch.save(model.state_dict(), save_path+'MultiHeadAttention.pkl')


# attention 可视化
model = model.eval()
output, importance = model(x_val_tensor, val_masks)
_, prediction = torch.max(F.softmax(output, dim=1), 1)
pred_val = prediction.data.numpy().squeeze()

id2word = {v: k for k,v in word2idx_dict.items()}
id2tar = {v: k for k,v in tar2idx_dict.items()}
mark = None
Attention_plot(x_val_tensor, y_val_tensor, importance, id2word, id2tar, pred_val, num_plot=5, mark=mark)

word2idx_dict['PAD']




# ============== 真实新闻文本分类  =================
path = 'D:/VSCode/pyStudy/NLP/data/THUCNews/'
path_model = 'D:/VSCode/pyStudy/NLP/data/THUCNews/'

import json
import numpy as np
class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):                                 
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)
def save_file(filename, dic):
    '''save dict into json file'''
    with open(filename,'w',  encoding='utf-8') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

# 读取js文件
def load_file(filename):
    '''load dict from json file'''
    with open(filename,"r", encoding='utf-8') as json_file:
	    dic = json.load(json_file)
    return dic

# load packages
import torch
import torch.utils.data as Data
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import sklearn.metrics as ms

# 构造 tar2idx and word2idx
label_all = ['游戏', '科技', '娱乐', '家居', '星座', '时尚', '教育', 
            '股票', '彩票', '时政', '体育', '财经', '房产', '社会']
tar2idx = {w: i for i, w in enumerate(label_all)}
word2idx = load_file(str(path+'word2idx'))

# load data
x_train = load_file(str(path+'x_train'))
y_train = load_file(str(path+'y_train'))
x_val = load_file(str(path+'x_val'))
y_val = load_file(str(path+'y_val'))

x_train_tensor, y_train_tensor = torch.LongTensor(x_train), torch.LongTensor(y_train)
x_val_tensor, y_val_tensor = torch.LongTensor(x_val), torch.LongTensor(y_val)

# dataloader 输入word2idx和tar2idx的数据类型都是LongTensor
batch_size = 128
dataset = Data.TensorDataset(x_train_tensor, y_train_tensor)
loader = Data.DataLoader(dataset, batch_size, True)


# train model 
tar_size = len(tar2idx)
vocab_size = len(word2idx)
embed_size = 2 * int(np.floor(np.power(vocab_size, 0.25)))

model = MultiHeadAttention(embed_dim=embed_size, d_k=32, d_v=16, n_heads=3, vocab_size=vocab_size, tar_size=tar_size)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


step, (batch_x, batch_y) = next(enumerate(loader))
output, importance = model(batch_x)

loss_his = []
EPOCH = 20
# step, (batch_x, batch_y) = next(enumerate(loader))
for epoch in tqdm(range(EPOCH)):
    model = model.train()
    for step, (batch_x, batch_y) in enumerate(loader):
        output, _ = model(batch_x)

        loss = loss_fun(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_his.append(loss.item())

    if epoch % 1 == 0:
        model = model.eval()
        output, _ = model(x_val_tensor)

        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.data.numpy().squeeze()
        acc_test = ms.accuracy_score(y_val_tensor.numpy(), pred_val)
        print('  EPOCH: (%d/%d)' % (epoch+1, EPOCH))
        print(' loss: %f' % loss.item(), '                test_acc:%f' % acc_test)

    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), save_path+'MultiHeadAttention.pkl')


model = model.eval()
output, importance = model(x_val_tensor)
_, prediction = torch.max(F.softmax(output, dim=1), 1)
pred_val = prediction.data.numpy().squeeze()

id2word = {v: k for k,v in word2idx.items()}
id2tar = {v: k for k,v in tar2idx_dict.items()}
mark = [word2idx['<PAD>'], word2idx['<UNK>']]
Attention_plot(x_val_tensor, y_val_tensor, importance, id2word, id2tar, pred_val, num_plot=5, mark=mark)

