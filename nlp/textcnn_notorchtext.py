from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import pkuseg

# 没有使用torchtext的工具

# 读取数据
data = pd.read_csv('../data/drink/train.csv')
# data = pd.read_csv('../data/drink/携程酒店.csv')

seg = pkuseg.pkuseg()

# 分词
data['cut'] = data["review"].apply(lambda x: list(seg.cut(x)))

# 生成词语表
# with open("../data/drink/vocab.txt", 'w', encoding='utf-8') as fout:
#     fout.write("<unk>\n")
#     fout.write("<pad>\n")
#     # 使用 < unk > 代表未知字符且将出现次数为1的作为未知字符
#     # 实用 < pad > 代表需要padding的字符(句子长度进行统一)
#     vocab = [word for word, freq in Counter(j for i in data['cut'] for j in i).most_common() if freq > 1]
#     for i in vocab:
#         fout.write(i + "\n")

# 初始化生成 词对序 与 序对词 表
with open("../data/drink/vocab.txt", encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
char2idx = {i: index for index, i in enumerate(vocab)}
idx2char = {index: i for index, i in enumerate(vocab)}
vocab_size = len(vocab)
pad_id = char2idx["<pad>"]
unk_id = char2idx["<unk>"]
# print(char2idx)

sequence_length = 385


# 对输入数据进行预处理
def tokenizer():
    inputs = []
    sentence_char = [[j for j in i] for i in data["cut"]]
    # 将输入文本进行padding
    for index, i in enumerate(sentence_char):
        temp = [char2idx.get(j, unk_id) for j in i]
        if len(temp) < sequence_length:
            for _ in range(sequence_length - len(temp)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs


data_input = tokenizer()

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Embedding_size = 100
Batch_Size = 32
Kernel = 3
Filter_num = 10
Epoch = 50
Dropout = 0.5
Learning_rate = 1e-4

torch.manual_seed(114514)


class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


TextCNNDataSet = TextCNNDataSet(data_input, list(data["label"]))
train_size = int(len(data_input) * 0.8)
test_size = len(data_input) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(TextCNNDataSet, [train_size, test_size])

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)

# nn.Conv2d(in_channels,#输入通道数 out_channels,#输出通道数 kernel_size#卷积核大小 )
num_classs = 2


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        Vocab = vocab_size  ## 已知词的数量
        Dim = Embedding_size  ##每个词向量长度
        Cla = num_classs  ##类别数
        Ci = 1  ##输入的channel数
        Knum = Filter_num  ## 每种卷积核的数量，输出通道
        Ks = [2, 3, 4]  ## 卷积核list，形如[2,3,4]

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        x = self.dropout(x)
        x = self.fc(x)
        return x


model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)


def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()


def train():
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()


# Training cycle
model_train_acc, model_test_acc = [], []
for epoch in range(Epoch):
    train_acc = train()
    test_acc = evaluate()
    if epoch % 10 == 9:
        print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
        print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))
    model_train_acc.append(train_acc)
    model_test_acc.append(test_acc)

plt.plot(model_train_acc)
plt.plot(model_test_acc)
plt.title("The accuracy of textCNN model")
plt.legend(['train', 'test'])
plt.show()
