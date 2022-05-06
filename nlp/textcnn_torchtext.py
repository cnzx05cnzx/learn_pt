import pandas as pd
from torchtext.legacy.data import Field, TabularDataset
import pkuseg
import torch

# 使用torchtext的工具
torch.manual_seed(114514)

# 读取数据
df = pd.read_csv('../data/drink/train.csv')
# data = pd.read_csv('../data/drink/携程酒店.csv')


seg = pkuseg.pkuseg()


# 定义分词函数
def tokenizer(text):
    return seg.cut(text)


TEXT = Field(sequential=True, tokenize=tokenizer, fix_length=385)
LABEL = Field(sequential=False, use_vocab=False)

# 因为测试集不要label，所以在field中令label列传入None
# 因为训练集要label，所以在field中令label列传入
test_field = [('label', None), ('content', TEXT)]
train_field = [('label', LABEL), ('content', TEXT)]
train, val = TabularDataset.splits(
    path='../data/drink', train='train.csv',
    validation='valid.csv', format='csv',
    fields=train_field, skip_header=True)

TEXT.build_vocab(train, min_freq=3, vectors='glove.6B.50d')

from torchtext.legacy.data import BucketIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64

# 同时对训练集和验证集进行迭代器构建
train_iter, val_iter = BucketIterator.splits(
    (train, val),
    batch_sizes=(train_batch_size, val_batch_size),
    device=device,
    sort_key=lambda x: len(x.content),
    sort_within_batch=False,
    repeat=False
)

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, emb_dim, kernel_sizes, kernel_num):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding_dropout = nn.Dropout(0.5)
        # 使用nn.ModuleList来装三个nn.Sequential构成的卷积块
        self.convs = nn.ModuleList([
            # 使用nn.Sequential构成卷积块,每个卷积块装有一层卷积和LeakyReLU激活函数
            nn.Sequential(
                nn.Conv1d(in_channels=emb_dim, out_channels=kernel_num, kernel_size=size),
                nn.LeakyReLU(),
            )
            for size in kernel_sizes])
        in_features = kernel_num * len(kernel_sizes)
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features // 2)
        self.fc_dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=in_features // 2, out_features=2)

    def forward(self, x):
        # 初始输入格式为(length, batch_size)
        out = self.embedding(x)
        out = self.embedding_dropout(out)
        # (length, batch_size, emb) -> (batch_size, emb, length)
        out = torch.transpose(out, 1, 2)
        out = torch.transpose(out, 0, 2)
        print(out.shape)
        out = [conv(out) for conv in self.convs]
        # stride为步幅
        out = [F.max_pool1d(one, kernel_size=one.size(2), stride=2) for one in out]
        # 拼接维度1，并去掉维度2
        out = torch.cat(out, dim=1).squeeze(2)
        out = self.linear1(F.leaky_relu(out))
        out = self.fc_dropout(out)
        out = self.linear2(F.leaky_relu(out))
        return out


# 4种卷积核，每种60个，拼接后就有240种特征
model = TextCNN(50, [3, 4, 5, 6], 60).to(device)

import numpy as np

import torch.optim as optim
import torch.nn.functional as F

learning_rate = 1e-3
# 定义优化器和损失函数
# Adam是有自适应学习率的优化器，利用梯度的一阶矩估计和二阶矩估计动态调整学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 因为是多分类问题，所以使用交叉熵损失函数，pytorch的交叉熵损失函数是会做softmax的，所以在模型中没有添加softmax层
criterion = F.cross_entropy
# 设置学习率下降策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# -----------------------------------模型训练--------------------------------------
epochs = 100
stop = 10
pos = 0
best_valid_acc = float('-inf')
model_save_path = '../model/torchtext.pkl'

for epoch in range(epochs):
    loss_one_epoch = 0.0
    correct_num = 0.0
    total_num = 0.0

    for i, batch in enumerate(train_iter):
        model.train()
        label, content = batch.label, batch.content
        # 进行forward()、backward()、更新权重
        optimizer.zero_grad()
        pred = model(content)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        total_num += label.size(0)
        # 预测有多少个标签是预测中的，并加总
        correct_num += (torch.argmax(pred, dim=1) == label).sum().float().item()
        loss_one_epoch += loss.item()

    loss_avg = loss_one_epoch / len(train_iter)

    print("Train: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".
          format(epoch + 1, epochs, loss_avg, correct_num / total_num))

    # ---------------------------------------验证------------------------------
    loss_one_epoch = 0.0
    total_num = 0.0
    correct_num = 0.0

    model.eval()
    for i, batch in enumerate(val_iter):
        label, content = batch.label, batch.content
        pred = model(content)
        pred.detach()
        # 计算loss
        loss = criterion(pred, label)
        loss_one_epoch += loss.item()

        # 统计预测信息
        total_num += label.size(0)
        # 预测有多少个标签是预测中的，并加总
        correct_num += (torch.argmax(pred, dim=1) == label).sum().float().item()
        loss_one_epoch += loss.item()

    loss_avg = loss_one_epoch / len(train_iter)

    # 学习率调整
    scheduler.step()

    # 打印验证集的准确率，numpy的trace()就是求对角线元素的和sum()是求所有元素的和
    print('{} Acc:{:.2%}'.format('Valid', correct_num / total_num))

    # 每个epoch计算一下验证集准确率如果模型效果变好，保存模型
    if (correct_num / total_num) > best_valid_acc:
        print("超过最好模型,保存")
        best_valid_acc = (correct_num / total_num)
        torch.save(model.state_dict(), model_save_path)
        pos = 0
    else:
        pos = pos + 1
        if pos > stop:
            print("模型基本无变化，停止训练")
            print("训练集最高准确率为%.2f" % (best_valid_acc))
            break
