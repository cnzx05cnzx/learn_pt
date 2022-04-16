from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

import unicodedata
import string


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 返回字母 letter 的索引 index
def letterToIndex(letter):
    return all_letters.find(letter)


# 字母独热编码
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    # 把字母 letter 的索引设定为1，其它都是0
    tensor[0][letterToIndex(letter)] = 1
    return tensor.to(device)


# 单词独热编码
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历单词中的所有字母，对每个字母 letter 它的索引设定为1，其它都是0
    # print(tensor)
    for index, letter in enumerate(line):
        tensor[index][0][letterToIndex(letter)] = 1
    return tensor.to(device)


import torch.nn as nn


class RNN(nn.Module):
    # 初始化定义每一层的输入大小，输出大小
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # 前向传播过程
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    # 初始化隐藏层状态 h0
    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
rnn = rnn.to(device)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long).to(device)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category = ', category, '/ line = ', line)

criterion = nn.NLLLoss()
learning_rate = 0.005


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    # RNN的循环
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 更新参数
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()
