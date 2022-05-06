import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, download=False,
                                               transform=transforms.ToTensor())

# plt.show()
batch_size = 256

# a,b=mnist_train.data[0],mnist_train.targets[0]
# print(a,b)
#
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    # 初始化定义每一层的输入大小，输出大小
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    # 前向传播过程
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer(x)
        return x


net = Net()
cal_loss = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(1, 6):
    acc1 = acc2 = 0

    # sum_loss1 = sum_loss2 = []
    for step, (x, y) in enumerate(train_iter):
        prediction = net(x)  # input x and predict based on x

        loss = cal_loss(prediction, y)  # must be (1. nn output, 2. target)
        opt.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        opt.step()  # apply gradients
        tar = torch.max(prediction, 1)[1].data.numpy()

        acc1 += (tar == y.data.numpy()).astype(int).sum()
        # sum_loss1.append(loss.data.item())

    with torch.no_grad():
        net.eval()
        for step, (x, y) in enumerate(test_iter):
            prediction = net(x)
            loss = cal_loss(prediction, y)
            tar = torch.max(prediction, 1)[1].data.numpy()
            acc2 += (tar == y.data.numpy()).astype(int).sum()
            # sum_loss2.append(loss.data.item())
        net.train()

    print('epoch %d, train_acc: %f, test_acc %f' % (epoch, acc1 / len(mnist_train), acc2 / len(mnist_test)))
