import torch
from torch import nn, optim
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import torch.utils.data as Data

torch.manual_seed(114514)

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
# print(features)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
labels = labels.view(-1, 1)

# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
dataset = Data.TensorDataset(features, labels)

loader = Data.DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,  # 每次训练打乱数据， 默认为False

)


class Net(nn.Module):
    # 初始化定义每一层的输入大小，输出大小
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)

    # 前向传播过程
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


net = Net()
cal_loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(5):
    sum_loss = []
    for step, (x, y) in enumerate(loader):
        prediction = net(x)  # input x and predict based on x

        loss = cal_loss(prediction, y)  # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        sum_loss.append(loss.item())

    print('epoch %d, loss: %f' % (epoch, sum(sum_loss) / len(sum_loss)))

pred_y = net(features[:10])

for x, y in zip(pred_y, labels[:10]):
    print(x.data, y.data)
