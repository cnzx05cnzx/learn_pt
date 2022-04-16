import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible
n_data = torch.ones(100, 2)

# 三组数据生成，x两列数值，y一列标签
x0 = torch.normal(5 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
x2 = torch.normal(10 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y2 = y1 * 2  # class0 y data (tensor), shape=(100, 1)
# 数据连接
x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer
#
# # The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# # x, y = Variable(x), Variable(y)
#
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

# 定义网络  n_feature为输入的维度, n_hidden为神经元维度, n_output为输出维度，三个标签，根据独热编码3维)
# 采用nn.Sequential生成，前向传播相较a娇柔
Net = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)
print(Net)

optimizer = torch.optim.SGD(Net.parameters(), lr=0.02, momentum=0.9)
loss_func = nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()  # something about plotting

temp = 0
for t in range(100):
    out = Net(x)  # input x and predict based on x
    # 预测在前，标签在后，且标签为初始格式
    loss = loss_func(out, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 20 == 0:
        # plot and show learning process
        plt.cla()
        # 获取分数最高的元素值与索引值，二维
        prediction = torch.max(out, 1)[1]

        pred_y = prediction.numpy()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')

        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        if int(temp * 10) == int(accuracy * 10):
            break
        else:
            temp = accuracy
        plt.text(1.5, -4, 'Accuracy=%.2f N=%d' % (accuracy, t), fontdict={'size': 15, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
