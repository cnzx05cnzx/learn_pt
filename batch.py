import torch
import torch.utils.data as Data

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# print(x, y)
batch_size = 3
# 先定义数据集，之后载入loder用batch——size提升训练速度
dataset = Data.TensorDataset(x, y)
loder = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)
for i in range(3):
    for step, (tx, ty) in enumerate(loder):
        print("epoch %d | step %d |" % (i, step), tx, ty)
