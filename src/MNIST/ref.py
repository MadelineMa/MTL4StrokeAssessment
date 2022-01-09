import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# 加载数据集
def get_data():
    # 定义数据预处理操作, transforms.Compose将各种预处理操作组合在一起
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_dataset

# 构建模型，三层神经网络
class batch_net(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, out_dim):
        super(batch_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.BatchNorm1d(hidden1_dim), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden1_dim, hidden2_dim), nn.BatchNorm1d(hidden2_dim), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden2_dim, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



if __name__ == "__main__":
    # 超参数配置
    batch_size = 256
    learning_rate = 1e-2
    num_epoches = 5
    # 加载数据集
    train_dataset, test_dataset = get_data()
    # 导入网络，并定义损失函数和优化器
    model = batch_net(28*28, 300, 100, 10)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opitimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # 开始训练
    for i in range(num_epoches):
        for img, label in train_dataset:
            img = img.view(batch_size, -1)
            img = Variable(img)
            #print(img.size())
            label = Variable(label)
            # forward
            out = model(img)
            loss = criterion(out, label)
            # backward
            opitimizer.zero_grad()
            loss.backward()
            opitimizer.step()
            # 打印
            print("epoches= {},loss is {}".format(i, loss))
    # 测试
    model.eval()
    count = 0
    for data in test_dataset:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img, volatile=True)
        #label = Variable(label, volatile=True)
        out = model(img)
        _, predict = torch.max(out, 1)
        if predict == label:
            count += 1
    print("acc = {}".format(count/len(test_dataset)))