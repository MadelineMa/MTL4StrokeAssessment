import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.utils.data as Data

# 创建解释器
# import lime
# import lime.lime_tabular

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# train_path = tf.keras.utils.get_file(
#     "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# train.to_csv('D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\iris.csv', encoding='utf_8_sig')
df = pd.read_csv('D:\\Mady\\github\\MLCode\\Stroked\\GN-data\\iris.csv')
df = df.sample(frac=1) # 乱序
df.head()

# 数据标准化
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
# Separating out the features
x = df.loc[:, features].values
x.astype(np.float)
# Separating out the target
y_label = df.loc[:,['Species']].values
y_label.shape=(120)
# Standardizing the features
x = StandardScaler().fit_transform(x)

X_train,Y_train,X_test,Y_test = x[:-30],y_label[:-30],x[-30:],y_label[-30:]

X_train = torch.from_numpy(X_train).float() # 输入 x 张量
X_test = torch.from_numpy(X_test).float()
Y_train = torch.from_numpy(np.array(Y_train)).long() # 输入 y 张量
Y_test = torch.from_numpy(np.array(Y_test)).long()
batch_size=10
# Dataset
train_dataset = Data.TensorDataset(X_train, Y_train)
test_dataset = Data.TensorDataset(X_test, Y_test)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 网络的构建
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.3))
        self.inLayer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hiddenLayer = nn.Sequential(*layers)
        self.outLayer = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.inLayer(x)
        out = self.relu(out)
        out = self.hiddenLayer(out)
        out = self.outLayer(out)
        out = self.softmax(out)
        return out
    
input_size = 4
hidden_size = 20
num_classes = 3
n_layers = 1
# 网络初始化
model = NeuralNet(input_size, hidden_size, num_classes, n_layers)
print(model)

# 模型的训练
num_epochs = 30
learning_rate = 0.001
# ------------------
# Loss and optimizer
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---------------
# Train the model
# ---------------
model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        # images = images.reshape(-1, 28*28).to(device)
        # labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            # 计算每个batch的准确率
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100*correct/total
            # 打印结果
            print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {}, Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, acc, loss.item()))
    # -----------------------------------
    # Test the model(每一个epoch打印一次)
    # -----------------------------------
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            # images = images.reshape(-1, 28*28).to(device)
            # labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network test dataset: {} %'.format(100 * correct / total))
        print('-'*10)
        
# 定义预测函数
def batch_predict(data, model=model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """
    X_tensor = torch.from_numpy(data).float()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_tensor = X_tensor.to(device)
    logits = model(X_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# 测试一下上面的函数
test_Data = x[:10]
prob = batch_predict(data=test_Data, model=model)
print(prob)

# 创建解释器
# targets =  ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa']
# features_names=['sepal length','sepal width','petal length','petal width']
# explainer = lime.lime_tabular.LimeTabularExplainer(x,
                                                  #  feature_names=features_names,
                                                  # class_names=targets,
                                                  # discretize_continuous=True)

# 解释某一个样本
# exp = explainer.explain_instance(x[5],
                                # batch_predict,
                                # num_features=5,
                                # top_labels=5)


# 结果的展示
# exp.show_in_notebook(show_table=True, show_all=False)
# exp.as_pyplot_figure(label=1)

import shap
shap.initjs() # 用来显示的

# 新建一个解释器
# 这里传入两个变量, 1. 模型; 2. 训练数据
explainer = shap.KernelExplainer(batch_predict, x)
print(explainer.expected_value) # 输出是三个类别概率的平均值
# 选择一个数据进行解释(还是选择第5个数据)
shap_values = explainer.shap_values(x[5])
# 对单个数据进行解释
shap.force_plot(base_value=explainer.expected_value[1], 
                shap_values=shap_values[1], 
                feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
               features=x[5])
# 对特征重要度进行解释S
shap_values = explainer.shap_values(x)

# --------
# 进行绘图
# --------
shap.summary_plot(shap_values=shap_values,
                 features=x,
                 feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
                 plot_type='bar')