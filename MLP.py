# -*- coding: utf-8 -*-
from dataLoaderSAE import GetLoaderSAE
from pickle import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import MLP
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# 1. 检查是否可以利用GPU
trainOnGpu = torch.cuda.is_available()

if not trainOnGpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

# 2. 解压缩训练数据集和测试数据集
trainF1 = open("./data/trainData3", "rb")
trainF2 = open("./data/trainData4", "rb")
testF1 = open("./data/testData3", "rb")
testF2 = open("./data/testData4", "rb")
trainDt1 = load(trainF1, encoding="latin1")
trainDt2 = load(trainF2, encoding="latin1")
testDt1 = load(testF1, encoding="latin1")
testDt2 = load(testF2, encoding="latin1")

trainSAESpecs1 = trainDt1["SAESpecs"]
trainLabels1 = np.array(trainDt1["labels"])

trainSAESpecs2 = trainDt2["SAESpecs"]
trainLabels2 = np.array(trainDt2["labels"])

testSAESpecs1 = testDt1["SAESpecs"]
testLabels1 = np.array(testDt1["labels"])

testSAESpecs2 = testDt2["SAESpecs"]
testLabels2 = np.array(testDt2["labels"])

trainData1 = GetLoaderSAE(trainSAESpecs1, trainLabels1)
trainData2 = GetLoaderSAE(trainSAESpecs2, trainLabels2)

testData1 = GetLoaderSAE(testSAESpecs1, testLabels1)
testData2 = GetLoaderSAE(testSAESpecs2, testLabels2)



# 3. 加载数据集
batchSize = 8

# 加载训练集和测试集
trainLoader1 = torch.utils.data.DataLoader(trainData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
trainLoader2 = torch.utils.data.DataLoader(trainData2, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader1 = torch.utils.data.DataLoader(testData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader2 = torch.utils.data.DataLoader(testData2, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)

# 4. 训练SAE模型
# 获取并打印SAE神经网络模型
model = MLP()
print(model)

# 使用GPU
if trainOnGpu:
    model.cuda()

# 设置模型训练相关参数
# 使用空间余弦嵌入损失函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降，学习率lr=0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)
# 训练模型的次数
epochs = 100

# 检查测试损失是否有变化
testLossMin = np.Inf

for epoch in range(1, epochs + 1):

    # 用来实时追踪训练损失和测试损失的变化
    trainLoss = 0.0
    ################
    # 训练集运算过程 #
    ################
    model.train()
    # with torch.no_grad():
    for data, target in trainLoader1:
        # 将数据流转换成GPU可用的形式
        if trainOnGpu:
            data, target = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.LongTensor)
        target = target.reshape(target.shape[0], 1)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传递: 通过输入数据到网络模型计算输出数据
        output = model(data)
        # 计算批损失
        loss = criterion(output, target)
        # 反向传递: 通过损失计算梯度并更新网络模型参数
        loss.backward()
        # 参数更新
        optimizer.step()
        # 更新训练损失
        trainLoss += loss.item() * data.size(0)
    # 计算平均损失
    trainLoss = trainLoss / len(trainLoader1.sampler)
    # 显示训练集与测试集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, trainLoss))

################
# 测试集运算过程 #
################
model.eval()
testLoss = 0.0
classes = ["拖船", "帆船", "客渡船", "滚装船", "海洋哺乳生物"]
classCorrect = list(0. for i in range(5))
classTotal = list(0. for i in range(5))
for data, target in testLoader1:
    # 将数据流转换成GPU可用的形式
    if trainOnGpu:
        data, target = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.LongTensor)
    target = target.reshape(target.shape[0], 1)
    # 前向传递: 通过输入数据到网络模型计算输出数据
    output = model(data)
    # 计算批损失
    loss = criterion(output, target)
    # 更新平均测试损失
    testLoss += loss.item() * data.size(0)
    # 将输出概率转换为对应的类别
    _, pred = torch.max(output, 1)
    # 将预测类别和真实的标签进行对比
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not trainOnGpu else np.squeeze(correct_tensor.cpu().numpy())
    # 计算测试集每一类的平均精度
    for i in range(batchSize):
        label = target.data[i]
        classCorrect[label] += correct[i].item()
        classTotal[label] += 1

testLoss = testLoss / len(testLoader1.dataset)
print('Test Loss: {:.6f}\n'.format(testLoss))

for i in range(5):
    if classTotal[i] > 0:
        print('Test Accuracy of %20s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * classCorrect[i] / classTotal[i],
            np.sum(classCorrect[i]), np.sum(classTotal[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(classCorrect) / np.sum(classTotal),
    np.sum(classCorrect), np.sum(classTotal)))