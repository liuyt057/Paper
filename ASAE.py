# -*- coding: utf-8 -*-
from dataLoaderSAE import GetLoaderSAE
from pickle import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import SAE
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from RGNU import GaussianNoise
from AEMU import AEMU
from contrastiveLoss import ContrastiveLoss


# 1. 检查是否可以利用GPU
trainOnGpu = torch.cuda.is_available()

if not trainOnGpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

# 2. 解压缩训练数据集和测试数据集
trainF1 = open("./data/trainData1", "rb")
trainF2 = open("./data/trainData2", "rb")
testF1 = open("./data/testData1", "rb")
testF2 = open("./data/testData2", "rb")
trainDt1 = load(trainF1, encoding="latin1")
trainDt2 = load(trainF2, encoding="latin1")
testDt1 = load(testF1, encoding="latin1")
testDt2 = load(testF2, encoding="latin1")

trainFBanks1 = trainDt1["FBanks"].reshape(-1, 1, 32, 32)
trainGBanks1 = np.array(trainDt1["GBanks"])
trainLabels1 = np.array(trainDt1["labels"])

trainFBanks2 = trainDt2["FBanks"].reshape(-1, 1, 32, 32)
trainGBanks2 = np.array(trainDt2["GBanks"])
trainLabels2 = np.array(trainDt2["labels"])

testFBanks1 = testDt1["FBanks"].reshape(-1, 1, 32, 32)
testGBanks1 = np.array(testDt1["GBanks"])
testLabels1 = np.array(testDt1["labels"])

testFBanks2 = testDt2["FBanks"].reshape(-1, 1, 32, 32)
testGBanks2 = np.array(testDt2["GBanks"])
testLabels2 = np.array(testDt2["labels"])

trainData1 = GetLoaderSAE(trainFBanks1, trainGBanks1)
trainData2 = GetLoaderSAE(trainFBanks2, trainGBanks2)

trainData3 = GetLoaderSAE(trainFBanks1, trainLabels1)
trainData4 = GetLoaderSAE(trainFBanks2, trainLabels2)

testData1 = GetLoaderSAE(testFBanks1, testGBanks1)
testData2 = GetLoaderSAE(testFBanks2, testGBanks2)

testData3 = GetLoaderSAE(testFBanks1, testLabels1)
testData4 = GetLoaderSAE(testFBanks2, testLabels2)



# 3. 加载数据集
batchSize = 8

# 加载训练集和测试集
trainLoader1 = torch.utils.data.DataLoader(trainData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
trainLoader2 = torch.utils.data.DataLoader(trainData2, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
trainLoader3 = torch.utils.data.DataLoader(trainData3, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
trainLoader4 = torch.utils.data.DataLoader(trainData4, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader1 = torch.utils.data.DataLoader(testData1, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader2 = torch.utils.data.DataLoader(testData2, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader3 = torch.utils.data.DataLoader(testData3, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)
testLoader4 = torch.utils.data.DataLoader(testData4, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=0)

# 4. 训练SAE模型
# 获取并打印SAE神经网络模型
model = SAE()
print(model)

# 使用GPU
if trainOnGpu:
    model.cuda()

# 是否激活RGNU模块
onRGNU = True

# 是否激活AEMU模块
onAEMU = False

# 设置模型训练相关参数
# 使用空间余弦嵌入损失函数
criterion = nn.CosineEmbeddingLoss(margin=0.2)
# 人为设置伪标签
dumLabel = torch.ones(batchSize)
# 使用随机梯度下降，学习率lr=0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)
# 训练模型的次数
epochs = 100

# 检查测试损失是否有变化
testLossMin = np.Inf

for epoch in range(1, epochs + 1):

    # 用来实时追踪训练损失和测试损失的变化
    trainLoss = 0.0
    testLoss = 0.0

    ################
    # 训练集运算过程 #
    ################
    model.train()
    # with torch.no_grad():
    for data, target in trainLoader1:
        # 将数据流转换成GPU可用的形式
        if onRGNU:
            print("use RGNU")
            data = GaussianNoise(data)
        if trainOnGpu:
            data, target, dumLabel = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.FloatTensor), dumLabel.cuda().type(torch.cuda.FloatTensor)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传递: 通过输入数据到网络模型计算输出数据
        _, output = model(data)
        output = output.reshape(batchSize, -1)
        target = target.reshape(batchSize, -1)
        # 计算批损失
        if onAEMU:
            AEMU.enqueueOrDequeue(output.detach(), target.detach())
            loss = ContrastiveLoss(output, target, output, target)
        else:
            loss = criterion(output, target, dumLabel)
        # 反向传递: 通过损失计算梯度并更新网络模型参数
        loss.backward()
        # 参数更新
        optimizer.step()
        # 更新训练损失
        trainLoss += loss.item() * data.size(0)

    ################
    # 测试集运算过程 #
    ################
    model.eval()
    for data, target in testLoader1:
        # 将数据流转换成GPU可用的形式
        if trainOnGpu:
            data, target, dumLabel = data.cuda().type(torch.cuda.FloatTensor), target.cuda().type(torch.cuda.FloatTensor), dumLabel.cuda().type(torch.cuda.FloatTensor)
        # 前向传递: 通过输入数据到网络模型计算输出数据
        _, output = model(data)
        output = model(data).reshape(batchSize, -1)
        target = target.reshape(batchSize, -1)
        # 计算批损失
        if onAEMU:
            AEMU.enqueueOrDequeue(output.detach(), target.detach())
            loss = ContrastiveLoss(output, target, output, target)
        else:
            loss = criterion(output, target, dumLabel)
        # 更新平均测试损失
        testLoss += loss.item() * data.size(0)

    # 计算平均损失
    trainLoss = trainLoss / len(trainLoader1.sampler)
    testLoss = testLoss / len(testLoader1.sampler)

    # 显示训练集与测试集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
        epoch, trainLoss, testLoss))

    # 如果测试集损失函数减少，就保存模型。
    if testLoss <= testLossMin:
        print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(testLossMin, testLoss))
        torch.save(model.state_dict(), 'SAE.pt')
        testLossMin = testLoss

model.load_state_dict(torch.load('SAE.pt'))
model.eval()

####################
# 生成SAE Spec.特征 #
####################
dt = {}
SAESpecs = []
labels = []

for data, label in trainLoader3:
    # 将数据流转换成GPU可用的形式
    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)
    # 前向传递: 通过输入数据到网络模型计算输出数据
    SAESpec, _ = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/trainData3", "wb")
dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

for data, label in trainLoader4:
    # 将数据流转换成GPU可用的形式
    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)
    # 前向传递: 通过输入数据到网络模型计算输出数据
    SAESpec, _ = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/trainData4", "wb")
dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

for data, label in testLoader3:
    # 将数据流转换成GPU可用的形式
    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)
    # 前向传递: 通过输入数据到网络模型计算输出数据
    SAESpec, _ = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/testData3", "wb")
dump(dt, trainFile)
trainFile.close()

dt = {}
SAESpecs = []
labels = []

for data, label in testLoader4:
    # 将数据流转换成GPU可用的形式
    if trainOnGpu:
        data = data.cuda().type(torch.cuda.FloatTensor)
    # 前向传递: 通过输入数据到网络模型计算输出数据
    SAESpec, _ = model(data)
    SAESpec = SAESpec.reshape(SAESpec.shape[0], 1, -1).tolist()
    label = label.tolist()
    SAESpecs.extend(SAESpec)
    labels.extend(label)

dt["labels"] = labels
dt["SAESpecs"] = np.array(SAESpecs)
print(dt["SAESpecs"].shape)
trainFile = open("./data/testData4", "wb")
dump(dt, trainFile)
trainFile.close()




