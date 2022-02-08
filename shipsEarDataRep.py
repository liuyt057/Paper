# -*- coding: utf-8 -*-
from pickle import *
import numpy as np
from librosa import load
from fbank import MyFBank
from gbank import MyGBank
import os

inputName = "D:/Research/ShipsEar/data/train_wavdata/"
outputName = "C:/Users/24552/C_C++Projects/Paper/data/"

# 训练集
dt1 = {}
labels1 = []
FBanks1 = []
GBanks1 = []

# 测试集
dt2 = {}
labels2 = []
FBanks2 = []
GBanks2 = []

for i in range(0, 4):
    labels = []
    FBanks = []
    GBanks = []
    fName = inputName + str(i)
    lst = os.listdir(fName)
    for j in range(len(lst)):
        fName = inputName + str(i) + "/" + lst[j]
        rawSignal, sampleRate = load(fName)
        cnt = 1
        while int(0.34 * sampleRate * cnt) < len(rawSignal):
            signal = rawSignal[int(0.34 * sampleRate * (cnt - 1)): int(0.34 * sampleRate * cnt)]
            cnt += 1
            FBank = MyFBank(signal, fs=sampleRate, nfilts=32)
            GBank = MyGBank(signal, fs=sampleRate, nfilts=32)
            FBanks.append(FBank)
            GBanks.append(GBank)
            labels.append(i)
    num = len(labels)
    num1 = round(num // 8 * 0.7 * 8)
    num2 = round(num // 8 * 0.3 * 8)
    FBanks1.extend(FBanks[:num1])
    FBanks2.extend(FBanks[-1 * num2:])
    GBanks1.extend(GBanks[:num1])
    GBanks2.extend(GBanks[-1 * num2:])
    labels1.extend(labels[:num1])
    labels2.extend(labels[-1 * num2:])

if __name__ == "__main__":
    dt1["labels"] = labels1
    dt1["FBanks"] = np.array(FBanks1)
    print(dt1["FBanks"].shape)
    dt1["GBanks"] = np.array(GBanks1)
    print(dt1["GBanks"].shape)

    dt2["labels"] = labels2
    dt2["FBanks"] = np.array(FBanks2)
    print(dt2["FBanks"].shape)
    dt2["GBanks"] = np.array(GBanks2)
    print(dt2["GBanks"].shape)

    trainDataFileName = outputName + "trainData2"
    trainData2 = open(trainDataFileName, "wb")
    dump(dt1, trainData2)
    trainData2.close()

    testDataFileName = outputName + "testData2"
    testData2 = open(testDataFileName, "wb")
    dump(dt2, testData2)
    testData2.close()

