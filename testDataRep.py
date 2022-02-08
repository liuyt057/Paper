# -*- coding: utf-8 -*-
from pickle import *
import numpy as np
from scipy.io import wavfile
from fbank import MyFBank
from gbank import MyGBank
import os

inputName = "D:/Research/水下数据集/data/test_wavdata/"
outputName = "C:/Users/24552/C_C++Projects/Paper/data/"

dt = {}
labels = []
FBanks = []
GBanks = []
for i in range(0, 6):
    fName = inputName + str(i)
    lst = os.listdir(fName)
    for j in range(len(lst)):
        fName = inputName + str(i) + "/" + lst[j]
        sampleRate, rawSignal = wavfile.read(fName)
        cnt = 1
        while int(0.34 * sampleRate * cnt) < len(rawSignal):
            signal = rawSignal[int(0.34 * sampleRate * (cnt - 1)): int(0.34 * sampleRate * cnt)]
            cnt += 1
            FBank = MyFBank(signal, fs=sampleRate, nfilts=32)
            GBank = MyGBank(signal, fs=sampleRate, nfilts=32)
            FBanks.append(FBank)
            GBanks.append(GBank)
            labels.append(i)
if __name__ == "__main__":
    dt["labels"] = labels
    dt["FBanks"] = np.array(FBanks)
    print(dt["FBanks"].shape)
    dt["GBanks"] = np.array(GBanks)
    print(dt["GBanks"].shape)
    testDataFileName = outputName + "testData1"
    testData1 = open(testDataFileName, "wb")
    dump(dt, testData1)
    testData1.close()

