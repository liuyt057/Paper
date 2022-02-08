# -*- coding: utf-8 -*-
import torch
import numpy as np
from pickle import *


class GetLoaderSAE(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data, target):
        self.data = data
        self.label = target

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)