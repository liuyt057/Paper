# -*- coding: utf-8 -*-
import numpy as np
import torch


def GaussianNoise(data, mean=0, var=0.001):
    """
    添加高斯噪声
    Args:
        data: 原始数据
        mean: 均值
        var: 方差，其值越大，噪声越大
    Returns:
        添加过高斯噪声的数据
    """
    noise = np.random.normal(mean, var ** 0.5, data.shape)
    out = data + noise
    if out.min() < 0:
        lowClip = -1.0
    else:
        lowClip = 0.0
    out = np.clip(out, lowClip, 1.0)
    return out


if __name__ == "__main__":
    x = torch.rand(32 * 8, 32).reshape(-1, 1, 32, 32)
    print(x.shape)
    # print(x)
    y = GaussianNoise(x)
    print(y.shape)
    print(y)
