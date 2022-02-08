# -*- coding: utf-8 -*-
import torch
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
lst = []
lst.extend(nums[-3:])
print(lst)

a = torch.randn(100, 128, requires_grad=True)
print(a.shape)
b = torch.randn(100, 128, requires_grad=True)
print(b.shape)
y = torch.ones(100)
print(y)
x = [[1, 2], [3, 4]]
print(len(x))