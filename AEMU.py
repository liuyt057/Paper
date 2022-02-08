# -*- codingL: utf-8 -*-
import torch


class AEMU:
    def __init__(self):
        self.n = 256
        self.feats = torch.zeros(self.n, 1024).cuda()
        self.targets = torch.zeros(self.n, dtype=torch.long).cuda()
        self.ptr = 0

    def isFull(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.isFull:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueueOrDequeue(self, feats, targets):
        qSize = len(targets)
        if self.ptr + qSize > self.n:
            self.feats[-qSize:] = feats
            self.targets[-qSize:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + qSize] = feats
            self.targets[self.ptr: self.ptr + qSize] = targets
            self.ptr += qSize
