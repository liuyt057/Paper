# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import *
from decoderBlock import DecoderBlock


class SAE(nn.Module):
    def __init__(self, BN_enable=True, resNet_pretrain=False):
        super(SAE, self).__init__()
        self.BN_enable = BN_enable
        resNet = resnet50(pretrained=resNet_pretrain)
        filters = [64, 256, 512, 1024, 2048]

        # Encoder部分
        self.firstConv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstBN = resNet.bn1
        self.firstRelu = resNet.relu
        self.firstMaxpool = resNet.maxpool
        self.encoder1 = resNet.layer1
        self.encoder2 = resNet.layer2
        self.encoder3 = resNet.layer3
        self.encoder4 = resNet.layer4

        # Decoder部分
        self.center = DecoderBlock(in_channels=filters[4], mid_channels=filters[4] * 4, out_channels=filters[4],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[4] + filters[3], mid_channels=filters[3] * 4,
                                     out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder4 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.firstRelu(x)
        x_ = self.firstMaxpool(x)

        e1 = self.encoder1(x_)  # filter[1]
        e2 = self.encoder2(e1)  # filter[2]
        e3 = self.encoder3(e2)  # filter[3]
        e4 = self.encoder4(e3)  # filter[4]

        center = self.center(e4)  # filter[4]

        d1 = self.decoder1(torch.cat([center, e3], dim=1))  # filter[3]
        d2 = self.decoder2(torch.cat([d1, e2], dim=1))  # filter[2]
        d3 = self.decoder3(torch.cat([d2, e1], dim=1))  # filter[1]
        d4 = self.decoder4(torch.cat([d3, x], dim=1))  # filter[0]

        return e4, self.final(d4)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(2048, 512)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 128)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(128, 32)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity="relu")
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(32, 8)
        nn.init.kaiming_uniform_(self.hidden4.weight, nonlinearity="relu")
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(8, 2)
        nn.init.kaiming_uniform_(self.hidden5.weight, nonlinearity="relu")
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.hidden6.weight)
        self.act6 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        x = self.hidden5(x)
        x = self.act5(x)
        x = self.hidden6(x)
        y = self.act6(x)
        return y


class MLR(nn.Module):
    def __init__(self):
        super(MLR, self).__init__()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 8)
        self.linear5 = nn.Linear(8, 2)
        self.linear6 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = self.sigmoid(self.linear5(x))
        y = self.sigmoid(self.linear6(x))
        return y


if __name__ == "__main__":
    # model = SAE(BN_enable=True, resNet_pretrain=False)
    # print(model)
    model = MLP()
    x = torch.rand(8, 32 * 64).reshape(8, 1, -1)
    print(x.shape)
    y = model(x)
    print(y.shape)
