import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network import SegmentationNetwork


#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  resnet18 :  BasicBlock, [2, 2, 2, 2]
#  resnet34 :  BasicBlock, [3, 4, 6, 3]
#  resnet50 :  Bottleneck  [3, 4, 6, 3]
#

# https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# https://github.com/ternaus/TernausNetV2
# https://github.com/neptune-ml/open-solution-salt-detection
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution


# 3
#  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
#  https://pytorch.org/docs/stable/torchvision/models.html


class ConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels, convT_channels=0):
        super(Decoder, self).__init__()
        self.convT_channels = convT_channels
        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)
        self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        if self.convT_channels:
            x = self.convT(x)
        else:
            x = F.upsample(x, scale_factor=2, mode='bilinear',
                           align_corners=True)  # False
        x = torch.cat([x, skip], 1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class UNetResNet34(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34, self).__init__(**kwargs)
        self.resnet = ResNet.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 256, 512, 256)
        self.decoder4 = Decoder(256 + 128, 256, 128)
        self.decoder3 = Decoder(128 + 64, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 32)
        # self.decoder1 = Decoder(128, 128, 32)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x)
        p = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2(p)   # ; print('e2',e2.size())
        e3 = self.encoder3(e2)  # ; print('e3',e3.size())
        e4 = self.encoder4(e3)  # ; print('e4',e4.size())
        e5 = self.encoder5(e4)  # ; print('e5',e5.size())

        # f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        # f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        # f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5)  # ; print('center',f.size())

        f = self.decoder5(f, e4)  # ; print('d5',f.size())
        f = self.decoder4(f, e3)  # ; print('d4',f.size())
        f = self.decoder3(f, e2)  # ; print('d3',f.size())
        f = self.decoder2(f, x)  # ; print('d2',f.size())
        # f = self.decoder1(f)  # ; print('d1',f.size())

        # f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit


class UNetResNet34_convT(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34_convT, self).__init__(**kwargs)
        self.resnet = ResNet.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 512, 512, 128, convT_channels=1024)
        self.decoder4 = Decoder(64 + 256, 256, 128, convT_channels=128)
        self.decoder3 = Decoder(64 + 128, 128, 128, convT_channels=128)
        self.decoder2 = Decoder(64 + 64, 64, 128, convT_channels=128)
        self.decoder1 = Decoder(64 + 64, 64, 32, convT_channels=128)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f = self.center(e4)  # 4

        f = self.decoder5(f, e4)  # 8
        f = self.decoder4(f, e3)  # 16
        f = self.decoder3(f, e2)  # 32
        f = self.decoder2(f, e1)  # 64
        f = self.decoder1(f, x)

        # f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit


class UNetResNet34_convT_hyper(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34_convT_hyper, self).__init__(**kwargs)
        self.resnet = ResNet.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = Decoder(512 + 256, 512, 512, convT_channels=1024)
        self.decoder3 = Decoder(256 + 128, 256, 256, convT_channels=512)
        self.decoder2 = Decoder(128 + 64, 128, 128, convT_channels=256)
        self.decoder1 = Decoder(64 + 64, 64, 32, convT_channels=128)

        self.logit = nn.Sequential(
            nn.Conv2d(32 + 128 + 256 + 512, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 8

        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)  # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
        ], 1)
        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit

##########################################################################


def run_check_net():
    import numpy as np
    import torch.optim as optim

    batch_size = 8
    C, H, W = 1, 128, 128

    input = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(2, (batch_size, C, H, W)).astype(np.float32)

    # ------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()

    # ---
    net = UNetResNet34().cuda()
    net.set_mode('train')
    # print(net)
    # exit(0)

    # net.load_pretrain('/root/share/project/kaggle/tgs/data/model/resnet50-19c8e357.pth')

    logit = net(input)
    loss = net.criterion(logit, truth)
    dice = net.metric(logit, truth)

    print('loss : %0.8f' % loss.item())
    print('dice : %0.8f' % dice.item())
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    i = 0
    optimizer.zero_grad()
    while i <= 500:

        logit = net(input)
        loss = net.criterion(logit, truth)
        dice = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f' %
                  (i, loss.item(), dice.item()))
        i = i + 1


##########################################################################
if __name__ == '__main__':
    import os

    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print('sucessful!')
