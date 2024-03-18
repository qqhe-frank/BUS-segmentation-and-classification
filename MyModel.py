import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from Multi_Scale_Module import PAFEM, GPM, FoldConv_aspp, HMU, SIM
from DCN import DeformConv2d


class CA_Module(nn.Module):
    def __init__(self, in_channel):
        super(CA_Module, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Sequential(nn.Linear(2 * in_channel, in_channel // 16),
                                    nn.ReLU(),
                                    nn.Linear(in_channel // 16, in_channel),
                                    nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        p1 = self.avgpool(x)
        p2 = self.maxpool(x)
        p = torch.flatten(torch.cat([p1, p2], dim=1), 1)
        po = self.linear(p).view(b, c, 1, 1)
        out = nn.ReLU()(x * po)
        return out


class classfiler_1(nn.Module):
    def __init__(self):
        super(classfiler_1, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Sequential(nn.Linear(1280, 640), nn.ReLU(), nn.Linear(640, 2))
        self.ca1 = CA_Module(256)
        self.ca2 = CA_Module(512)
        self.ca3 = CA_Module(512)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2, input3):
        c1 = self.ca1(input1)
        c2 = self.up(self.ca2(input2))
        c3 = self.ca3(input3)

        all = self.avgpool(torch.cat([c1, c2, c3], 1))
        c5 = torch.flatten(all, 1)
        out = self.fc(c5)
        return out


class GSA_Module(nn.Module):
    def __init__(self, in_channel):
        super(GSA_Module, self).__init__()

        self.output = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))

        self.gate = nn.Sequential(nn.Conv2d(in_channel, in_channel // 8, 3, 1),
                                  nn.BatchNorm2d(in_channel // 8), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel // 8, 1, kernel_size=3, padding=1))

        self.sa = SpatialAttentionModule()

    def forward(self, x):
        b, c, _, _ = x.size()

        sal = self.sa(x)*x

        g1 = self.gate(x)
        g2 = F.adaptive_avg_pool2d(torch.sigmoid(g1), 1)
        g3 = self.output(g2.repeat(1, c, 1, 1) * sal)  #.unsqueeze(1)
        return g3


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = nn.ReLU(self.sigmoid(self.conv2d(out)) * x)
        return out


class GSA(nn.Module):
    def __init__(self, in_channel):
        super(GSA, self).__init__()

        self.output = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                                    nn.GroupNorm(in_channel // 4, in_channel), nn.ReLU())

        self.gate = nn.Sequential(nn.Conv2d(in_channel, in_channel // 8, kernel_size=3, padding=1),
                                  nn.GroupNorm(in_channel // 16, in_channel // 8), nn.ReLU(),
                                  nn.Conv2d(in_channel // 8, 2, kernel_size=3, padding=1))

    def forward(self, x):
        b, c, _, _ = x.size()
        g1 = self.gate(x)
        g2 = F.adaptive_avg_pool2d(torch.sigmoid(g1), 1)
        g3 = F.adaptive_max_pool2d(torch.sigmoid(g1), 1)
        output = self.output(g2[:, 0, :, :].unsqueeze(1).repeat(1, c, 1, 1) * x + \
                             g3[:, 1, :, :].unsqueeze(1).repeat(1, c, 1, 1) * x)

        return output


class DSAModule(nn.Module):
    def __init__(self, in_ch):
        super(DSAModule, self).__init__()

        self.dcn1 = nn.Sequential(DeformConv2d(in_ch, in_ch, 3, padding=1, modulation=True),
                                  nn.GroupNorm(in_ch // 4, in_ch), nn.ReLU())

        self.dcn2 = nn.Sequential(DeformConv2d(2 * in_ch, in_ch, 3, padding=1, modulation=True),
                                  nn.GroupNorm(in_ch // 4, in_ch), nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        c1 = self.dcn1(x)
        # torch.mean(x,1).unsqueeze(1)
        avgout = self.conv1(torch.mean(c1, 1).unsqueeze(1))
        o1 = avgout * c1
        # torch.max(x, dim=1, keepdim=True)
        maxout = self.conv2(torch.max(c1, 1)[0].unsqueeze(1))
        o2 = maxout * c1

        out = torch.cat([o1, o2], dim=1)
        out = self.dcn2(out)
        return out



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64*128*128
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 64*64*64
        self.layer2 = self.base_layers[5]  # 128  *32*32
        self.layer3 = self.base_layers[6]  # 256 *16*16
        self.layer4 = self.base_layers[7]  # 512 *8*8

        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.GroupNorm(16, 512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1), nn.GroupNorm(16, 512),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(8, 256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(8, 256),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=True))

        self.up1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=2))

        self.up2 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=2))

        self.up3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=2))

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=2))

        self.out = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                 nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(4, 128),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 1, 1))

        self.classfer = classfiler_1()

        self.g1 = GSA(512)
        self.g2 = GSA(256)
        self.g3 = GSA(128)
        self.g4 = GSA(64)
        self.g5 = GSA(64)

        self.sa1 = DSAModule(512)
        self.sa2 = DSAModule(256)


    def forward(self, x):
        layer0 = self.layer0(x)  # 64
        layer1 = self.layer1(layer0)  # 64
        layer2 = self.layer2(layer1)  # 128
        layer3 = self.layer3(layer2)  # 256
        layer4 = self.layer4(layer3)  # 512

        up1 = self.up1(self.g1(layer4))
        ffm1 = torch.cat([self.g2(layer3), up1], dim=1)
        ffm1 = self.sa1(self.conv2(ffm1))

        classfiler = self.classfer(layer3, layer4, ffm1)

        up2 = self.up2(ffm1)
        ffm2 = torch.cat([self.g3(layer2), up2], dim=1)
        ffm2 = self.sa2(self.conv3(ffm2))

        up3 = self.up3(ffm2)
        ffm3 = torch.cat([self.g4(layer1), up3], dim=1)
        ffm3 = self.conv4(ffm3)

        up4 = self.up4(ffm3)
        ffm4 = torch.cat([self.g5(layer0), up4], dim=1)
        out = self.conv5(ffm4)
        out = self.out(out)

        return classfiler, out



if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 3, 256, 256)).cuda()
    model = MyModel().cuda()
    c, s = model(x)
    print('Output s shape:', s.shape)
    print('Output c shape:', c.shape)
