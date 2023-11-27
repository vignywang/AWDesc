import torch
import torch.nn as nn
import torch.nn.functional as f
from nets.vit.vit_seg_modeling import *
from nets.vit.vit_seg_modeling import PatchTransfomer

class SuperPointNet(nn.Module):

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # detect head
        cPa = self.relu(self.convPa(x))
        logit = self.convPb(cPa)
        prob = self.softmax(logit)[:, :-1, :, :]

        # descriptor head
        cDa = self.relu(self.convDa(x))
        feature = self.convDb(cDa)

        dn = torch.norm(feature, p=2, dim=1, keepdim=True)
        desc = feature.div(dn)

        return logit, desc, prob

class MTLDesc(nn.Module):
    def __init__(self):
        super(MTLDesc, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        #关键点金字塔
        self.heatmap1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #权重初始化
        self.fuse_weight_1.data.fill_(0.1)
        self.fuse_weight_2.data.fill_(0.2)
        self.fuse_weight_3.data.fill_(0.3)
        self.fuse_weight_4.data.fill_(0.4)

        self.scalemap = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.active = f.softplus
        self.conv_avg = nn.Conv2d(128, 384, kernel_size=3, stride=1, padding=1)

        self.transfomer=PatchTransfomer(img_size=64,in_channels=128)
        self.pool_size=64
        self.adapool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.mask=nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1)
        self.conv_des = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv_des_1 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_des_2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv_des_3 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_des_4 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=18, dilation=18)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128
        #top=c4
        # KeyPoint Map
        heatmap1 = self.heatmap1(c1)
        heatmap2 = self.heatmap2(c2)
        heatmap3 = self.heatmap3(c3)
        heatmap4 = self.heatmap3(c4)
        des_size = heatmap1.shape[2:]  # 1/4 HxW
        heatmap2 = f.interpolate(heatmap2, des_size, mode='bilinear')
        heatmap3 = f.interpolate(heatmap3, des_size, mode='bilinear')
        heatmap4 = f.interpolate(heatmap4, des_size, mode='bilinear')
        heatmap = heatmap1 * self.fuse_weight_1 + heatmap2 * self.fuse_weight_2 + heatmap3 * self.fuse_weight_3 + heatmap4 * self.fuse_weight_4

        # Descriptor
        des_size = c3.shape[2:]  # 1/4 HxW
        c1 = f.interpolate(c1, des_size, mode='bilinear')
        c2 = f.interpolate(c2, des_size, mode='bilinear')
        c3 = c3
        c4 = f.interpolate(c4, des_size, mode='bilinear')
        feature = torch.cat((c1, c2, c3, c4), dim=1)

        # attention map
        meanmap = torch.mean(feature, dim=1, keepdim=True)
        attmap = self.scalemap(meanmap)
        attmap = self.active(attmap)

        # Global Context
        top=self.adapool(c4)
        mask=self.relu(self.mask(top))
        avg=self.transfomer(top)
        avg=f.interpolate(avg,des_size,mode='bilinear')
        mask=f.interpolate(mask,des_size,mode='bilinear')
        descriptor = feature
        descriptor = self.conv_des(descriptor)+avg*mask
        descriptor_1 = self.conv_des_1(descriptor)
        descriptor_2 = self.conv_des_2(descriptor)
        descriptor_3 = self.conv_des_3(descriptor)
        descriptor_4 = self.conv_des_4(descriptor)
        descriptor_refine = torch.cat((descriptor_1, descriptor_2, descriptor_3, descriptor_4), dim=1)
        descriptor = descriptor + descriptor_refine
        return heatmap, descriptor,attmap
class Lite16(nn.Module):
    def __init__(self):
        super(Lite16, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, stride=1,padding=1)

        self.conv2a = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.heatmap1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        #self.heatmap = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.scalemap = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.active = f.softplus
        self.descriptor = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # 权重初始化
        self.fuse_weight_1.data.fill_(0.1)
        self.fuse_weight_2.data.fill_(0.2)
        self.fuse_weight_3.data.fill_(0.3)
        self.fuse_weight_4.data.fill_(0.4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x1))
        c1 = torch.cat((x1, c1), dim=1)  # 1

        x2 = self.pool(c1)
        x2 = self.relu(self.conv2a(x2))
        c2 = self.relu(self.conv2b(x2))
        c2 = torch.cat((x2, c2), dim=1)  # 1/2

        x3 = self.pool(c2)
        x3 = self.relu(self.conv3a(x3))
        c3 = self.relu(self.conv3b(x3))
        c3 = torch.cat((x3, c3), dim=1)  # 1/4

        x4 = self.pool(c3)
        x4 = self.relu(self.conv4a(x4))
        c4 = self.relu(self.conv4b(x4))
        c4 = torch.cat((x4, c4), dim=1)# 1/8

        # heatmap = self.heatmap(c1)
        heatmap1 = self.heatmap1(c1)
        heatmap2 = self.heatmap2(c2)
        heatmap3 = self.heatmap3(c3)
        heatmap4 = self.heatmap4(c4)

        des_size = c1.shape[2:]  # 1/4 HxW
        heatmap2 = f.interpolate(heatmap2, des_size, mode='bilinear')
        heatmap3 = f.interpolate(heatmap3, des_size, mode='bilinear')
        heatmap4 = f.interpolate(heatmap4, des_size, mode='bilinear')
        heatmap = heatmap1 * self.fuse_weight_1 + heatmap2 * self.fuse_weight_2 + heatmap3 * self.fuse_weight_3 + heatmap4 * self.fuse_weight_4
        #heatmap = torch.cat((c1,heatmap2,heatmap3,heatmap4),dim=1)
        #heatmap = self.heatmap(heatmap)
        # Descriptor
        des_size = c3.shape[2:]  # 1/4 HxW
        c1 = f.interpolate(c1, des_size, mode='bilinear')
        c2 = f.interpolate(c2, des_size, mode='bilinear')
        c3 = c3
        c4 = f.interpolate(c4, des_size, mode='bilinear')
        features = torch.cat((c1, c2, c3,c4), dim=1)
        descriptor = self.descriptor(features)

        # attention map
        meanmap = torch.mean(features, dim=1, keepdim=True)
        attmap = self.scalemap(meanmap)
        attmap = self.active(attmap)

        return heatmap, descriptor, attmap
class Lite32(nn.Module):
    def __init__(self):
        super(Lite32, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1)

        self.conv2a = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.heatmap1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        #self.heatmap = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.scalemap = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.active = f.softplus
        self.descriptor = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # 权重初始化
        self.fuse_weight_1.data.fill_(0.1)
        self.fuse_weight_2.data.fill_(0.2)
        self.fuse_weight_3.data.fill_(0.3)
        self.fuse_weight_4.data.fill_(0.4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x1))
        c1 = torch.cat((x1, c1), dim=1)  # 1

        x2 = self.pool(c1)
        x2 = self.relu(self.conv2a(x2))
        c2 = self.relu(self.conv2b(x2))
        c2 = torch.cat((x2, c2), dim=1)  # 1/2

        x3 = self.pool(c2)
        x3 = self.relu(self.conv3a(x3))
        c3 = self.relu(self.conv3b(x3))
        c3 = torch.cat((x3, c3), dim=1)  # 1/4

        x4 = self.pool(c3)
        x4 = self.relu(self.conv4a(x4))
        c4 = self.relu(self.conv4b(x4))
        c4 = torch.cat((x4, c4), dim=1)# 1/8

        # heatmap = self.heatmap(c1)
        heatmap1 = self.heatmap1(c1)
        heatmap2 = self.heatmap2(c2)
        heatmap3 = self.heatmap3(c3)
        heatmap4 = self.heatmap4(c4)

        des_size = c1.shape[2:]  # 1/4 HxW
        heatmap2 = f.interpolate(heatmap2, des_size, mode='bilinear')
        heatmap3 = f.interpolate(heatmap3, des_size, mode='bilinear')
        heatmap4 = f.interpolate(heatmap4, des_size, mode='bilinear')
        heatmap = heatmap1 * self.fuse_weight_1 + heatmap2 * self.fuse_weight_2 + heatmap3 * self.fuse_weight_3 + heatmap4 * self.fuse_weight_4
        #heatmap = torch.cat((c1,heatmap2,heatmap3,heatmap4),dim=1)
        #heatmap = self.heatmap(heatmap)
        # Descriptor
        des_size = c3.shape[2:]  # 1/4 HxW
        c1 = f.interpolate(c1, des_size, mode='bilinear')
        c2 = f.interpolate(c2, des_size, mode='bilinear')
        c3 = c3
        c4 = f.interpolate(c4, des_size, mode='bilinear')
        features = torch.cat((c1, c2, c3,c4), dim=1)
        descriptor = self.descriptor(features)

        # attention map
        meanmap = torch.mean(features, dim=1, keepdim=True)
        attmap = self.scalemap(meanmap)
        attmap = self.active(attmap)

        return heatmap, descriptor, attmap
