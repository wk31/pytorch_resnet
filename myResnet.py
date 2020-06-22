# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 00:26:49 2020

@author: 11597
"""

import torch
import torch.nn as nn
import numpy
from torchvision import transforms

from PIL import Image
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm_layer(in_planes):
    return nn.BatchNorm2d(in_planes)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # print('forward')
        # print(out.shape)
        # print(identity.shape)
        out = out + identity
        out = self.relu(out)
        
        
        return out
    


class BottleNeck(nn.Module):
    # 对输入的卷积核个数进行4倍升维
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
                
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # print('forward')
        # print(out.shape)
        # print(identity.shape)
        out = out + identity
        out = self.relu(out)
        
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layer1 = self._make_layer(block, 64, num_block[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        # print(self.in_planes, planes)
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        

        layers = []
        # 第一次需要下采样 stride需要传入
        layers.append(block(self.in_planes, planes, stride, downsample))  
        self.in_planes = planes * block.expansion
        
        # 后面几次bottleneck不需要，则不传入stride
        for stride in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
            
        return nn.Sequential(*layers)            
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
        

def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2], 1000)

def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3], 1000)

def ResNet50():
    return ResNet(BottleNeck,[3,4,6,3], 1000)

def ResNet101():
    return ResNet(BottleNeck,[3,4,23,3], 1000)

def ResNet152():
    return ResNet(BottleNeck,[3,8,36,3], 1000)



if __name__ == "__main__":
    # model18 = ResNet18()
    # model18.load_state_dict(torch.load('./weights/resnet18-5c106cde.pth'))
    # print(model18)
    
    # model34 = ResNet34()
    # model34.load_state_dict(torch.load('./weights/resnet34-333f7ec4.pth'))
    # print(model34)

    model50 = ResNet50()
    model50.load_state_dict(torch.load('./weights/resnet50-19c8e357.pth'))
    print(model50)

    # model101 = ResNet101()
    # model101.load_state_dict(torch.load('./weights/resnet101-5d3b4d8f.pth'))
    # print(model101)

    # model152 = ResNet152()
    # model152.load_state_dict(torch.load('./weights/resnet152-b121ed2d.pth'))
    # print(model152)

   
    '''
    读取图像,前向传播
    '''
    img = Image.open(r'E:\wk\picture\201810\IMG_0687.JPG')
    img = img.resize((224,224),Image.BILINEAR)
    # img = np.array(img)
    # img = img/255.0
    # img = Image.fromarray(np.uint8(img))
    # img.show()
    transform = transforms.ToTensor()
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor,0)
    print(tensor.shape)
    y = model50.forward(tensor)
    print(torch.argmax(y,1))
    
    
    
    # 标准模型
    import torchvision.models as models
    resnet152 = models.resnet152()
    resnet152.load_state_dict(torch.load('./weights/resnet152-b121ed2d.pth'))
    y1 = resnet152.forward(tensor)
    print(torch.argmax(y1,1))





    
        
        
    
































