'''
FPN module see the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
class SE_Block(nn.Module):
    def __init__(self, c, r=16,mid=0):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if mid==0:
            mid = c//r
        self.excitation = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        #print(f'[se]1:{y.size()},bs:{bs},c:{c}')
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1,groups=1,usese=False,fusion='rc'):
        super(BasicBlock, self).__init__()
        self.fusion = fusion
        self.groups=groups
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), padding=(1,0), stride=stride, groups=groups,bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3,1), padding=(1,0), groups=groups,bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.usese=usese
        self.relu = nn.LeakyReLU(inplace=True)
        if 'se' in fusion:
            self.se = SE_Block(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            ds_group = 1 if ('rc' in fusion) else groups
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=ds_group),#groups=groups,
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if 'cs' in self.fusion:
            channel_shuffle(out,self.groups)
        out = self.bn2(self.conv2(out))
        if self.usese:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class GRNet(nn.Module):
    def __init__(self, num_blocks=[2,2,2,2],groups=1,expand=[2,2,4,8,16],num_class=100,usels=True,fusion='rc',f_ideal=None,f_idx=None):
        super(GRNet, self).__init__()
        
        block = BasicBlock
        self.in_planes = groups
        self.usefpn = False
        self.groups = groups
        self.usels=usels
        self.fusion=fusion
        self.usese=False
        self.quad=False
        init_channel_repeat=1
        if 'se' in self.fusion:
            self.usese = True
        if 'fpn' in self.fusion:
            self.usefpn = True
        band_width = 128
        symbol_len = 1024
        self.f_ideal = f_ideal
        self.f_ideal = torch.complex(torch.Tensor(f_ideal[0,:,0]),torch.Tensor(f_ideal[0,:,1])).cuda()/100.0
        self.f_idx = f_idx
        
        print(expand)
        print(f'[grnet] 0:{self.in_planes*init_channel_repeat},1:{self.in_planes*expand[0]},2:{self.in_planes*expand[1]},3:{self.in_planes*expand[2]},4:{self.in_planes*expand[3]},5:{self.in_planes*expand[4]}')
        self.first_conv = nn.Sequential(
                nn.Conv2d(self.in_planes*init_channel_repeat, self.in_planes*expand[0], kernel_size=(7,2), stride=1, padding=0, bias=False,groups=self.groups),
                nn.BatchNorm2d(self.in_planes*expand[0]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3,1), stride=2, padding=0)
            )
        # Bottom-up layers
        self.layer1 = self._make_layer(block, self.in_planes*expand[0], self.in_planes*expand[1], num_blocks[0], stride=1, groups=self.groups)
        self.layer2 = self._make_layer(block, self.in_planes*expand[1], self.in_planes*expand[2], num_blocks[1], stride=1, groups=self.groups)
        self.layer3 = self._make_layer(block, self.in_planes*expand[2], self.in_planes*expand[3], num_blocks[2], stride=2, groups=self.groups)
        self.layer4 = self._make_layer(block, self.in_planes*expand[3], self.in_planes*expand[4], num_blocks[3], stride=2, groups=self.groups)
        print(f'[seresnet]expand:{expand}')
        print(f'[seresnet] out channels:{self.in_planes*expand[0]},{self.in_planes*expand[1]},{self.in_planes*expand[2]},{self.in_planes*expand[3]},{self.in_planes*expand[4]}')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes*expand[4], num_class)
        if self.usefpn:
            # Top layer
            print('[seresnet]init with fpn')
            self.toplayer = nn.Conv2d(self.in_planes*expand[4], self.in_planes*2, kernel_size=1, stride=1, padding=0)  # Reduce channels

            # Smooth layers
            self.smooth1 = nn.Conv2d(self.in_planes*2, self.in_planes*2, kernel_size=(3,1), stride=1, padding=1)
            self.smooth2 = nn.Conv2d(self.in_planes*2, self.in_planes*2, kernel_size=(3,1), stride=1, padding=1)
            self.smooth3 = nn.Conv2d(self.in_planes*2, self.in_planes*2, kernel_size=(3,1), stride=1, padding=1)

            # Lateral layers
            self.latlayer1 = nn.Conv2d(self.in_planes*expand[3], self.in_planes*2, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( self.in_planes*expand[2], self.in_planes*2, kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d( self.in_planes*expand[1], self.in_planes*2, kernel_size=1, stride=1, padding=0)

            # classifier
            self.fpn_fc4 = nn.Linear(self.in_planes*2, num_class)
            self.fpn_fc3 = nn.Linear(self.in_planes*2, num_class)
            self.fpn_fc2 = nn.Linear(self.in_planes*2, num_class)
            self.fpn_fc5 = nn.Linear(self.in_planes*2, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride, groups):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, out_planes, stride,groups,self.usese,self.fusion))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x, r=None,snr=-100):
        if self.usels:
            B,C,H,W = x.size()
            x = x.view(B,1,C*H,W)
            sym_num = C*H//1024
            if sym_num >8:
                repeatnum = 8
            else:
                repeatnum = sym_num
            x = ls_equ(x,self.f_ideal,self.f_idx,repeatnum,sym_num,1024)
            x = x.view(B,C,H,W)
        outs={}
        c1 = self.first_conv(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        if self.usefpn:
            # Top-down
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            #classifier
            c5 = self.avgpool(c5)
            c5 = torch.flatten(c5, 1)
            p5 = self.avgpool(p5)
            p5 = torch.flatten(p5, 1)
            p5 = self.fpn_fc5(p5)
            p4 = self.avgpool(p4)
            p4 = torch.flatten(p4, 1)
            p4 = self.fpn_fc4(p4)
            p3 = self.avgpool(p3)
            p3 = torch.flatten(p3, 1)
            p3 = self.fpn_fc3(p3)
            p2 = self.avgpool(p2)
            p2 = torch.flatten(p2, 1)
            p2 = self.fpn_fc2(p2)
            outs['fpn1'] = p2
            outs['fpn2'] = p3
            outs['fpn3'] = p4
            outs['fpn4'] = p5
            outs['final'] = self.fc(c5)
            return outs
        else:
            c5 = self.avgpool(c5)
            c5 = torch.flatten(c5, 1)
            outs['final'] = self.fc(c5)
            return outs


from fvcore.nn import FlopCountAnalysis
if __name__ == '__main__':
    flops_list = []
    slicelen = 64
    for datalen_idx,w in zip([1,5,6],[4,4,4]):#3,4,6,8,11,16,24,32
        
        c = datalen_idx*1024//slicelen
        model = GRNet(expand=[w,w,w*2,w*4,w*8],groups=c,num_class=25,usels=False,fusion=['']).cuda()#'se'
        x = torch.randn((1,c,slicelen,2)).cuda()
        outs=model(x)
        flops = FlopCountAnalysis(model, x)
        flops = flops.total()/1.0e6
        flops_list.append(flops)
        print(f'--------------------')
        print(f'datalen_idx:{datalen_idx},width{w}')
        print('FLOPs:',flops,'M')
    print(slicelen)
    print('flops M:',flops_list)
