import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.py import (SEBlock, DoubleConv, Down, AttentionGate, UpConv, PAM_Module,
PositionEmbeddingLearned, ScaledDotProductAttention)

class MasetNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MasetNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UpConv(1024, 512 // factor, bilinear)
        self.up2 = UpConv(1024, 256 // factor, bilinear)
        self.up3 = UpConv(512, 128 // factor, bilinear)
        self.up4 = UpConv(256, 64, bilinear)

        self.outc = nn.Conv2d(128, n_classes, kernel_size=1)
        '''position encoding'''
        self.pos = PositionEmbeddingLearned(512 // factor)

        '''spatial attention mechanism'''
        self.pam = PAM_Module(512)

        '''self-attention mechanism'''
        self.sdpa = ScaledDotProductAttention(512)

    def forward(self, x):
        x1 = self.inc(x)
        '''Encoder'''
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        '''Bottle neck'''
        '''Setting 1'''
        x5_pam = self.pam(x5)

        '''Setting 2'''
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos

        '''Setting 3'''
        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam

        # Decoder path with feature map capture
        feature_maps = {}

        '''Decoder'''
        x6 = self.up1(x5, x4)
        feature_maps['d1'] = x6
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        x7 = self.up2(x6_cat, x3)
        feature_maps['d2'] = x7
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        x8 = self.up3(x7_cat, x2)
        feature_maps['d3'] = x8
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        x9 = self.up4(x8_cat, x1)
        feature_maps['d4'] = x9
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9 = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9)
        feature_maps['output'] = logits
        return logits, feature_maps
