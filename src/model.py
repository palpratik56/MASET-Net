import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.py import (DoubleConv, Down, OutConv, UpConv, PAM_Module, ASIB, FAAB, TAAC )

class MASETNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, use_pam=True, 
                 use_asib=True, use_taac=True, use_faab=True):
        super().__init__()

        self.use_asib = use_asib
        self.use_taac = use_taac
        self.use_faab = use_faab
        self.use_pam = use_pam

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outc = OutConv(128, n_classes)

        self.pam  = PAM_Module(512)
        # self.sdpa = ScaledDotProductAttention(512)
        self.faab = FAAB(512)

        self.asib1 = ASIB(256, 512, 256)
        self.asib2 = ASIB(128, 256, 128)
        self.asib3 = ASIB(64, 128, 64)

        self.taac1 = TAAC(x_ch=128, task_ch=256)
        self.taac2 = TAAC(x_ch=64,  task_ch=128)
        self.taac3 = TAAC(x_ch=64,  task_ch=64)


    def forward(self, x):
        feature_maps = {}

         # -------- Encoder --------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
    
        # -------- Bottleneck --------
        if self.use_pam:
            x5 = self.pam(x5)
        if self.use_faab:
            x5 = self.faab(x5)
    
        # -------- Decoder --------
        d1 = self.up1(x5, x4)
        if self.use_asib:
            d1 = self.asib1(d1, x4)
        feature_maps['dec1'] = d1
    
        d2 = self.up2(d1, x3)
        if self.use_asib:
            d2 = self.asib2(d2, x3)
    
        if self.use_taac:
            d2 = self.taac1(d2, d1)
        feature_maps['dec2'] = d2
    
        d3 = self.up3(d2, x2)
        if self.use_asib:
            d3 = self.asib3(d3, x2)
    
        if self.use_taac:
            d3 = self.taac2(d3, d2)
        feature_maps['dec3'] = d3
    
        d4 = self.up4(d3, x1)
        if self.use_taac:
            d4 = self.taac3(d4, d3)
        feature_maps['dec4'] = d4
    
        # -------- Output --------
        out = self.outc(torch.cat([d4, x1], dim=1))
        feature_maps['output'] = out
    
        return out, feature_maps


#Initialize teh model and send it to GPU
model = MasetNet(n_channels=3, n_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Wrap the model for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
    
model.to(device)

if torch.__version__ >= "2.0":
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=False
    )
