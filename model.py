import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(SegNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.dec5 = self.conv_block(512, 512)
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1, idx1 = self.pool(self.enc1(x))
        x2, idx2 = self.pool(self.enc2(x1))
        x3, idx3 = self.pool(self.enc3(x2))
        x4, idx4 = self.pool(self.enc4(x3))
        x5, idx5 = self.pool(self.enc5(x4))

        x = F.max_unpool2d(x5, idx5, kernel_size=2)
        x = self.dec5(x)
        x = F.max_unpool2d(x, idx4, kernel_size=2)
        x = self.dec4(x)
        x = F.max_unpool2d(x, idx3, kernel_size=2)
        x = self.dec3(x)
        x = F.max_unpool2d(x, idx2, kernel_size=2)
        x = self.dec2(x)
        x = F.max_unpool2d(x, idx1, kernel_size=2)
        x = self.dec1(x)
        return x
