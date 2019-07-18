import torch.nn as nn
import torch.nn.functional as F


class ITN2D(nn.Module):

    def __init__(self, input_channels):
        super(ITN2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(input_channels, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(4, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv22 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, bias=use_bias)
        self.conv13 = nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv15 = nn.Conv2d(2, input_channels, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x2 = self.up2(x3) + x2
        x2 = F.relu(self.conv22(x2))
        x1 = self.up1(x2) + x1
        x1 = F.relu(self.conv13(x1))
        x1 = F.relu(self.conv14(x1))
        x = self.conv15(x1)

        return x


class ITN3D(nn.Module):

    def __init__(self, input_channels):
        super(ITN3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(input_channels, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv3d(2, 4, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv3d(4, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(8, 16, kernel_size=2, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv22 = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2, bias=use_bias)
        self.conv13 = nn.Conv3d(4, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv3d(2, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv15 = nn.Conv3d(2, input_channels, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x2 = self.up2(x3) + x2
        x2 = F.relu(self.conv22(x2))
        x1 = self.up1(x2) + x1
        x1 = F.relu(self.conv13(x1))
        x1 = F.relu(self.conv14(x1))
        x = self.conv15(x1)

        return x
