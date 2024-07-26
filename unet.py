import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_double = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_double(x)

class Down_sample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class Up_sample(nn.Module):
    def __init__(self, in_channels):
        super(Up_sample, self,).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Convblock(in_channels, 32)
        self.down1 = Down_sample()
        self.conv2 = Convblock(32, 64)
        self.down2 = Down_sample()
        self.conv3 = Convblock(64, 128)
        self.down3 = Down_sample()
        self.conv4 = Convblock(128, 256)
        self.down4 = Down_sample()
        self.conv5 = Convblock(256, 512)

        self.up1 = Up_sample(512)
        self.conv6 = Convblock(512, 256)
        self.up2 = Up_sample(256)
        self.conv7 = Convblock(256, 128)
        self.up3 = Up_sample(128)
        self.conv8 = Convblock(128, 64)
        self.up4 = Up_sample(64)
        self.conv9 = Convblock(64, 32)

        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(self.down1(out1))
        out3 = self.conv3(self.down2(out2))
        out4 = self.conv4(self.down3(out3))
        out5 = self.conv5(self.down4(out4))

        out6 = self.conv6(torch.cat([self.up1(out5), out4], 1))
        out7 = self.conv7(torch.cat([self.up2(out6), out3], 1))
        out8 = self.conv8(torch.cat([self.up3(out7), out2], 1))
        out9 = self.conv9(torch.cat([self.up4(out8), out1], 1))
        out = self.conv10(out9)
        return out

if __name__ == '__main__':
    model = Unet(3, 3)
    data = torch.rand((32, 3, 256, 256))
    output = model(data)
    print(output.size())









