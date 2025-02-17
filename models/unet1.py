import torch
import torch.nn as nn


# Define the Basic Unet
class Unet(nn.Module):

    def __init__(self, in_channels, out_channels, nconvs=1):

        super().__init__()

        # Encoder layers
        self.enc1 = ConvBlock(in_channels, 64, n_convs=1)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128, n_convs=1)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256, n_convs=1)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(256, 512, n_convs=1)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc5 = ConvBlock(512, 512, n_convs=1)

        # Decoder layers
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.skip4 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.dec4 = ConvBlock(512, 256, n_convs=1)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.skip3 = nn.Conv2d(2 * 256, 256, kernel_size=1, padding=0)
        self.dec3 = ConvBlock(256, 128, n_convs=1)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.skip2 = nn.Conv2d(2 * 128, 128, kernel_size=1, padding=0)
        self.dec2 = ConvBlock(128, 64, n_convs=1)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.skip1 = nn.Conv2d(2 * 64, 64, kernel_size=1, padding=0)
        self.dec1 = ConvBlock(64, 64, n_convs=1)
        self.out = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=1, padding=0))
        self.dropout = nn.Dropout(0.5)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, x):

        # Apply encoder layers
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        enc5 = self.enc5(self.down4(enc4))

        # Apply decoder layers and skip connections
        up4 = self.up4(enc5)
        skip4 = self.skip4(torch.cat((up4, enc4), dim=1))
        dec4 = self.dec4(skip4)
        up3 = self.up3(dec4)
        skip3 = self.skip3(torch.cat((up3, enc3), dim=1))
        dec3 = self.dec3(skip3)
        up2 = self.up2(dec3)
        skip2 = self.skip2(torch.cat((up2, enc2), dim=1))
        dec2 = self.dec2(skip2)
        up1 = self.up1(dec2)
        skip1 = self.skip1(torch.cat((up1, enc1), dim=1))
        dec1 = self.dec1(skip1)

        # Output colors
        dec1 = self.dropout(dec1)
        out = self.out_activation(self.out(dec1))
        return out


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, n_convs=2):
        if n_convs < 1:
            raise ValueError('n_convs must be >= 1')
        layers = []
        for i in range(n_convs):
            conv2d = nn.Conv2d(in_channels if i == 0 else out_channels,
                               out_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]

        super().__init__(*layers)



