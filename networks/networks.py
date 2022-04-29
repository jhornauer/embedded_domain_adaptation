import torch
import torch.nn as nn


class PatchGAN_Discriminator(nn.Module):
    def __init__(self):
        super(PatchGAN_Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.InstanceNorm2d(128))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.InstanceNorm2d(256))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.InstanceNorm2d(512))
        self.conv5 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1))

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Discriminator_Latent_Space_KITTI(nn.Module):
    def __init__(self, resolution):
        super(Discriminator_Latent_Space_KITTI, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(4, 7)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.6),
            nn.InstanceNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(3, 5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.6),
            nn.InstanceNorm2d(128))

        if resolution == 'low':
            self.linear = nn.Sequential(nn.Linear(256, 1))
        elif resolution == 'high':
            self.linear = nn.Sequential(nn.Linear(2048, 1))
        else:
            raise NotImplementedError

    def forward(self, input):
        x = self.conv(input)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Discriminator_Latent_Space_DIML(nn.Module):
    def __init__(self):
        super(Discriminator_Latent_Space_DIML, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.6),
            nn.InstanceNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.6),
            nn.InstanceNorm2d(128))
        self.linear = nn.Sequential(
            nn.Linear(128, 1))

    def forward(self, input):
        x = self.conv(input)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
