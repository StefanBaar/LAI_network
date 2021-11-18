import torch
from torch import nn
import torch.nn.functional as F

class large_UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convolution1 = self.contract_block(in_channels, 32, 7, 3)
        self.convolution2 = self.contract_block(32 , 64 , 3, 1)
        self.convolution3 = self.contract_block(64 , 128, 3, 1)
        self.convolution4 = self.contract_block(128, 256, 3, 1)
        self.convolution5 = self.contract_block(256, 512, 3, 1)

        self.upconvolution5 = self.expand_block(512, 256, 3, 1)
        self.upconvolution4 = self.expand_block(512, 128, 3, 1)
        self.upconvolution3 = self.expand_block(256,  64, 3, 1)
        self.upconvolution2 = self.expand_block(128,  32, 3, 1)
        self.upconvolution1 = self.expand_block(64 , out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        convolution1 = self.convolution1(x)
        convolution2 = self.convolution2(convolution1)
        convolution3 = self.convolution3(convolution2)
        convolution4 = self.convolution4(convolution3)
        convolution5 = self.convolution5(convolution4)

        # upsampling
        upconvolution5 = self.upconvolution5(convolution5)
        upconvolution4 = self.upconvolution4(torch.cat([upconvolution5, convolution4], 1))
        upconvolution3 = self.upconvolution3(torch.cat([upconvolution4, convolution3], 1))
        upconvolution2 = self.upconvolution2(torch.cat([upconvolution3, convolution2], 1))
        upconvolution1 = self.upconvolution1(torch.cat([upconvolution2, convolution1], 1))

        return upconvolution1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
                            )
        return expand

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convolution1 = self.contract_block(in_channels, 32, 7, 3)
        self.convolution2 = self.contract_block(32, 64, 3, 1)
        self.convolution3 = self.contract_block(64, 128, 3, 1)

        self.upconvolution3 = self.expand_block(128, 64, 3, 1)
        self.upconvolution2 = self.expand_block(64*2, 32, 3, 1)
        self.upconvolution1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        convolution1 = self.convolution1(x)
        convolution2 = self.convolution2(convolution1)
        convolution3 = self.convolution3(convolution2)

        upconvolution3 = self.upconvolution3(convolution3)

        upconvolution2 = self.upconvolution2(torch.cat([upconvolution3, convolution2], 1))
        upconvolution1 = self.upconvolution1(torch.cat([upconvolution2, convolution1], 1))

        return upconvolution1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
                            )
        return expand

class super_large_UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convolution1 = self.contract_block(in_channels, 32, 7, 3)
        self.convolution2 = self.contract_block(32 , 64 , 3, 1)
        self.convolution3 = self.contract_block(64 , 128, 3, 1)
        self.convolution4 = self.contract_block(128, 256, 3, 1)
        self.convolution5 = self.contract_block(256, 512, 3, 1)
        self.convolution6 = self.contract_block(512, 1024, 3, 1)

        self.upconvolution6 = self.expand_block(1024, 512, 3, 1)
        self.upconvolution5 = self.expand_block(1024, 256, 3, 1)
        self.upconvolution4 = self.expand_block(512, 128, 3, 1)
        self.upconvolution3 = self.expand_block(256,  64, 3, 1)
        self.upconvolution2 = self.expand_block(128,  32, 3, 1)
        self.upconvolution1 = self.expand_block(64 , out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        convolution1 = self.convolution1(x)
        convolution2 = self.convolution2(convolution1)
        convolution3 = self.convolution3(convolution2)
        convolution4 = self.convolution4(convolution3)
        convolution5 = self.convolution5(convolution4)
        convolution6 = self.convolution6(convolution5)

        # upsampling
        upconvolution6 = self.upconvolution6(convolution6)
        upconvolution5 = self.upconvolution5(torch.cat([upconvolution6, convolution5], 1))
        upconvolution4 = self.upconvolution4(torch.cat([upconvolution5, convolution4], 1))
        upconvolution3 = self.upconvolution3(torch.cat([upconvolution4, convolution3], 1))
        upconvolution2 = self.upconvolution2(torch.cat([upconvolution3, convolution2], 1))
        upconvolution1 = self.upconvolution1(torch.cat([upconvolution2, convolution1], 1))

        return upconvolution1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
                            )
        return expand
