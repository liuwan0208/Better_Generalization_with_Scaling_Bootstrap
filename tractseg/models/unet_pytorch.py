
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d


class UNet_Pytorch(torch.nn.Module):
    def __init__(self, img_size, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(UNet_Pytorch, self).__init__()

        self.img_size = img_size
        self.dropout = dropout

        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.n_filt = n_filt

        self.contr_1_1 = conv2d(n_input_channels, n_filt, batchnorm=batchnorm)
        self.contr_1_2 = conv2d(n_filt, n_filt, batchnorm=batchnorm)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2, batchnorm=batchnorm)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2, batchnorm=batchnorm)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4, batchnorm=batchnorm)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4, batchnorm=batchnorm)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8, batchnorm=batchnorm)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)#encoder finish

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16, batchnorm=batchnorm)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16, batchnorm=batchnorm)##bottleneck finish

        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        # self.deconv_1 = nn.Upsample(scale_factor=2)  # does only upscale width and height; Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8, batchnorm=batchnorm)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1, batchnorm=batchnorm)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1, batchnorm=batchnorm)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        pool_4 = self.pool_4(contr_4_2)

        if self.dropout:
            pool_4 = self.dropout(pool_4)##encoder

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        deconv_2 = self.deconv_2(expand_1_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)
        return conv_5



    def flops_conv3_relu(self, in_size, in_channel, out_channel, kernel_size=3, stride=1):
        out_size = in_size
        ## conv3*3 + bias
        flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
        ## relu
        flops += (out_size**2)*out_channel
        return flops
    def flops_conv1(self,in_size, in_channel, out_channel, kernel_size=1, stride=1):
        out_size = in_size
        ## conv1*1 + bias
        flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
        return flops
    def flops_deconv2_relu(self, in_size, in_channel, out_channel, kernel_size=2, stride=2):
        out_size = in_size*kernel_size
        ## deconv2*2 + bias
        flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
        ## relu
        flops += (out_size**2)*out_channel
        return flops
    def flops_maxpool(self, in_size, in_channel, kernel_size=2, stride=2):
        ## number of elements
        flops = (in_size**2)*in_channel
        return flops

    def flops(self, downsample=4):
        flops = 0
        ## encoder (4 block: 2conv+maxpool)
        for block_index in range(downsample):
            if block_index == 0:
                flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
                                               in_channel=self.in_channel,
                                               out_channel=self.n_filt * (2 ** block_index))
            else:
                flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
                                               in_channel=self.n_filt * (2 ** (block_index - 1)),
                                               out_channel=self.n_filt * (2 ** block_index))
            flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
                                           in_channel=self.n_filt * (2 ** block_index),
                                           out_channel=self.n_filt * (2 ** block_index))
            flops += self.flops_maxpool(in_size=self.img_size / (2 ** block_index),
                                  in_channel=self.n_filt * (2 ** block_index))
        ## bottleneck
        flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** downsample),
                                       in_channel=self.n_filt * (2 ** (downsample-1)),
                                       out_channel=self.n_filt * (2 ** downsample))
        flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** downsample),
                                       in_channel=self.n_filt * (2 ** downsample),
                                       out_channel=self.n_filt * (2 ** downsample))
        flops += self.flops_deconv2_relu(in_size=self.img_size / (2 ** downsample),
                                       in_channel=self.n_filt * (2 ** downsample),
                                       out_channel=self.n_filt * (2 ** downsample))
        ## decoder
        for block in range(downsample):
            block_index = downsample-1-block
            flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
                                           in_channel=self.n_filt * (2**block_index + 2**(block_index+1)),
                                           out_channel=self.n_filt * (2 ** block_index))
            flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
                                           in_channel=self.n_filt * (2 ** block_index),
                                           out_channel=self.n_filt * (2 ** block_index))
            if block_index>0:
                flops += self.flops_deconv2_relu(in_size=self.img_size / (2 ** block_index),
                                                 in_channel=self.n_filt * (2 ** block_index),
                                                 out_channel=self.n_filt * (2 ** block_index))
            else:
                flops += self.flops_conv1(in_size = self.img_size, in_channel = self.n_filt, out_channel = self.n_classes)
        return flops
