import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
import functools
import numpy as np
from util import *
# import resnet

'''
UNet architecture
'''
class ConvDoubleBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, is_bn=True):
        super(ConvDoubleBlock, self).__init__()
        if is_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1),
                nn.BatchNorm2d(out_num_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1),
                nn.BatchNorm2d(out_num_ch),
                nn.ReLU(inplace=True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.conv(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvDoubleBlock(in_num_ch, out_num_ch, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, down_num_ch, up_num_ch, out_num_ch, upsample=True):
        super(UpBlock, self).__init__()
        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(up_num_ch, out_num_ch, 3, padding=1)
                )
        else:
            self.up = nn.ConvTranspose2d(up_num_ch, out_num_ch, 3, padding=1, stride=2) # (H-1)*stride-2*padding+kernel_size

        self.conv = ConvDoubleBlock(out_num_ch+down_num_ch, out_num_ch, 3)

    def forward(self, x_down, x_up):
        x_up = self.up(x_up)
        x = torch.cat([x_down, x_up], 1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus'):
        super(UNet, self).__init__()
        self.down_1 = ConvDoubleBlock(in_num_ch, first_num_ch, 3)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch)
        self.up_4 = UpBlock(8*first_num_ch, 16*first_num_ch, 8*first_num_ch)
        self.up_3 = UpBlock(4*first_num_ch, 8*first_num_ch, 4*first_num_ch)
        self.up_2 = UpBlock(2*first_num_ch, 4*first_num_ch, 2*first_num_ch)
        self.up_1 = UpBlock(first_num_ch, 2*first_num_ch, first_num_ch)
        self.output = nn.Conv2d(first_num_ch, out_num_ch, 1)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'linear':
            self.output_act = nn.Linear()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(up_1)
        output_act = self.output_act(output)
        return output_act, {}

'''
GAN Standard architecture
'''
class Conv_BN_Act(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1, activation='lrelu', is_bn=True):
        super(Conv_BN_Act, self).__init__()
        if is_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding),
                nn.BatchNorm2d(out_num_ch)
                )
        else:
            self.conv = nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class Act_Deconv_BN_Concat(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu', upsample=True, is_last=False, is_bn=True):
        super(Act_Deconv_BN_Concat, self).__init__()
        self.is_bn = is_bn
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_down, x_up):
        # pdb.set_trace()
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.is_last == False:
            if self.is_bn:
                x_up = self.bn(x_up)
            x = torch.cat([x_down, x_up], 1)
        else:
            x = x_up
        return x

class Act_Deconv_BN(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu', upsample=True, is_last=False, is_bn=True):
        super(Act_Deconv_BN, self).__init__()
        self.is_bn = is_bn
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_up):
        # pdb.set_trace()
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.is_last == False and self.is_bn == True:
            x = self.bn(x_up)
        else:
            x = x_up
        return x

class GANStandardGenerator(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus'):
        super(GANStandardGenerator, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_6 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_7 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_8 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 1 x 1
        self.up_7 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)     # 512
        self.up_6 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)    #
        self.up_5 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)
        self.up_4 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)
        up_7 = self.up_7(down_7, down_8)
        up_6 = self.up_6(down_6, up_7)
        up_5 = self.up_5(down_5, up_6)
        up_4 = self.up_4(down_4, up_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {}

class GANShortGenerator(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus'):
        super(GANShortGenerator, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {}

class GANShortNoShortCutGenerator(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus'):
        super(GANShortNoShortCutGenerator, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8
        self.up_4 = Act_Deconv_BN(8*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN(8*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN(4*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN(2*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN(first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        up_4 = self.up_4(down_5)
        up_3 = self.up_3(up_4)
        up_2 = self.up_2(up_3)
        up_1 = self.up_1(up_2)
        output = self.output(up_1)
        output_act = self.output_act(output)
        return output_act, {}

class GANShortGeneratorWithSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSpatialAttention, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)


        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}


class GANShortGeneratorWithSplitInputAndSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSplitInputAndSpatialAttention, self).__init__()

        # self.down_1_list = []
        # for i in range(in_num_ch):
        #     self.down_1 = nn.Sequential(
        #         nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
        #         nn.LeakyReLU(0.2, inplace=True)
        #     )
        #     self.down_1_list.append(self.down_1)
        self.down_1_1 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_2 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_comb = nn.Sequential(
            nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        # down_1_list = []
        # for i in range(x.shape[1]):
        #     self.down_1 = self.down_1_list[i].cuda()
        #     down_1 = self.down_1(x[:,i,...].unsqueeze(1))
        #     down_1_list.append(down_1)
        # down_1_list = torch.cat(down_1_list, dim=1)
        down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
        down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
        down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
        down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        down_1_comb = self.down_1_comb(down_1_list)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1_comb, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class GANStandardGeneratorWithSplitInputChannelAttentionOne(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANStandardGeneratorWithSplitInputChannelAttentionOne, self).__init__()

        self.down_1_1 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_2 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_4 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_ca = ChannelAttentionLayer(4*first_num_ch, 4)
        self.down_1_comb = nn.Sequential(
            nn.Conv2d(4*first_num_ch, first_num_ch, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_6 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_7 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch)
        self.down_8 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 1 x 1

        self.up_7 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)     # 512
        self.up_6 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)    #
        self.up_5 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)
        self.up_4 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
        down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
        down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
        down_1_4 = self.down_1_4(x[:,3,...].unsqueeze(1))
        down_1_list = torch.cat([down_1_1, down_1_2, down_1_3, down_1_4], dim=1)
        down_1_ca, _ = self.down_1_ca(down_1_list)
        down_1_comb = self.down_1_comb(down_1_ca)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)
        up_7 = self.up_7(down_7, down_8)
        up_6 = self.up_6(down_6, up_7)
        up_5 = self.up_5(down_5, up_6)
        up_4 = self.up_4(down_4, up_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1_comb, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {}

class GANShortGeneratorWithSymmetrySpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSymmetrySpatialAttention, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SymmetrySpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SymmetrySpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SymmetrySpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SymmetrySpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)


        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class GANShortGeneratorWithSymmetryResidualSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSymmetryResidualSpatialAttention, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SymmetryResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SymmetryResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SymmetryResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SymmetryResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)


        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class GANShortGeneratorWithSymmetryGateResidualSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSymmetryGateResidualSpatialAttention, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)


        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class GANShortGeneratorWithSplitInputAndSymmetryGateResidualSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSplitInputAndSymmetryGateResidualSpatialAttention, self).__init__()
        self.down_1_1 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_2 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_comb = nn.Sequential(
            nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)


        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
        down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
        down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
        down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        down_1_comb = self.down_1_comb(down_1_list)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1_comb, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class GANShortGeneratorWithSplitInputChannelAttentionOneAndSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSplitInputChannelAttentionOneAndSpatialAttention, self).__init__()

        self.in_num_ch = in_num_ch
        if self.in_num_ch == 3: # zero-dose
            self.down_1_1 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_2 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_3 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_ca = ChannelAttentionLayer(3*first_num_ch, 4)
            self.down_1_comb = nn.Sequential(
                nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.down_1_1 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_2 = nn.Sequential(
                    nn.Conv2d(2, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_3 = nn.Sequential(
                    nn.Conv2d(2, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_4 = nn.Sequential(
                    nn.Conv2d(3, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_ca = ChannelAttentionLayer(4*first_num_ch, 4)
            self.down_1_comb = nn.Sequential(
                nn.Conv2d(4*first_num_ch, first_num_ch, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )


        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.att_3 = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.att_2 = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.att_1 = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        if self.in_num_ch == 3:
            down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
            down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
            down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
            down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        else:
            input_1 = x[:,2,...].unsqueeze(1)
            input_2 = x[:,:2,...]
            input_3 = x[:,6:,...]
            input_4 = x[:,3:6,...]
            down_1_1 = self.down_1_1(input_1)
            down_1_2 = self.down_1_2(input_2)
            down_1_3 = self.down_1_3(input_3)
            down_1_4 = self.down_1_4(input_4)
            down_1_list = torch.cat([down_1_1, down_1_2, down_1_3, down_1_4], dim=1)

        down_1_ca, _ = self.down_1_ca(down_1_list)
        down_1_comb = self.down_1_comb(down_1_ca)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1_comb, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

'''
current best model 2019/6/12
'''
class GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttention, self).__init__()

        self.in_num_ch = in_num_ch
        if self.in_num_ch == 3: # zero-dose
            self.down_1_1 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_2 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_3 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_ca = ChannelAttentionLayer(3*first_num_ch, 4)
            self.down_1_comb = nn.Sequential(
                nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.down_1_1 = nn.Sequential(
                    nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_2 = nn.Sequential(
                    nn.Conv2d(2, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_3 = nn.Sequential(
                    nn.Conv2d(2, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_4 = nn.Sequential(
                    nn.Conv2d(3, first_num_ch, 4, 2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.down_1_ca = ChannelAttentionLayer(4*first_num_ch, 4)
            self.down_1_comb = nn.Sequential(
                nn.Conv2d(4*first_num_ch, first_num_ch, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4_c = ChannelAttentionLayer(8*first_num_ch, 8)
        self.att_4_s = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)

        self.att_3_c = ChannelAttentionLayer(4*first_num_ch, 4)
        self.att_3_s = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)

        self.att_2_c = ChannelAttentionLayer(2*first_num_ch, 2)
        self.att_2_s = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)

        self.att_1_c = ChannelAttentionLayer(first_num_ch, 1)
        self.att_1_s = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        if self.in_num_ch == 3:
            down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
            down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
            down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
            down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        else:
            input_1 = x[:,2,...].unsqueeze(1) # DWI
            input_2 = x[:,:2,...]   # ADC, ADC thres
            input_3 = x[:,6:,...]   # TMAX, TMAX thres
            input_4 = x[:,3:6,...]  # CBV, CBF, MTT
            down_1_1 = self.down_1_1(input_1)
            down_1_2 = self.down_1_2(input_2)
            down_1_3 = self.down_1_3(input_3)
            down_1_4 = self.down_1_4(input_4)
            down_1_list = torch.cat([down_1_1, down_1_2, down_1_3, down_1_4], dim=1)

        down_1_ca, _ = self.down_1_ca(down_1_list)
        down_1_comb = self.down_1_comb(down_1_ca)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        concat_4_c, _ = self.att_4_c(down_4)
        concat_4_s, alpha_4 = self.att_4_s(down_4, down_5)
        up_4 = self.up_4(concat_4_c+concat_4_s, down_5)

        concat_3_c, _ = self.att_3_c(down_3)
        concat_3_s, alpha_3 = self.att_3_s(down_3, up_4)
        up_3 = self.up_3(concat_3_c+concat_3_s, up_4)

        concat_2_c, _ = self.att_2_c(down_2)
        concat_2_s, alpha_2 = self.att_2_s(down_2, up_3)
        up_2 = self.up_2(concat_2_c+concat_2_s, up_3)

        concat_1_c, _ = self.att_1_c(down_1_comb)
        concat_1_s, alpha_1 = self.att_1_s(down_1_comb, up_2)
        up_1 = self.up_1(concat_1_c+concat_1_s, up_2)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}


class GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention, self).__init__()

        self.down_1 = nn.Sequential(
                nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4_c = ChannelAttentionLayer(8*first_num_ch, 8)
        self.att_4_s = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)

        self.att_3_c = ChannelAttentionLayer(4*first_num_ch, 4)
        self.att_3_s = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)

        self.att_2_c = ChannelAttentionLayer(2*first_num_ch, 2)
        self.att_2_s = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)

        self.att_1_c = ChannelAttentionLayer(first_num_ch, 1)
        self.att_1_s = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        concat_4_c, _ = self.att_4_c(down_4)
        concat_4_s, alpha_4 = self.att_4_s(down_4, down_5)
        up_4 = self.up_4(concat_4_c+concat_4_s, down_5)

        concat_3_c, _ = self.att_3_c(down_3)
        concat_3_s, alpha_3 = self.att_3_s(down_3, up_4)
        up_3 = self.up_3(concat_3_c+concat_3_s, up_4)

        concat_2_c, _ = self.att_2_c(down_2)
        concat_2_s, alpha_2 = self.att_2_s(down_2, up_3)
        up_2 = self.up_2(concat_2_c+concat_2_s, up_3)

        concat_1_c, _ = self.att_1_c(down_1)
        concat_1_s, alpha_1 = self.att_1_s(down_1, up_2)
        up_1 = self.up_1(concat_1_c+concat_1_s, up_2)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}


class GANShortGeneratorWithChannelAttentionAllAndSpatialAttention(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithChannelAttentionAllAndSpatialAttention, self).__init__()

        self.down_1 = nn.Sequential(
                nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4_c = ChannelAttentionLayer(8*first_num_ch, 8)
        self.att_4_s = SpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)

        self.att_3_c = ChannelAttentionLayer(4*first_num_ch, 4)
        self.att_3_s = SpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)

        self.att_2_c = ChannelAttentionLayer(2*first_num_ch, 2)
        self.att_2_s = SpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)

        self.att_1_c = ChannelAttentionLayer(first_num_ch, 1)
        self.att_1_s = SpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        concat_4_c, _ = self.att_4_c(down_4)
        concat_4_s, alpha_4 = self.att_4_s(down_4, down_5)
        up_4 = self.up_4(concat_4_c+concat_4_s, down_5)

        concat_3_c, _ = self.att_3_c(down_3)
        concat_3_s, alpha_3 = self.att_3_s(down_3, up_4)
        up_3 = self.up_3(concat_3_c+concat_3_s, up_4)

        concat_2_c, _ = self.att_2_c(down_2)
        concat_2_s, alpha_2 = self.att_2_s(down_2, up_3)
        up_2 = self.up_2(concat_2_c+concat_2_s, up_3)

        concat_1_c, _ = self.att_1_c(down_1)
        concat_1_s, alpha_1 = self.att_1_s(down_1, up_2)
        up_1 = self.up_1(concat_1_c+concat_1_s, up_2)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}



class GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttentionNoBN(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), output_activation='softplus'):
        super(GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttentionNoBN, self).__init__()

        self.down_1_1 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_2 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_ca = ChannelAttentionLayer(3*first_num_ch, 4)
        self.down_1_comb = nn.Sequential(
            nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch, is_bn=False)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch, is_bn=False)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch, is_bn=False)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no', is_bn=False) # 8 x 8

        self.att_4_c = ChannelAttentionLayer(8*first_num_ch, 8)
        self.att_4_s = SymmetryGateResidualSpatialAttentionLayer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor, is_bn=False)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch, is_bn=False)

        self.att_3_c = ChannelAttentionLayer(4*first_num_ch, 4)
        self.att_3_s = SymmetryGateResidualSpatialAttentionLayer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor, is_bn=False)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch, is_bn=False)

        self.att_2_c = ChannelAttentionLayer(2*first_num_ch, 2)
        self.att_2_s = SymmetryGateResidualSpatialAttentionLayer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor, is_bn=False)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch, is_bn=False)

        self.att_1_c = ChannelAttentionLayer(first_num_ch, 1)
        self.att_1_s = SymmetryGateResidualSpatialAttentionLayer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor, is_bn=False)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch, is_bn=False)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True, is_bn=False)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
        down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
        down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
        down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        down_1_ca, _ = self.down_1_ca(down_1_list)
        down_1_comb = self.down_1_comb(down_1_ca)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        concat_4_c, _ = self.att_4_c(down_4)
        concat_4_s, alpha_4 = self.att_4_s(down_4, down_5)
        up_4 = self.up_4(concat_4_c+concat_4_s, down_5)

        concat_3_c, _ = self.att_3_c(down_3)
        concat_3_s, alpha_3 = self.att_3_s(down_3, up_4)
        up_3 = self.up_3(concat_3_c+concat_3_s, up_4)

        concat_2_c, _ = self.att_2_c(down_2)
        concat_2_s, alpha_2 = self.att_2_s(down_2, up_3)
        up_2 = self.up_2(concat_2_c+concat_2_s, up_3)

        concat_1_c, _ = self.att_1_c(down_1_comb)
        concat_1_s, alpha_1 = self.att_1_s(down_1_comb, up_2)
        up_1 = self.up_1(concat_1_c+concat_1_s, up_2)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}


'''
dual-attention model
'''
class GANShortGeneratorWithSplitInputMultiAttentionAll(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus', is_bn=True):
        super(GANShortGeneratorWithSplitInputMultiAttentionAll, self).__init__()

        self.down_1_1 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_2 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(1, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_1_ca = ChannelAttentionLayer(3*first_num_ch, 4)
        self.down_1_comb = nn.Sequential(
            nn.Conv2d(3*first_num_ch, first_num_ch, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        self.att_4 = MultiAttentionLayer(8*first_num_ch, 8*first_num_ch)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)

        self.att_3 = MultiAttentionLayer(4*first_num_ch, 16*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)

        self.att_2 = MultiAttentionLayer(2*first_num_ch, 8*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)

        self.att_1 = MultiAttentionLayer(first_num_ch, 4*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        down_1_1 = self.down_1_1(x[:,0,...].unsqueeze(1))
        down_1_2 = self.down_1_2(x[:,1,...].unsqueeze(1))
        down_1_3 = self.down_1_3(x[:,2,...].unsqueeze(1))
        down_1_list = torch.cat([down_1_1, down_1_2, down_1_3], dim=1)
        down_1_ca, _ = self.down_1_ca(down_1_list)
        down_1_comb = self.down_1_comb(down_1_ca)
        down_2 = self.down_2(down_1_comb)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        concat_4, alpha_4 = self.att_4(down_4, down_5)
        up_4 = self.up_4(concat_4, down_5)
        concat_3, alpha_3 = self.att_3(down_3, up_4)
        up_3 = self.up_3(concat_3, up_4)
        concat_2, alpha_2 = self.att_2(down_2, up_3)
        up_2 = self.up_2(concat_2, up_3)
        concat_1, alpha_1 = self.att_1(down_1_comb, up_2)
        up_1 = self.up_1(concat_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}

class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
        super(SpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        self.W_out = nn.Sequential(
            nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
            nn.BatchNorm2d(in_num_ch)
        )

    def forward(self, x, g):
        x_size = x.size()
        x_post = self.W_x(x)
        x_post_size = x_post.size()

        g_post = F.upsample(self.W_g(g), size=x_post_size[2:], mode='bilinear')
        xg_post = F.relu(x_post + g_post, inplace=True)
        alpha = F.sigmoid(self.W_psi(xg_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out(alpha_upsample * x)
        return out, alpha_upsample

class SymmetrySpatialAttentionLayer(nn.Module):
    # x + g, flip g
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
        super(SymmetrySpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        self.W_out = nn.Sequential(
            nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
            nn.BatchNorm2d(in_num_ch)
        )

    def forward(self, x, g):
        x_size = x.size()
        x_post = self.W_x(x)
        x_post_size = x_post.size()

        g_flip = torch.flip(g, dims=[2])
        g_diff = torch.abs(g - g_flip)
        g_post = F.upsample(self.W_g(g_diff), size=x_post_size[2:], mode='bilinear')

        xg_post = F.relu(x_post + g_post, inplace=True)
        alpha = F.sigmoid(self.W_psi(xg_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out(alpha_upsample * x)
        return out, alpha_upsample

class SymmetryResidualSpatialAttentionLayer(nn.Module):
    # x + g, flip g
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
        super(SymmetryResidualSpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        self.W_out = nn.Sequential(
            nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
            nn.BatchNorm2d(in_num_ch)
        )

    def forward(self, x, g):
        x_size = x.size()
        x_post = self.W_x(x)
        x_post_size = x_post.size()

        g_flip = torch.flip(g, dims=[2])
        g_diff = torch.abs(g - g_flip)
        g_post = F.upsample(self.W_g(g_diff), size=x_post_size[2:], mode='bilinear')

        xg_post = F.relu(x_post + g_post, inplace=True)
        alpha = F.sigmoid(self.W_psi(xg_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out((1+alpha_upsample) * x)
        return out, alpha_upsample

class SymmetryGateResidualSpatialAttentionLayer(nn.Module):
    # only g
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2), is_bn=True):
        super(SymmetryGateResidualSpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_g_diff = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        if is_bn:
            self.W_out = nn.Sequential(
                nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
                nn.BatchNorm2d(in_num_ch)
            )
        else:
            self.W_out = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)

    def forward(self, x, g):
        x_size = x.size()
        g_flip = torch.flip(g, dims=[2])
        g_diff = torch.abs(g - g_flip)
        g_post = F.relu(self.W_g(g) + self.W_g_diff(g_diff), inplace=True)
        alpha = F.sigmoid(self.W_psi(g_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out((1+alpha_upsample) * x)
        return out, alpha_upsample

class ChannelAttentionLayer(nn.Module):
    # CVPR2018 squeeze and excitation
    def __init__(self, in_num_ch, sample_factor=16):
        super(ChannelAttentionLayer, self).__init__()

        self.W_down = nn.Linear(in_num_ch, in_num_ch//sample_factor)
        self.W_up = nn.Linear(in_num_ch//sample_factor, in_num_ch)

    def forward(self, x):
        x_gp = torch.mean(x, (2,3))

        x_down = F.relu(self.W_down(x_gp))
        alpha = F.sigmoid(self.W_up(x_down))

        alpha_exp = alpha.unsqueeze(2).unsqueeze(3).expand_as(x)
        out = (1 + alpha_exp) * x
        return out, alpha

class MultiAttentionLayer(nn.Module):
    def __init__(self, in_num_ch, gate_num_ch, sample_factor_spatial=(2,2), sample_factor_channel=16, kernel_stride_ratio=4, is_bn=True):
        super(MultiAttentionLayer, self).__init__()
        self.W_x = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)
        self.W_g = nn.Conv2d(gate_num_ch, in_num_ch, 1, 1)
        self.AvgPool = nn.AvgPool2d(kernel_size=tuple([z * kernel_stride_ratio for z in sample_factor_spatial]), stride=sample_factor_spatial)
        self.W_down = nn.Conv2d(in_num_ch, in_num_ch/sample_factor_channel, 1, 1)
        self.W_up = nn.Conv2d(in_num_ch/sample_factor_channel, in_num_ch, 1, 1)
        if is_bn:
            self.W_out = nn.Sequential(
            nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
            nn.BatchNorm2d(in_num_ch)
            )
        else:
            self.W_out = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)

    def forward(self, x, g):
        # add symmetry, combine x and g_diff
        # pdb.set_trace()
        x_size = x.size()
        x_post = self.W_x(x)
        g_diff = g - torch.flip(g, dims=[2])
        g_post = F.interpolate(self.W_g(g_diff), size=x_size[2:], mode='bilinear')
        xg_post = F.relu(x_post + g_post, inplace=True)

        # channel-wise attention for each spatial sample square
        xg_post_avg = self.AvgPool(xg_post)
        xg_down = F.relu(self.W_down(xg_post_avg))
        alpha = F.sigmoid(self.W_up(xg_down))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out((1+alpha_upsample) * x)
        return out, alpha_upsample

# class SymmetryResidualSpatialAttentionLayer_0419(nn.Module):
#     # bug... should use flip x, actually not...
#     def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
#         super(SymmetryResidualSpatialAttentionLayer_0419, self).__init__()
#
#         # in_num_ch, out_num_ch, kernel_size, stride, padding
#         self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
#         self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
#         self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
#         self.W_out = nn.Sequential(
#             nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
#             nn.BatchNorm2d(in_num_ch)
#         )
#
#     def forward(self, x, g):
#         x_size = x.size()
#         x_flip = torch.flip(x, dims=[2])
#         x_diff = torch.abs(x - x_flip)
#         x_post = self.W_x(x)    # should use x_diff
#         x_post_size = x_post.size()
#
#         g_post = F.upsample(self.W_g(g), size=x_post_size[2:], mode='bilinear')
#
#         xg_post = F.relu(x_post + g_post, inplace=True)
#         alpha = F.sigmoid(self.W_psi(xg_post))
#         alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')
#
#         out = self.W_out((1+alpha_upsample) * x)
#         return out, alpha_upsample
#
# class SymmetrySpatialAttentionLayer_0418(nn.Module):
#     # flip g
#     def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
#         super(SymmetrySpatialAttentionLayer_0418, self).__init__()
#
#         # in_num_ch, out_num_ch, kernel_size, stride, padding
#         self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
#         self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
#         self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
#         self.W_out = nn.Sequential(
#             nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
#             nn.BatchNorm2d(in_num_ch)
#         )
#
#     def forward(self, x, g):
#         x_size = x.size()
#         x_post = self.W_x(x)
#         x_post_size = x_post.size()
#
#         g_flip = torch.flip(g, dims=[2])
#         g_diff = torch.abs(g - g_flip)
#         g_post = F.upsample(self.W_g(g_diff), size=x_post_size[2:], mode='bilinear')
#
#         xg_post = F.relu(x_post + g_post, inplace=True)
#         alpha = F.sigmoid(self.W_psi(xg_post))
#         alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')
#
#         out = self.W_out(alpha_upsample * x)
#         return out, alpha_upsample
#
# class SymmetrySpatialAttentionLayer_0419(nn.Module):
#     # flip x, bug...
#     def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
#         super(SymmetrySpatialAttentionLayer_0419, self).__init__()
#
#         # in_num_ch, out_num_ch, kernel_size, stride, padding
#         self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
#         self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
#         self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
#         self.W_out = nn.Sequential(
#             nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
#             nn.BatchNorm2d(in_num_ch)
#         )
#
#     def forward(self, x, g):
#         x_size = x.size()
#         x_flip = torch.flip(x, dims=[2])
#         x_diff = torch.abs(x - x_flip)
#         x_post = self.W_x(x)
#         x_post_size = x_post.size()
#
#         g_post = F.upsample(self.W_g(g), size=x_post_size[2:], mode='bilinear')
#
#         xg_post = F.relu(x_post + g_post, inplace=True)
#         alpha = F.sigmoid(self.W_psi(xg_post))
#         alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')
#
#         out = self.W_out(alpha_upsample * x)
#         return out, alpha_upsample

# class Discriminator(nn.Module):
#     def __init__(self, in_num_ch, first_num_ch=64):
#         super(Discriminator, self).__init__()
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
#         self.conv_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
#         self.conv_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch, stride=1)
#         self.output = nn.Conv2d(8*first_num_ch, 1, 4)
#         # self.output_bn = nn.Sequential(
#         #     nn.BatchNorm2d(1),
#         #     nn.LeakyReLU(0.2, inplace=True)
#         # )
#         self.output_act = nn.Sigmoid()
#
#     def forward(self, x):
#         conv_1 = self.conv_1(x)
#         conv_2 = self.conv_2(conv_1)
#         conv_3 = self.conv_3(conv_2)
#         conv_4 = self.conv_4(conv_3)
#         output = self.output(conv_4)
#         output_act = self.output_act(output)
#         pdb.set_trace()
#         # conv_5 = self.output_bn(output)
#         # return output_act, [conv_1, conv_2, conv_3, conv_4, conv_5]
#         return output, [conv_1, conv_2, conv_3, conv_4]

'''
pretrained amyloid classifier
'''
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, 1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        output = self.net(x)
        output_act = self.output_act(output)
        return output_act

'''
generator for GBM VAE
'''
class GANShortGeneratorVAE(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), output_activation='softplus'):
        super(GANShortGeneratorVAE, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8
        self.up_4 = Act_Deconv_BN_Concat(16*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, input_G, output_P):
        down_1 = self.down_1(input_G)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        cat_5 = torch.cat((down_5, output_P), 1)
        up_4 = self.up_4(down_4, cat_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {}


class LatentLayer(nn.Module):
    def __init__(self, device, stddev=1.0):
        super(LatentLayer, self).__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x, is_sampling=True):
        mean = x
        if not is_sampling:
            return mean
        stddev = self.stddev
        eps_np = np.random.normal(0, 1, size=mean.size())
        eps = torch.from_numpy(eps_np).float().to(self.device)
        return mean + stddev * eps


class VariationNet(nn.Module):
    def __init__(self, device, in_num_ch, first_num_ch=64, input_size=(256,256)):
        super(VariationNet, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8
        self.latent = LatentLayer(device)

    def forward(self, x, is_sampling=True):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        latent = self.latent(down_5, is_sampling)
        return latent


'''
implementation for DANet: Dual Attention Network for Scene Segmentation
part of code modified from https://github.com/junfu1115/DANet
major difference:
1. different input channel, add a upsample layer to make input 224x224
2. smaller backbone, resnet18 or resnet50
'''

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        data_kwargs = {'dilated':True, 'norm_layer':nn.BatchNorm2d, 'root':'./pretrain_models',
                    'multi_grid':True, 'multi_dilation':[4,8,16]}
        self.pretrained = resnet.resnet50(pretrained=True, **data_kwargs)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c4

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        norm_layer = nn.BatchNorm2d
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

class DANet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, input_size=(112,112)):
        super(DANet, self).__init__()
        # 128x128x8->256x256x3 for resnet
        self.input_module = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(in_num_ch, 3, 3, padding=1),
                        nn.ReLU(inplace=True)
                        )
        self.backbone = BackBone()
        self.head = DANetHead(in_channels=2048, out_channels=out_num_ch)

    def forward(self, x):
        imsize = x.size()[2:]
        input = self.input_module(x)
        c4 = self.backbone(input)

        x = self.head(c4)
        x = list(x)
        x[0] = F.upsample(x[0], imsize, mode='bilinear', align_corners=True)
        x[1] = F.upsample(x[1], imsize, mode='bilinear', align_corners=True)
        x[2] = F.upsample(x[2], imsize, mode='bilinear', align_corners=True)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return x[0], []

# 2020/11/29, 3d model for BraTS segmentation
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class VAEBranch(nn.Module):

    def __init__(self, input_shape, init_channels, out_channels, squeeze_channels=None):
        super(VAEBranch, self).__init__()
        self.input_shape = input_shape

        if squeeze_channels:
            self.squeeze_channels = squeeze_channels
        else:
            self.squeeze_channels = init_channels * 4

        self.hidden_conv = nn.Sequential(nn.GroupNorm(8, init_channels * 8),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(init_channels * 8, self.squeeze_channels, (3, 3, 3),
                                                   padding=(1, 1, 1)),
                                         nn.AdaptiveAvgPool3d(1))

        self.mu_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)
        self.logvar_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)

        recon_shape = np.prod(self.input_shape) // (16 ** 3)

        self.reconstraction = nn.Sequential(nn.Linear(self.squeeze_channels // 2, init_channels * 8 * recon_shape),
                                            nn.ReLU(inplace=True))

        self.vconv4 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 8, (1, 1, 1)),
                                    nn.Upsample(scale_factor=2))

        self.vconv3 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 4, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 4, init_channels * 4))

        self.vconv2 = nn.Sequential(nn.Conv3d(init_channels * 4, init_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 2, init_channels * 2))

        self.vconv1 = nn.Sequential(nn.Conv3d(init_channels * 2, init_channels, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels, init_channels))

        self.vconv0 = nn.Conv3d(init_channels, out_channels, (1, 1, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.hidden_conv(x)
        batch_size = x.size()[0]
        x = x.view((batch_size, -1))
        mu = x[:, :self.squeeze_channels // 2]
        mu = self.mu_fc(mu)
        logvar = x[:, self.squeeze_channels // 2:]
        logvar = self.logvar_fc(logvar)
        z = self.reparameterize(mu, logvar)
        re_x = self.reconstraction(z)
        # recon_shape = [batch_size,
        #                self.squeeze_channels // 2,
        #                self.input_shape[0] // 16,
        #                self.input_shape[1] // 16,
        #                self.input_shape[2] // 16]
        # re_x = re_x.view(recon_shape)
        re_x = re_x.view([batch_size,-1, self.input_shape[0] // 16,self.input_shape[1] // 16,self.input_shape[2] // 16])
        x = self.vconv4(re_x)
        x = self.vconv3(x)
        x = self.vconv2(x)
        x = self.vconv1(x)
        vout = self.vconv0(x)

        return vout, mu, logvar


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        # uout = F.sigmoid(uout)

        return uout, c4d


class NVNet3D(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=16, p=0.2):
        super(NVNet3D, self).__init__()
        self.unet = UNet3D(input_shape, in_channels, out_channels, init_channels, p)
        self.vae_branch = VAEBranch(input_shape, init_channels, out_channels=in_channels)

    def forward(self, x):
        uout, c4d = self.unet(x)
        vout, mu, logvar = self.vae_branch(c4d)

        return uout, vout, mu, logvar



# 2020/08/26, new multimodality model with missing inputs
class _routing(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(_routing, self).__init__()
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, inputs_type):
        x = self.fc(inputs_type)
        return F.sigmoid(x)

class CondConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, embeddings=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(embeddings, num_experts)  # embeddings=shape of inputs_type label

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        self.init_weights()
        # self.reset_parameters()

    def init_weights(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


    def forward(self, inputs, inputs_type):
        bs, _, _, _ = inputs.size()
        res = []
        routing_weights = self._routing_fn(inputs_type)
        # print((routing_weights[: , :, None, None, None, None, None] * self.weight).shape)
        kernels = torch.sum(routing_weights[:, :, None, None, None, None] * self.weight, 1)
        for i in range(bs):
            out = self._conv_forward(inputs[i].unsqueeze(0), kernels[i])
            res.append(out)
        return torch.cat(res, dim=0)

def Conv2d(is_cond):
    return CondConv2d if is_cond else nn.Conv2d

class Conv_BN_Act_New(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1, activation='lrelu', is_bn=True, is_cond=False):
        super(Conv_BN_Act_New, self).__init__()
        self.is_bn = is_bn
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('Ana Encoder enc conv block', conv2d)

        self.conv = conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
        if is_bn:
            self.bn = nn.BatchNorm2d(out_num_ch)

        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x, inputs_type=None):
        # if inputs_type == None:
        #     inputs_type = torch.ones(x.shape[0], 1).cuda()
        if self.is_cond:
            x = self.conv(x, inputs_type)
        else:
            x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x

class Act_Deconv_BN_Concat_New(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu', upsample=True, is_last=False, is_bn=True, is_cond=False):
        super(Act_Deconv_BN_Concat_New, self).__init__()
        self.is_bn = is_bn
        self.is_cond = is_cond
        self.upsample = upsample
        conv2d = Conv2d(is_cond)
        print('Ana Encoder dec deconv block', conv2d)

        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_down, x_up, inputs_type=None):
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.upsample:
            if self.is_cond:
                x_up = self.conv(x_up, inputs_type)
            else:
                x_up = self.conv(x_up)
        if self.is_last == False:
            if self.is_bn:
                x_up = self.bn(x_up)
            x = torch.cat([x_down, x_up], 1)
        else:
            x = x_up
        return x

class AnatomyEncoderEnc(nn.Module):
    def __init__(self, in_num_ch=7, first_num_ch=32):
        super(AnatomyEncoderEnc, self).__init__()
        # encoder part of anatomy encoder (unet)
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        return [down_1, down_2, down_3, down_4, down_5]

class AnatomyEncoderEncNew(nn.Module):
    def __init__(self, in_num_ch=7, first_num_ch=32, is_cond=False):
        super(AnatomyEncoderEncNew, self).__init__()
        # encoder part of anatomy encoder (unet)
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('Ana Encoder enc', conv2d)

        self.down_1 = conv2d(in_num_ch, first_num_ch, 4, 2, padding=1)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.down_2 = Conv_BN_Act_New(first_num_ch, 2*first_num_ch, is_cond=is_cond)
        self.down_3 = Conv_BN_Act_New(2*first_num_ch, 4*first_num_ch, is_cond=is_cond)
        self.down_4 = Conv_BN_Act_New(4*first_num_ch, 8*first_num_ch, is_cond=is_cond)
        self.down_5 = Conv_BN_Act_New(8*first_num_ch, 8*first_num_ch, activation='no', is_cond=is_cond) # 8 x 8

    def forward(self, x, inputs_type=None):
        # if inputs_type == None:
        #     inputs_type = torch.ones(x.shape[0], 1).cuda()
        if self.is_cond:
            down_1 = self.down_1(x, inputs_type)
        else:
            down_1 = self.down_1(x)
        down_1 = self.act_1(down_1)
        down_2 = self.down_2(down_1, inputs_type)
        down_3 = self.down_3(down_2, inputs_type)
        down_4 = self.down_4(down_3, inputs_type)
        down_5 = self.down_5(down_4, inputs_type)
        return [down_1, down_2, down_3, down_4, down_5]

class AnatomyEncoderDec(nn.Module):
    def __init__(self, first_num_ch=32, out_num_ch=8, output_act='softmax'):
        super(AnatomyEncoderDec, self).__init__()
        # decoder part of anatomy encoder (unet)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)
        if output_act == 'softmax':
            self.output_act = nn.Softmax()
        else:
            self.output_act = nn.Softplus()

    def forward(self, down_list):
        up_4 = self.up_4(down_list[3], down_list[4])
        up_3 = self.up_3(down_list[2], up_4)
        up_2 = self.up_2(down_list[1], up_3)
        up_1 = self.up_1(down_list[0], up_2)
        output = self.output(None, up_1)
        output_act = self.output_act(output)
        output_act_bi = torch.round(output_act)
        return output_act, output_act_bi

class AnatomyEncoderDecNew(nn.Module):
    def __init__(self, first_num_ch=32, out_num_ch=8, output_act='softmax', is_cond=False):
        super(AnatomyEncoderDecNew, self).__init__()
        # decoder part of anatomy encoder (unet)
        self.up_4 = Act_Deconv_BN_Concat_New(8*first_num_ch, 8*first_num_ch, is_cond=is_cond)
        self.up_3 = Act_Deconv_BN_Concat_New(16*first_num_ch, 4*first_num_ch, is_cond=is_cond)
        self.up_2 = Act_Deconv_BN_Concat_New(8*first_num_ch, 2*first_num_ch, is_cond=is_cond)
        self.up_1 = Act_Deconv_BN_Concat_New(4*first_num_ch, first_num_ch, is_cond=is_cond)
        self.output = Act_Deconv_BN_Concat_New(2*first_num_ch, out_num_ch, is_last=True, is_cond=is_cond)
        # if output_act == 'softmax':
        #     self.output_act = nn.Softmax()
        # else:
        #     self.output_act = nn.Softplus()

    def forward(self, down_list, inputs_type=None):
        if inputs_type == None:
            inputs_type = torch.ones(down_list[0].shape[0], 1).cuda()
        up_4 = self.up_4(down_list[3], down_list[4], inputs_type)
        up_3 = self.up_3(down_list[2], up_4, inputs_type)
        up_2 = self.up_2(down_list[1], up_3, inputs_type)
        up_1 = self.up_1(down_list[0], up_2, inputs_type)
        output = self.output(None, up_1, inputs_type)
        # output_act = self.output_act(output)
        # output_act_bi = torch.round(output_act)
        # return output_act, output_act_bi
        return output, output

class ModalityEncoder(nn.Module):
    def __init__(self, img_num_ch=7, s_num_ch=8, first_num_ch=16, z_size=16):
        super(ModalityEncoder, self).__init__()
        self.s_num_ch = s_num_ch
        self.convs = nn.Sequential(
                        nn.Conv2d(img_num_ch+s_num_ch, first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(first_num_ch, 2*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(2*first_num_ch, 4*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(4*first_num_ch, 8*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(8*first_num_ch, 8*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True)
                    )

        self.fcs = nn.Sequential(
                        nn.Linear(5*6*8*first_num_ch, 2*z_size),
                        nn.LeakyReLU(0.2, inplace=True))
        self.mean = nn.Linear(2*first_num_ch, z_size)
        self.log_var = nn.Linear(2*first_num_ch, z_size)

    def forward(self, xi, si):
        if self.s_num_ch == 0:
            x = self.convs(xi)
        else:
            x = self.convs(torch.cat([xi, si], 1))
        x = x.view(-1, 5*6*128)
        x = self.fcs(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var

class ModalityEncoderNew(nn.Module):
    def __init__(self, img_num_ch=7, s_num_ch=8, first_num_ch=16, z_size=16, is_cond=False):
        super(ModalityEncoderNew, self).__init__()
        self.s_num_ch = s_num_ch
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('Mod Encoder enc', conv2d)

        self.conv1 = conv2d(img_num_ch+s_num_ch, first_num_ch, 3, 2, padding=1)
        self.conv2 = conv2d(first_num_ch, 2*first_num_ch, 3, 2, padding=1)
        self.conv3 = conv2d(2*first_num_ch, 4*first_num_ch, 3, 2, padding=1)
        self.conv4 = conv2d(4*first_num_ch, 8*first_num_ch, 3, 2, padding=1)
        self.conv5 = conv2d(8*first_num_ch, 8*first_num_ch, 3, 2, padding=1)

        self.convs = nn.Sequential(
                        nn.Conv2d(img_num_ch+s_num_ch, first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(first_num_ch, 2*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(2*first_num_ch, 4*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(4*first_num_ch, 8*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(8*first_num_ch, 8*first_num_ch, 3, 2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True)
                    )

        self.fcs = nn.Sequential(
                        nn.Linear(5*6*8*first_num_ch, 2*z_size),
                        nn.LeakyReLU(0.2, inplace=True))
        # TODO: should try TanH, got inf here
        self.mean = nn.Linear(2*first_num_ch, z_size)
        self.log_var = nn.Linear(2*first_num_ch, z_size)

    def forward(self, xi, si, inputs_type=None):
        # if inputs_type == None:
        #     inputs_type = torch.ones(xi.shape[0], 1).cuda()
        if self.s_num_ch == 0:
            xsi = xi
        else:
            xsi = torch.cat([xi, si], 1)
        if self.is_cond:
            x1 = self.conv1(xsi, inputs_type)
            x1 = F.leaky_relu(x1, 0.2)
            x2 = self.conv2(x1, inputs_type)
            x2 = F.leaky_relu(x2, 0.2)
            x3 = self.conv3(x2, inputs_type)
            x3 = F.leaky_relu(x3, 0.2)
            x4 = self.conv4(x3, inputs_type)
            x4 = F.leaky_relu(x4, 0.2)
            x5 = self.conv5(x4, inputs_type)
            x5 = F.leaky_relu(x5, 0.2)
        else:
            x1 = self.conv1(xsi)
            x1 = F.leaky_relu(x1, 0.2)
            x2 = self.conv2(x1)
            x2 = F.leaky_relu(x2, 0.2)
            x3 = self.conv3(x2)
            x3 = F.leaky_relu(x3, 0.2)
            x4 = self.conv4(x3)
            x4 = F.leaky_relu(x4, 0.2)
            x5 = self.conv5(x4)
            x5 = F.leaky_relu(x5, 0.2)

        x = x5.view(-1, 5*6*128)
        x = self.fcs(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var

class SPADEBlock(nn.Module):
    def __init__(self, input_size, in_num_ch=128, out_num_ch=128, s_num_ch=8):
        super(SPADEBlock, self).__init__()
        self.zi_layers = nn.InstanceNorm2d(in_num_ch)
        self.si_layers = nn.Sequential(
                            nn.Upsample(size=input_size, mode='bilinear'),
                            nn.Conv2d(s_num_ch, in_num_ch, 3, 1, padding=1)
                            )
        self.gamma = nn.Conv2d(in_num_ch, in_num_ch, 3, 1, padding=1)
        self.beta = nn.Conv2d(in_num_ch, in_num_ch, 3, 1, padding=1)
        self.out = nn.Conv2d(in_num_ch, out_num_ch, 3, 1, padding=1)

    def forward(self, si, zi):
        # pdb.set_trace()
        zi_out = self.zi_layers(zi)
        si_out = self.si_layers(si)
        gamma = self.gamma(si_out)
        beta = self.beta(si_out)
        mix = zi_out * (1 + gamma) + beta
        out = self.out(mix)
        return out

class SPADEBlockNew(nn.Module):
    def __init__(self, input_size, in_num_ch=128, out_num_ch=128, s_num_ch=8, is_cond=False):
        super(SPADEBlockNew, self).__init__()
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('SPADE block', conv2d)

        self.zi_layers = nn.InstanceNorm2d(in_num_ch)
        self.up = nn.Upsample(size=input_size, mode='bilinear')
        self.si_layers = conv2d(s_num_ch, in_num_ch, 3, 1, padding=1)
        self.gamma = conv2d(in_num_ch, in_num_ch, 3, 1, padding=1)
        self.beta = conv2d(in_num_ch, in_num_ch, 3, 1, padding=1)
        self.out = conv2d(in_num_ch, out_num_ch, 3, 1, padding=1)

    def forward(self, si, zi, inputs_type=None):
        # pdb.set_trace()
        zi_out = self.zi_layers(zi)
        si = self.up(si)
        if self.is_cond:
            si_out = self.si_layers(si, inputs_type)
            gamma = self.gamma(si_out, inputs_type)
            beta = self.beta(si_out, inputs_type)
            mix = zi_out * (1 + gamma) + beta
            out = self.out(mix, inputs_type)
        else:
            si_out = self.si_layers(si)
            gamma = self.gamma(si_out)
            beta = self.beta(si_out)
            mix = zi_out * (1 + gamma) + beta
            out = self.out(mix)
        return out

class SPADE(nn.Module):
    def __init__(self, image_size=(192, 160), in_num_ch=7, z_size=16, z_num_ch=128, s_num_ch=8):
        super(SPADE, self).__init__()
        self.z_num_ch = z_num_ch
        self.image_size = image_size
        self.zi_scaler = nn.Linear(z_size, image_size[0]*image_size[1]*z_num_ch//1024)
        self.sp1 = SPADEBlock(input_size=(image_size[0]//32,image_size[1]//32), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch)
        self.up1 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp2 = SPADEBlock(input_size=(image_size[0]//16,image_size[1]//16), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch)
        self.up2 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp3 = SPADEBlock(input_size=(image_size[0]//8,image_size[1]//8), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch)
        self.up3 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp4 = SPADEBlock(input_size=(image_size[0]//4,image_size[1]//4), in_num_ch=z_num_ch, out_num_ch=z_num_ch//2, s_num_ch=s_num_ch)
        self.up4 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp5 = SPADEBlock(input_size=(image_size[0]//2,image_size[1]//2), in_num_ch=z_num_ch//2, out_num_ch=z_num_ch//4, s_num_ch=s_num_ch)
        self.up5 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp6 = SPADEBlock(input_size=(image_size[0],image_size[1]), in_num_ch=z_num_ch//4, out_num_ch=z_num_ch//8, s_num_ch=s_num_ch)
        self.out = nn.Sequential(
                        nn.Conv2d(z_num_ch//8, in_num_ch, 1, 1),
                        nn.Softplus())

    def forward(self, si, zi):
        # pdb.set_trace()
        zi_rescale = self.zi_scaler(zi)
        zi_reshape = zi_rescale.reshape(-1, self.z_num_ch, self.image_size[0]//32, self.image_size[1]//32)
        zi_sp1 = self.sp1(si, zi_reshape)
        zi_sp2 = self.sp2(si, self.up1(zi_sp1))
        zi_sp3 = self.sp3(si, self.up2(zi_sp2))
        zi_sp4 = self.sp4(si, self.up3(zi_sp3))
        zi_sp5 = self.sp5(si, self.up4(zi_sp4))
        zi_sp6 = self.sp6(si, self.up5(zi_sp5))
        out = self.out(zi_sp6)
        return out

class SPADENew(nn.Module):
    def __init__(self, image_size=(192, 160), in_num_ch=7, z_size=16, z_num_ch=128, s_num_ch=8, is_cond=False, output_activation='softplus'):
        super(SPADENew, self).__init__()
        self.z_num_ch = z_num_ch
        self.image_size = image_size
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('SPADE', conv2d)

        self.zi_scaler = nn.Linear(z_size, image_size[0]*image_size[1]*z_num_ch//1024)
        self.sp1 = SPADEBlockNew(input_size=(image_size[0]//32,image_size[1]//32), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up1 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp2 = SPADEBlockNew(input_size=(image_size[0]//16,image_size[1]//16), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up2 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp3 = SPADEBlockNew(input_size=(image_size[0]//8,image_size[1]//8), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up3 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp4 = SPADEBlockNew(input_size=(image_size[0]//4,image_size[1]//4), in_num_ch=z_num_ch, out_num_ch=z_num_ch//2, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up4 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp5 = SPADEBlockNew(input_size=(image_size[0]//2,image_size[1]//2), in_num_ch=z_num_ch//2, out_num_ch=z_num_ch//4, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up5 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp6 = SPADEBlockNew(input_size=(image_size[0],image_size[1]), in_num_ch=z_num_ch//4, out_num_ch=z_num_ch//8, s_num_ch=s_num_ch, is_cond=is_cond)
        self.out = conv2d(z_num_ch//8, in_num_ch, 1, 1)

        if output_activation == 'softplus':
            self.out_act = nn.Softplus()
        elif output_activation == 'no':
            self.out_act = nn.Sequential()
        else:
            raise ValueError('No activation in SPADENotShared')
        print('input decoder activation: ', output_activation)

    def forward(self, si, zi, inputs_type=None):
        # pdb.set_trace()
        # if inputs_type == None:
        #     inputs_type = torch.ones(si.shape[0], 1).cuda()
        zi_rescale = self.zi_scaler(zi)
        zi_reshape = zi_rescale.reshape(-1, self.z_num_ch, self.image_size[0]//32, self.image_size[1]//32)
        zi_sp1 = self.sp1(si, zi_reshape, inputs_type)
        zi_sp2 = self.sp2(si, self.up1(zi_sp1), inputs_type)
        zi_sp3 = self.sp3(si, self.up2(zi_sp2), inputs_type)
        zi_sp4 = self.sp4(si, self.up3(zi_sp3), inputs_type)
        zi_sp5 = self.sp5(si, self.up4(zi_sp4), inputs_type)
        zi_sp6 = self.sp6(si, self.up5(zi_sp5), inputs_type)
        if self.is_cond:
            out = self.out(zi_sp6, inputs_type)
        else:
            out = self.out(zi_sp6)
        out_act = self.out_act(out)
        return out_act

class SPADENewShared(nn.Module):
    def __init__(self, image_size=(192, 160), in_num_ch=7, z_size=16, z_num_ch=128, s_num_ch=8, is_cond=False):
        super(SPADENewShared, self).__init__()
        self.z_num_ch = z_num_ch
        self.image_size = image_size
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('SPADENewShared', conv2d)

        self.zi_scaler = nn.Linear(z_size, image_size[0]*image_size[1]*z_num_ch//1024)
        self.sp1 = SPADEBlockNew(input_size=(image_size[0]//32,image_size[1]//32), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up1 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp2 = SPADEBlockNew(input_size=(image_size[0]//16,image_size[1]//16), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up2 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp3 = SPADEBlockNew(input_size=(image_size[0]//8,image_size[1]//8), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up3 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.sp4 = SPADEBlockNew(input_size=(image_size[0]//4,image_size[1]//4), in_num_ch=z_num_ch, out_num_ch=z_num_ch//2, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.up4 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.sp5 = SPADEBlockNew(input_size=(image_size[0]//2,image_size[1]//2), in_num_ch=z_num_ch//2, out_num_ch=z_num_ch//4, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.up5 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.sp6 = SPADEBlockNew(input_size=(image_size[0],image_size[1]), in_num_ch=z_num_ch//4, out_num_ch=z_num_ch//8, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.out = conv2d(z_num_ch//8, in_num_ch, 1, 1)
        # self.out_act = nn.Softplus()

    def forward(self, si, zi, inputs_type=None):
        # pdb.set_trace()
        # if inputs_type == None:
        #     inputs_type = torch.ones(si.shape[0], 1).cuda()
        zi_rescale = self.zi_scaler(zi)
        zi_reshape = zi_rescale.reshape(-1, self.z_num_ch, self.image_size[0]//32, self.image_size[1]//32)
        zi_sp1 = self.sp1(si, zi_reshape, inputs_type)
        zi_sp2 = self.sp2(si, self.up1(zi_sp1), inputs_type)
        zi_sp3 = self.sp3(si, self.up2(zi_sp2), inputs_type)
        zi_sp4_input = self.up2(zi_sp3)
        # zi_sp4 = self.sp4(si, self.up3(zi_sp3), inputs_type)
        # zi_sp5 = self.sp5(si, self.up4(zi_sp4), inputs_type)
        # zi_sp6 = self.sp6(si, self.up5(zi_sp5), inputs_type)
        # if self.is_cond:
        #     out = self.out(zi_sp6, inputs_type)
        # else:
        #     out = self.out(zi_sp6)
        # out_act = self.out_act(out)
        return zi_sp4_input

class SPADENewNotShared(nn.Module):
    def __init__(self, image_size=(192, 160), in_num_ch=7, z_size=16, z_num_ch=128, s_num_ch=8, is_cond=False, output_activation='softplus'):
        super(SPADENewNotShared, self).__init__()
        self.z_num_ch = z_num_ch
        self.image_size = image_size
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('SPADENewNotShared', conv2d)

        # self.zi_scaler = nn.Linear(z_size, image_size[0]*image_size[1]*z_num_ch//1024)
        # self.sp1 = SPADEBlockNew(input_size=(image_size[0]//32,image_size[1]//32), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.up1 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.sp2 = SPADEBlockNew(input_size=(image_size[0]//16,image_size[1]//16), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.up2 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.sp3 = SPADEBlockNew(input_size=(image_size[0]//8,image_size[1]//8), in_num_ch=z_num_ch, out_num_ch=z_num_ch, s_num_ch=s_num_ch, is_cond=is_cond)
        # self.up3 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp4 = SPADEBlockNew(input_size=(image_size[0]//4,image_size[1]//4), in_num_ch=z_num_ch, out_num_ch=z_num_ch//2, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up4 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp5 = SPADEBlockNew(input_size=(image_size[0]//2,image_size[1]//2), in_num_ch=z_num_ch//2, out_num_ch=z_num_ch//4, s_num_ch=s_num_ch, is_cond=is_cond)
        self.up5 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.sp6 = SPADEBlockNew(input_size=(image_size[0],image_size[1]), in_num_ch=z_num_ch//4, out_num_ch=z_num_ch//8, s_num_ch=s_num_ch, is_cond=is_cond)
        self.out = conv2d(z_num_ch//8, in_num_ch, 1, 1)

        if output_activation == 'softplus':
            self.out_act = nn.Softplus()
        elif output_activation == 'no':
            self.out_act = nn.Sequential()
        else:
            raise ValueError('No activation in SPADENotShared')
        print('input decoder activation: ', output_activation)

    def forward(self, si, zi_sp4_input, inputs_type=None):
        # pdb.set_trace()
        # if inputs_type == None:
        #     inputs_type = torch.ones(si.shape[0], 1).cuda()
        # zi_rescale = self.zi_scaler(zi)
        # zi_reshape = zi_rescale.reshape(-1, self.z_num_ch, self.image_size[0]//32, self.image_size[1]//32)
        # zi_sp1 = self.sp1(si, zi_reshape, inputs_type)
        # zi_sp2 = self.sp2(si, self.up1(zi_sp1), inputs_type)
        # zi_sp3 = self.sp3(si, self.up2(zi_sp2), inputs_type)
        zi_sp4 = self.sp4(si, zi_sp4_input, inputs_type)
        zi_sp5 = self.sp5(si, self.up4(zi_sp4), inputs_type)
        zi_sp6 = self.sp6(si, self.up5(zi_sp5), inputs_type)
        if self.is_cond:
            out = self.out(zi_sp6, inputs_type)
        else:
            out = self.out(zi_sp6)
        out_act = self.out_act(out)
        return out_act

class Conv_BN_Act_New2(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1,
                activation='lrelu', is_bn=True, is_cond=False, embeddings=16):
        # Conv_BN_Act_New conditioned on contrast type, Conv_BN_Act_New2 conditioned on z
        super(Conv_BN_Act_New2, self).__init__()
        self.is_bn = is_bn
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('Ana Encoder enc conv block', conv2d)

        self.conv = conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding, embeddings=embeddings)
        if is_bn:
            self.bn = nn.BatchNorm2d(out_num_ch)

        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x, inputs_type=None):
        # if inputs_type == None:
        #     inputs_type = torch.ones(x.shape[0], 1).cuda()
        if self.is_cond:
            x = self.conv(x, inputs_type)
        else:
            x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class Act_Deconv_BN_Concat_New2(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu',
                    upsample=True, is_last=False, is_bn=True, is_cond=False, embeddings=16):
        super(Act_Deconv_BN_Concat_New2, self).__init__()
        self.is_bn = is_bn
        self.is_cond = is_cond
        self.upsample = upsample
        conv2d = Conv2d(is_cond)
        print('Ana Encoder dec deconv block', conv2d)

        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding, embeddings=embeddings)
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_down, x_up, inputs_type=None):
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.upsample:
            if self.is_cond:
                x_up = self.conv(x_up, inputs_type)
            else:
                x_up = self.conv(x_up)
        if self.is_last == False:
            if self.is_bn:
                x_up = self.bn(x_up)
            x = torch.cat([x_down, x_up], 1)
        else:
            x = x_up
        return x


class GANShortGeneratorNew(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, z_size=16, input_size=(256,256), output_activation='softplus', is_cond=False):
        super(GANShortGeneratorNew, self).__init__()
        self.is_cond = is_cond
        conv2d = Conv2d(is_cond)
        print('Short Generator', conv2d)

        self.down_1 = conv2d(in_num_ch, first_num_ch, 4, 2, padding=1, embeddings=z_size)
        self.down_1_act = nn.LeakyReLU(0.2, inplace=True)
        self.down_2 = Conv_BN_Act_New2(first_num_ch, 2*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.down_3 = Conv_BN_Act_New2(2*first_num_ch, 4*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.down_4 = Conv_BN_Act_New2(4*first_num_ch, 8*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.down_5 = Conv_BN_Act_New2(8*first_num_ch, 8*first_num_ch, activation='no', is_cond=is_cond, embeddings=z_size)
        self.up_4 = Act_Deconv_BN_Concat_New2(8*first_num_ch, 8*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.up_3 = Act_Deconv_BN_Concat_New2(16*first_num_ch, 4*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.up_2 = Act_Deconv_BN_Concat_New2(8*first_num_ch, 2*first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.up_1 = Act_Deconv_BN_Concat_New2(4*first_num_ch, first_num_ch, is_cond=is_cond, embeddings=z_size)
        self.output = Act_Deconv_BN_Concat_New2(2*first_num_ch, out_num_ch, is_last=True, is_cond=is_cond, embeddings=z_size)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

    def forward(self, xi, zi=None, inputs_type=None):
        if self.is_cond:
            down_1 = self.down_1_act(self.down_1(xi, zi))
            down_2 = self.down_2(down_1, zi)
            down_3 = self.down_3(down_2, zi)
            down_4 = self.down_4(down_3, zi)
            down_5 = self.down_5(down_4, zi)
            up_4 = self.up_4(down_4, down_5, zi)
            up_3 = self.up_3(down_3, up_4, zi)
            up_2 = self.up_2(down_2, up_3, zi)
            up_1 = self.up_1(down_1, up_2, zi)
            output = self.output(None, up_1, zi)
        else:
            down_1 = self.down_1_act(self.down_1(x))
            down_2 = self.down_2(down_1)
            down_3 = self.down_3(down_2)
            down_4 = self.down_4(down_3)
            down_5 = self.down_5(down_4)
            up_4 = self.up_4(down_4, down_5)
            up_3 = self.up_3(down_3, up_4)
            up_2 = self.up_2(down_2, up_3)
            up_1 = self.up_1(down_1, up_2)
            output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act


class Discriminator(nn.Module):
    def __init__(self, in_num_ch=8, inter_num_ch=16, input_shape=[160,192], is_patch_gan=False):
        super(Discriminator, self).__init__()
        self.discrim = nn.Sequential(
                            nn.Conv2d(in_num_ch, inter_num_ch, 4, 2, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(inter_num_ch, 2*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(2*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(2*inter_num_ch, 4*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(4*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(4*inter_num_ch, 8*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(8*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(8*inter_num_ch, 4*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(4*inter_num_ch),
                            nn.LeakyReLU(0.2))
                            # nn.Tanh())
        if is_patch_gan:
            self.fc = nn.Conv2d(4*inter_num_ch, 1, 3, 1, padding=1)
        else:
            self.fc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(int(input_shape[0]*input_shape[1]*4*inter_num_ch/(32*32)), inter_num_ch*16),
                                nn.LeakyReLU(0.2),
                                nn.Linear(inter_num_ch*16, 1))

    def forward(self, x):
        conv = self.discrim(x)
        fc = self.fc(conv)
        return fc


class LowdoseModel(nn.Module):
    def __init__(self, in_num_ch=3, out_num_ch=1):
        super(LowdoseModel, self).__init__()
        self.conv1 = nn.Sequential(
                            nn.Conv2d(in_num_ch, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.conv2 = nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.conv3 = nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.bottleneck = nn.MaxPool2d(2)
        self.conv4 = nn.Sequential(
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.up3 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.dconv3 = nn.Sequential(
                            nn.Conv2d(128, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.dconv2 = nn.Sequential(
                            nn.Conv2d(96, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.dconv1 = nn.Sequential(
                            nn.Conv2d(64, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 1, 3, padding=1),
                            nn.Tanh())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        bottleneck = self.bottleneck(conv3)
        conv4 = self.conv4(bottleneck)
        up3 = self.up3(conv4 + bottleneck)
        dconv3 = self.dconv3(torch.cat([up3, conv3], 1))
        up2 = self.up2(dconv3)
        dconv2 = self.dconv2(torch.cat([up2, conv2], 1))
        up1 = self.up1(dconv2)
        dconv1 = self.dconv1(torch.cat([up1, conv1], 1))
        output = x[:,0:1]+dconv1
        return output, None



class ModalityDistribution(nn.Module):
    def __init__(self, z_size=16, inter_num_ch=128):
        super(ModalityDistribution, self).__init__()
        self.z_size = z_size
        self.linear = nn.Sequential(
                        nn.Linear(1, inter_num_ch),
                        nn.LeakyReLU(0.2),
                        nn.Linear(inter_num_ch, 2*z_size))


    def forward(self, x):
        feat = self.linear(x)
        return feat[:,:self.z_size], feat[:,self.z_size:]   # mean, log_var

class MultimodalModel(nn.Module):
    def __init__(self, input_size=(160,192), modality_num=4, in_num_ch=7, out_num_ch=1, s_num_ch=8, z_size=16,
                        is_discrim_s=False, is_distri_z=False, shared_ana_enc=False, shared_mod_enc=True, shared_inp_dec=True,
                        s_compact_method='max', s_sim_method='cosine', z_sim_method='cosine',
                        is_cond=True, input_output_act='softplus', target_output_act='softplus', target_model_name='U',
                        fuse_method='mean', device=torch.device('cuda:0'), others={'mod_enc_s': True, 'ana_dec_act': 'softmax'}):
        super(MultimodalModel, self).__init__()
        self.input_size = input_size
        self.modality_num = modality_num
        self.in_num_ch = in_num_ch
        self.out_num_ch = out_num_ch
        self.fuse_method = fuse_method
        self.device = device
        self.shared_ana_enc = shared_ana_enc
        self.shared_mod_enc = shared_mod_enc
        self.shared_inp_dec = shared_inp_dec
        self.s_compact_method = s_compact_method
        self.s_sim_method = s_sim_method
        self.z_sim_method = z_sim_method
        self.is_cond = is_cond
        self.others = others
        self.anatomy_encoder_enc_list, self.anatomy_encoder_dec = self.define_anatomy_encoder_list(out_num_ch=s_num_ch)
        if self.others['old']:
            self.modality_encoder = ModalityEncoder(img_num_ch=in_num_ch, s_num_ch=s_num_ch, first_num_ch=16, z_size=z_size).to(self.device)
            self.input_decoder = SPADE(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch).to(self.device)
        else:
            self.modality_encoder_list = self.define_modality_encoder_list(in_num_ch=in_num_ch, s_num_ch=s_num_ch, z_size=z_size)
            self.input_decoder_list = self.define_input_decoder_list(input_size=input_size, in_num_ch=in_num_ch, z_size=z_size, s_num_ch=s_num_ch, output_activation=input_output_act)

        if self.s_compact_method == 'vgg':
            self.vgg_pre = nn.Conv2d(s_num_ch, 3, 3, padding=1).to(self.device)
            self.vgg = models.vgg16(pretrained=True).to(self.device)
            for param in self.vgg.parameters():
                param.requires_grad = False

        if fuse_method == 'mean-max-min':
            fuse_num_ch = 3
        else:
            fuse_num_ch = 1
        if target_model_name == 'U':
            self.output_decoder = GANShortGenerator(in_num_ch=fuse_num_ch*s_num_ch, out_num_ch=out_num_ch, first_num_ch=64, input_size=input_size, output_activation=target_output_act).to(self.device)
        elif target_model_name == 'U+SA':
            self.output_decoder = GANShortGeneratorWithSpatialAttention(in_num_ch=fuse_num_ch*s_num_ch, out_num_ch=out_num_ch, first_num_ch=64, input_size=input_size, output_activation=target_output_act).to(self.device)
        elif target_model_name == 'U+SA+CA':
            self.output_decoder = GANShortGeneratorWithChannelAttentionAllAndSpatialAttention(in_num_ch=fuse_num_ch*s_num_ch, out_num_ch=out_num_ch, first_num_ch=64, input_size=input_size, output_activation=target_output_act).to(self.device)
        elif target_model_name == 'U+SSA+CA':
            self.output_decoder = GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention(in_num_ch=fuse_num_ch*s_num_ch, out_num_ch=out_num_ch, first_num_ch=64, input_size=input_size, output_activation=target_output_act).to(self.device)
        else:
            raise ValueError('Not implemented')
        # extra structures for loss
        if is_discrim_s:
            self.discrim_s = Discriminator(in_num_ch=s_num_ch, inter_num_ch=16).to(self.device)
        if is_distri_z:
            self.distri_z = ModalityDistribution(z_size=z_size, inter_num_ch=128).to(self.device)


    '''
    Old nn.Conv2d model
    '''
    '''
    def define_anatomy_encoder_list(self, out_num_ch=8, first_num_ch=32):
        # pdb.set_trace()
        if 'ana_dec_act' in self.others and self.others['ana_dec_act'] == 'softplus':
            output_act = 'softplus'
        else:
            output_act = 'softmax'
        encoder_list = nn.ModuleList([])
        if self.shared_ana_enc:
            encoder_list.append(AnatomyEncoderEnc(in_num_ch=self.in_num_ch, first_num_ch=first_num_ch).to(self.device))
        else:
            for i in range(self.modality_num):
                encoder_list.append(AnatomyEncoderEnc(in_num_ch=self.in_num_ch, first_num_ch=first_num_ch).to(self.device))
        decoder = AnatomyEncoderDec(first_num_ch=first_num_ch, out_num_ch=out_num_ch, output_act=output_act).to(self.device)
        return encoder_list, decoder

    def define_modality_encoder_list(self, in_num_ch=7, s_num_ch=8, z_size=16):
        # pdb.set_trace()
        if 'mod_enc_s' in self.others and self.others['mod_enc_s'] == False:
            s_num_ch = 0
        encoder_list = nn.ModuleList([])
        if self.shared_mod_enc:
            encoder_list.append(ModalityEncoder(img_num_ch=in_num_ch, s_num_ch=s_num_ch, first_num_ch=16, z_size=z_size).to(self.device))
        else:
            for i in range(self.modality_num):
                encoder_list.append(ModalityEncoder(img_num_ch=in_num_ch, s_num_ch=s_num_ch, first_num_ch=16, z_size=z_size).to(self.device))
        return encoder_list

    def define_input_decoder_list(self, input_size=(160,192), in_num_ch=7, z_size=16, s_num_ch=8):
        # pdb.set_trace()
        decoder_list = nn.ModuleList([])
        if self.shared_inp_dec:
            decoder_list.append(SPADE(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch).to(self.device))
        else:
            for i in range(self.modality_num):
                decoder_list.append(SPADE(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch).to(self.device))
        return decoder_list

    def compute_anatomy_encoding(self, inputs_list):
        # pdb.set_trace()
        si_list = []
        for i in range(self.modality_num):
            if self.shared_ana_enc:
                feat_list = self.anatomy_encoder_enc_list[0](inputs_list[i])
            else:
                feat_list = self.anatomy_encoder_enc_list[i](inputs_list[i])
            si, _ = self.anatomy_encoder_dec(feat_list)
            si_list.append(si)
        return si_list

    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(self.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def compute_modality_encoding(self, inputs_list, si_list, phase='train'):
        # pdb.set_trace()
        zi_list = []
        zi_mean_list = []
        zi_log_var_list = []
        for i in range(self.modality_num):
            if self.others['old']:
                z_mean, z_log_var = self.modality_encoder(inputs_list[i], si_list[i])
            elif self.shared_mod_enc:
                z_mean, z_log_var = self.modality_encoder_list[0](inputs_list[i], si_list[i])
            else:
                z_mean, z_log_var = self.modality_encoder_list[i](inputs_list[i], si_list[i])

            if phase == 'train':
                zi = self.sample(z_mean, z_log_var)
            else:
                zi = z_mean
            zi_list.append(zi)
            zi_mean_list.append(z_mean)
            zi_log_var_list.append(z_log_var)
        return zi_list, zi_mean_list, zi_log_var_list

    def reconstruct_input_si_zi(self, si_list, zi_list):
        # pdb.set_trace()
        xi_fake_list = []
        for i in range(self.modality_num):
            if self.others['old']:
                xi_fake = self.input_decoder(si_list[i], zi_list[i])
            elif self.shared_inp_dec:
                xi_fake = self.input_decoder_list[0](si_list[i], zi_list[i])
            else:
                xi_fake = self.input_decoder_list[i](si_list[i], zi_list[i])
            xi_fake_list.append(xi_fake)
        return xi_fake_list

    def reconstruct_input_si_zj(self, si_list, zi_list):
        # pdb.set_trace()
        xi_fake_list = []
        for i in range(self.modality_num):
            for j in range(self.modality_num):
                if i == j:
                    continue
                if self.others['old']:
                    xi_fake = self.input_decoder(si_list[i], zi_list[j])
                elif self.shared_inp_dec:
                    xi_fake = self.input_decoder_list[0](si_list[i], zi_list[j])
                else:
                    xi_fake = self.input_decoder_list[j](si_list[i], zi_list[j])
                xi_fake_list.append(xi_fake)
        return xi_fake_list
    '''

    '''
    New CondConv model start
    '''
    # '''
    def define_anatomy_encoder_list(self, out_num_ch=8, first_num_ch=32):
        # pdb.set_trace()
        # if 'ana_dec_act' in self.others and self.others['ana_dec_act'] == 'softplus':
        #     output_act = 'softplus'
        # else:
        #     output_act = 'softmax'
        encoder_list = nn.ModuleList([])
        if self.shared_ana_enc:
            encoder_list.append(AnatomyEncoderEncNew(in_num_ch=self.in_num_ch, first_num_ch=first_num_ch, is_cond=self.is_cond).to(self.device))
        else:
            for i in range(self.modality_num):
                encoder_list.append(AnatomyEncoderEncNew(in_num_ch=self.in_num_ch, first_num_ch=first_num_ch, is_cond=self.is_cond).to(self.device))
        decoder = AnatomyEncoderDecNew(first_num_ch=first_num_ch, out_num_ch=out_num_ch, is_cond=self.is_cond).to(self.device)
        # decoder = AnatomyEncoderDecNew(first_num_ch=first_num_ch, out_num_ch=out_num_ch, output_act=output_act, is_cond=self.is_cond).to(self.device)
        return encoder_list, decoder

    def define_modality_encoder_list(self, in_num_ch=7, s_num_ch=8, z_size=16):
        # pdb.set_trace()
        if 'mod_enc_s' in self.others and self.others['mod_enc_s'] == False:
            s_num_ch = 0
        encoder_list = nn.ModuleList([])
        if self.shared_mod_enc:
            encoder_list.append(ModalityEncoderNew(img_num_ch=in_num_ch, s_num_ch=s_num_ch, first_num_ch=16, z_size=z_size, is_cond=self.is_cond).to(self.device))
        else:
            for i in range(self.modality_num):
                encoder_list.append(ModalityEncoderNew(img_num_ch=in_num_ch, s_num_ch=s_num_ch, first_num_ch=16, z_size=z_size, is_cond=self.is_cond).to(self.device))
        return encoder_list

    def define_input_decoder_list(self, input_size=(160,192), in_num_ch=7, z_size=16, s_num_ch=8, output_activation='softplus'):
        # pdb.set_trace()
        decoder_list = nn.ModuleList([])
        if self.shared_inp_dec:
            # normal U-Net, conditioned on z
            # decoder_list.append(GANShortGeneratorNew(in_num_ch=s_num_ch, out_num_ch=in_num_ch, first_num_ch=64, z_size=z_size, input_size=input_size, output_activation='softplus', is_cond=self.is_cond).to(self.device))

            # SPADE, conditioned on contrast type
            decoder_list.append(SPADENew(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch, is_cond=self.is_cond, output_activation=output_activation).to(self.device))
        else:
            # not shared
            # for i in range(self.modality_num):
            #     decoder_list.append(SPADENew(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch, is_cond=self.is_cond).to(self.device))

            # shared half, conditioned on contrast type
            for i in range(self.modality_num):
                decoder_list.append(SPADENewNotShared(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch, is_cond=self.is_cond, output_activation=output_activation).to(self.device))
            decoder_list.append(SPADENewShared(image_size=input_size, in_num_ch=in_num_ch, z_size=z_size, z_num_ch=128, s_num_ch=s_num_ch, is_cond=self.is_cond).to(self.device))

        return decoder_list

    def compute_anatomy_encoding(self, inputs_list, mask_img):
        si_list = []
        for i in range(self.modality_num):
            inputs_type = (1+i) * torch.ones(inputs_list[0].shape[0], 1, device=self.device)
            # inputs_type = i * torch.ones(inputs_list[0].shape[0], 1, device=self.device)
            if self.shared_ana_enc:
                feat_list = self.anatomy_encoder_enc_list[0](inputs_list[i], inputs_type)
            else:
                feat_list = self.anatomy_encoder_enc_list[i](inputs_list[i], inputs_type)
            si, _ = self.anatomy_encoder_dec(feat_list, inputs_type)
            if 'ana_dec_act' in self.others and self.others['ana_dec_act'] == 'softplus':
                si_act = F.softplus(si)
            else:
                # pdb.set_trace()
                if 'softmax_remove_mask' in self.others and self.others['softmax_remove_mask'] == True:
                    si_cat = torch.cat([100*mask_img.unsqueeze(1), si], dim=1)
                    # si_cat = torch.cat([mask_img.unsqueeze(1), si], dim=1)
                    si_act_cat = F.softmax(si_cat, dim=1)
                    si_act = si_act_cat[:,1:]
                else:
                    si_act = F.softmax(si, dim=1)
            si_list.append(si_act)
        return si_list

    def sample(self, z_mean, z_log_var):
        eps = torch.normal(0, 1, size=(z_mean.shape[0], z_mean.shape[1])).to(self.device)
        zi = z_mean + eps * torch.exp(0.5 * z_log_var)
        return zi

    def compute_modality_encoding(self, inputs_list, si_list, phase='train'):
        zi_list = []
        zi_mean_list = []
        zi_log_var_list = []
        for i in range(self.modality_num):
            inputs_type = (1+i) * torch.ones(inputs_list[0].shape[0], 1, device=self.device)
            # inputs_type = i * torch.ones(inputs_list[0].shape[0], 1, device=self.device)
            if self.others['old']:
                z_mean, z_log_var = self.modality_encoder(inputs_list[i], si_list[i])
            elif self.shared_mod_enc:
                z_mean, z_log_var = self.modality_encoder_list[0](inputs_list[i], si_list[i], inputs_type)
            else:
                z_mean, z_log_var = self.modality_encoder_list[i](inputs_list[i], si_list[i], inputs_type)

            if phase == 'train':
                zi = self.sample(z_mean, z_log_var)
            else:
                zi = z_mean
            zi_list.append(zi)
            zi_mean_list.append(z_mean)
            zi_log_var_list.append(z_log_var)
        return zi_list, zi_mean_list, zi_log_var_list

    def reconstruct_input_si_zi(self, si_list, zi_list):
        xi_fake_list = []
        for i in range(self.modality_num):
            inputs_type = (1+i) * torch.ones(si_list[0].shape[0], 1, device=self.device)
            # inputs_type = i * torch.ones(si_list[0].shape[0], 1, device=self.device)
            if self.others['old']:
                xi_fake = self.input_decoder(si_list[i], zi_list[i])
            elif self.shared_inp_dec:
                xi_fake = self.input_decoder_list[0](si_list[i], zi_list[i], inputs_type)
            else:
                # not shared
                # xi_fake = self.input_decoder_list[i](si_list[i], zi_list[i], inputs_type)
                # shared half
                mid_out = self.input_decoder_list[-1](si_list[i], zi_list[i], inputs_type)
                xi_fake = self.input_decoder_list[i](si_list[i], mid_out, inputs_type)
            xi_fake_list.append(xi_fake)
        return xi_fake_list

    def reconstruct_input_si_zj(self, si_list, zi_list):
        xi_fake_list = []
        for i in range(self.modality_num):
            for j in range(self.modality_num):
                if i == j:
                    continue
                inputs_type = (1+j) * torch.ones(si_list[0].shape[0], 1, device=self.device)
                # inputs_type = j * torch.ones(si_list[0].shape[0], 1, device=self.device)
                if self.others['old']:
                    xi_fake = self.input_decoder(si_list[i], zi_list[j])
                elif self.shared_inp_dec:
                    xi_fake = self.input_decoder_list[0](si_list[i], zi_list[j], inputs_type)
                else:
                    # not shared
                    # xi_fake = self.input_decoder_list[j](si_list[i], zi_list[j], inputs_type)
                    # shared half
                    mid_out = self.input_decoder_list[-1](si_list[i], zi_list[j], inputs_type)
                    xi_fake = self.input_decoder_list[i](si_list[i], mid_out, inputs_type)
                xi_fake_list.append(xi_fake)
        return xi_fake_list
    # '''
    '''
    CondConv stop here
    '''

    def reconstruct_output_si(self, si_list):
        y_fake_list = []
        bs = si_list[0].shape[0]
        for i in range(self.modality_num):
            # output, _ = self.output_decoder(si_list[i])
            output = self.reconstruct_output_si_fused([si_list[i]], torch.ones(bs, 1, device=self.device))
            y_fake_list.append(output)
        return y_fake_list

    def reconstruct_output_si_fused(self, si_list, mask):
        # pdb.set_trace()
        si_cat = torch.stack(si_list,1)
        si_sel_list = si_cat[mask==1]
        if len(si_cat.shape) != len(si_sel_list.shape):
            si_sel_list = si_sel_list.unsqueeze(1)
        if self.fuse_method == 'mean':
            si_fused = torch.mean(si_sel_list, 1)
        elif self.fuse_method == 'max':
            si_fused, _ = torch.max(si_sel_list, 1)
        elif self.fuse_method == 'mean-max-min':
            si_max, _ = torch.max(si_sel_list, 1)
            si_mean = torch.mean(si_sel_list, 1)
            si_min, _ = torch.min(si_sel_list, 1)
            si_fused = torch.cat([si_mean, si_max, si_min], dim=1)
        else:
            raise ValueError('No fused method')

        output, _ = self.output_decoder(si_fused)
        return output

    def compute_recon_loss(self, gt, output, p=2):
        # loss on reconstruction
        dim = [i for i in range(1,len(gt.shape))]
        if p == 1:
            return torch.abs(gt - output).mean(dim)
        else:
            return torch.pow(gt - output, 2).mean(dim)

    def compute_recon_loss_y_list(self, gt, y_list, mask, p=2):
        loss = torch.tensor(0., device=self.device)
        idx = 0
        for i in range(len(y_list)):
            if mask[:,i].sum() == 0:
                continue
            idx += 1
            loss += (mask[:,i] * self.compute_recon_loss(gt, y_list[i], p)).sum() / (mask[:,i]).sum()
        if idx == 0:
            return loss
        return loss / idx

    def compute_recon_loss_y(self, gt, y, p=2):
        try:
            loss = self.compute_recon_loss(gt, y, p).mean()
        except:
            pdb.set_trace()
        return self.compute_recon_loss(gt, y, p).mean()

    def compute_segmentation_loss_y(self, gt, y, weight=torch.tensor([1.,5.,5.,5.])):
        loss_seg = F.cross_entropy(y, gt.squeeze(1).long(), weight=weight.to(self.device))
        loss_dice = 0
        y_act = F.softmax(y)
        for i in range(1, 4):
            gt_i = (gt[:,0] == i).float()
            numerator = 2 * torch.sum(y_act[:,i] * gt_i)
            denominator = torch.sum(y_act[:,i]**2 + gt_i**2)
            loss_dice += 1 - numerator / (denominator + 1e-6)
        loss_dice = loss_dice / 3
        return loss_seg + loss_dice

    def compute_segmentation_loss_y_list(self, gt, y_list, mask, weight=torch.tensor([1.,5.,5.,5.])):
        loss = torch.tensor(0., device=self.device)
        idx = 0
        for i in range(len(y_list)):
            if mask[:,i].sum() == 0:
                continue
            idx += 1
            loss_y = self.compute_segmentation_loss_y(gt, y_list[i], weight)
            loss += loss_y
            # gt = gt.squeeze(1).long()
            # dim = [j for j in range(1,len(gt.shape))]
            # loss += (mask[:,i] * F.cross_entropy(y_list[i], gt, weight=weight.to(self.device), reduction='none').mean(dim)).sum() / (mask[:,i]).sum()
        if idx == 0:
            return loss
        return loss / idx

    def compute_recon_loss_x_list(self, gt_list, x_list, mask, p=2):
        loss = torch.tensor(0., device=self.device)
        idx = 0
        for i in range(len(x_list)):
            if mask[:,i].sum() == 0:
                continue
            idx += 1
            loss += (mask[:,i] * self.compute_recon_loss(gt_list[i], x_list[i], p)).sum() / (mask[:,i]).sum()
        if idx == 0:
            return loss
        return loss / idx

    def compute_recon_loss_x_mix_list(self, gt_list, x_list, mask, p=2):
        idx = 0
        loss = torch.tensor(0., device=self.device)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[1]):
                if i == j:
                    continue
                mask_mix = mask[:,i] * mask[:,j]
                if mask_mix.sum() == 0:
                    continue
                loss += (mask_mix * self.compute_recon_loss(gt_list[j], x_list[idx], p)).sum() / mask_mix.sum()
                idx += 1
        if idx == 0:
            return loss
        return loss / idx

    def compute_kl_loss_standard(self, z_mean, z_log_var, mask):
        if mask.sum() == 0:
            loss = torch.tensor(0., device=self.device)
        kl = 0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1. - z_log_var, 1)
        return (kl * mask).sum() / mask.sum()

    # def compute_kl_loss_list_standard(self, zi_mean_list, zi_log_var_list, mask):
    #     loss = 0
    #     for i in range(len(zi_mean_list)):
    #         loss += self.compute_kl_loss_standard(zi_mean_list[i], zi_log_var_list[i], mask[:,i])
    #     return loss / len(zi_mean_list)

    def compute_kl_loss_list_standard(self, zi_mean_list, zi_log_var_list, mask):
        zi_mean_all = torch.cat(zi_mean_list, 0)
        zi_log_var_all = torch.cat(zi_log_var_list, 0)
        mask_all = torch.cat([mask[:,i] for i in range(mask.shape[1])], 0)
        loss = self.compute_kl_loss_standard(zi_mean_all, zi_log_var_all, mask_all)
        return loss / len(zi_mean_list)

    def compute_zi_prior_distribution(self, bs, num_distri, device):
        zi_prior_mean_list = []
        zi_prior_log_var_list = []
        for i in range(num_distri):
            input = (i+1) * torch.ones(bs, 1).to(self.device)
            zi_mean, zi_log_var = self.distri_z(input)
            zi_prior_mean_list.append(zi_mean)
            zi_prior_log_var_list.append(zi_log_var)
        return zi_prior_mean_list, zi_prior_log_var_list

    def compute_kl_loss_two_gaussian(self, z_mean, z_log_var, z_prior_mean, z_prior_log_var, mask):
        if mask.sum() == 0:
            loss = torch.tensor(0., device=self.device)
        kl = 0.5*(-1 + (z_prior_log_var-z_log_var) + (torch.exp(z_log_var)+(z_mean-z_prior_mean)**2) / torch.exp(z_prior_log_var))
        return (kl * mask.unsqueeze(1)).sum() / mask.sum()

    def compute_kl_loss_list_two_gaussian(self, z_mean_list, z_log_var_list, z_prior_mean_list, z_prior_log_var_list, mask):
        loss = 0
        for i in range(len(z_mean_list)):
            loss += self.compute_kl_loss_two_gaussian(z_mean_list[i], z_log_var_list[i], z_prior_mean_list[i], z_prior_log_var_list[i], mask[:,i])
        return loss / len(z_mean_list)

    def compute_latent_z_loss(self, zi_mean_list, zi_mean_list_new, mask):
        loss = torch.tensor(0., device=self.device)
        idx = 0
        for i in range(len(zi_mean_list)):
            if mask[:,i].sum() == 0:
                continue
            idx += 1
            loss += (mask[:,i].unsqueeze(1) * torch.abs(zi_mean_list[i] - zi_mean_list_new[i])).sum() / mask[:,i].sum()
        if idx == 0:
            return loss
        return loss / idx

    def compute_nearest_neighbour_z_by_s(self, s_all, z_all, s_tar):
        cosine_list = []
        s_tar_tile = s_tar.unsqueeze(0).repeat(s_all.shape[0],1)
        cosine_list = self.compute_cosine(s_all, s_tar_tile)
        idx_sel = torch.argmax(cosine_list)
        print(idx_sel)
        return z_all[idx_sel]

    def compute_mean_z_by_s(self, z_all):
        return z_all.mean(0)

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def compute_perceptual(self, x, y):
        pad_x = (224 - self.input_size[0]) // 2
        pad_y = (224 - self.input_size[1]) // 2
        x_pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x), "constant", 0)
        x_pre = self.vgg_pre(x_pad)
        y_pad = F.pad(y, (pad_y, pad_y, pad_x, pad_x), "constant", 0)
        y_pre = self.vgg_pre(y_pad)

        # content: conv4_2 (21)
        content_feat_x = self.vgg.features[:21](x_pre)
        content_feat_y = self.vgg.features[:21](y_pre)
        content_loss = F.mse_loss(content_feat_x, content_feat_y)

        def gram_matrix(feature):
            batch_size, num_ch, height, width = feature.size()  # NCHW
            feature = feature.view(batch_size, num_ch, height * width)
            gram = torch.bmm(feature, feature.transpose(1,2))
            return gram.div(height * width)

        # style: conv1_1 (0), conv2_1 (5), conv3_1 (10), conv4_1 (17), conv5_1 (24)
        style_loss = 0
        for i in [0, 5, 10, 17, 24]:
            style_feat_x = self.vgg.features[:i](x_pre)
            style_feat_y = self.vgg.features[:i](y_pre)
            gram_x = gram_matrix(style_feat_x)
            gram_y = gram_matrix(style_feat_y)
            style_loss += F.mse_loss(gram_x, gram_y)/(gram_x.shape[-1]**2)

        return -(content_loss + 1e3 * style_loss)


    def compute_compact_s_max(self, x):
        x_pool = F.max_pool2d(x, kernel_size=(16,16))
        x_vec = x_pool.view(x.shape[0], -1)
        return x_vec

    def compute_compact_s_mean(self, x):
        x_pool = F.avg_pool2d(x, kernel_size=(16,16))
        x_vec = x_pool.view(x.shape[0], -1)
        return x_vec

    def compute_compact_s_vgg(self, x):
        pad_x = (224 - self.input_size[0]) // 2
        pad_y = (224 - self.input_size[1]) // 2
        x_pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x), "constant", 0)
        x_pre = self.vgg_pre(x_pad)
        x_vgg = self.vgg.features(x_pre)
        x_vec = F.avg_pool2d(x_vgg, kernel_size=7).view(x.shape[0], -1)
        return x_vec

    def compute_compact_s(self, x):
        # pdb.set_trace()
        if self.s_compact_method == 'max':
            x_vec = self.compute_compact_s_max(x)
        elif self.s_compact_method == 'mean':
            x_vec = self.compute_compact_s_mean(x)
        else:
            x_vec = self.compute_compact_s_vgg(x)
        return x_vec


    def compute_similarity_s_loss(self, si_list, mask, margin=0.1):
        loss = torch.tensor(0., device=self.device)
        if len(si_list) == 1:
            return loss
        if len(si_list) == 2:
            i, j = 0, 1
        else:
            idx_sel = np.random.choice(len(si_list), 2, replace=False)
            i, j = idx_sel[0], idx_sel[1]

        # use selected i and j
        si = si_list[i]
        si_perm = torch.cat([si[1:], si[0:1]], 0)
        mask_i_perm = torch.cat([mask[1:,i], mask[0:1,i]], 0)
        sj = si_list[j]
        mask_mix = mask[:,i] * mask[:,j] * mask_i_perm
        if mask_mix.sum() > 0:
            if self.s_sim_method == 'cosine':
                si_c = self.compute_compact_s(si)
                sj_c = self.compute_compact_s(sj)
                si_perm_c = self.compute_compact_s(si_perm)
                sim = self.compute_cosine(si_c, sj_c)
                # sim_mix = self.compute_cosine(si_perm_c, sj_c)    # wrong
                sim_mix = self.compute_cosine(si_perm_c, si_c)
                loss = (mask_mix * torch.max(torch.zeros_like(mask[:,0]), margin - sim + sim_mix)).sum() / mask_mix.sum()
            else:
                sim = self.compute_perceptual(si, sj)
                # sim_mix = self.compute_perceptual(si_perm, sj)
                # print(sim, sim_mix, - sim + sim_mix)
                # sim, larger -> better
                # loss = (mask_mix * torch.max(torch.zeros_like(mask[:,0]), 2. - sim + sim_mix)).sum() / mask_mix.sum()
                loss = -(mask_mix * sim).sum() / mask_mix.sum()

        else:
            loss = 0
        return loss

        # old, compare each pair of si sj
        # idx = 0
        # for i in range(len(si_list)-1):
        #     si = si_list[i]
        #     si_perm = torch.cat([si[1:], si[0:1]], 0)
        #     mask_i_perm = torch.cat([mask[1:,i], mask[0:1,i]], 0)
        #     for j in range(i+1, len(si_list)):
        #         sj = si_list[j]
        #         mask_mix = mask[:,i] * mask[:,j] * mask_i_perm
        #         if mask_mix.sum() == 0:
        #             continue
        #         idx += 1
        #         si_c = self.compute_compact_s(si)
        #         sj_c = self.compute_compact_s(sj)
        #         si_perm_c = self.compute_compact_s(si_perm)
        #         cosine = self.compute_cosine(si_c, sj_c)
        #         cosine_mix = self.compute_cosine(si_perm_c, sj_c)
        #         loss += (mask_mix * torch.max(torch.zeros_like(mask[:,0]), margin - cosine + cosine_mix)).sum() / mask_mix.sum()
        # if idx == 0:
        #     return loss
        # return loss / idx

    def compute_similarity_z_loss(self, zi_list, mask, margin=0.1):
        loss = torch.tensor(0., device=self.device)
        if len(zi_list) == 1:
            return loss
        idx = 0
        for i in range(len(zi_list)-1):
            zi = zi_list[i]
            zi_perm = torch.cat([zi[1:], zi[0:1]], 0)
            mask_i_perm = torch.cat([mask[1:,i], mask[0:1,i]], 0)
            for j in range(i+1, len(zi_list)):
                zj = zi_list[j]
                mask_mix = mask[:,i] * mask[:,j] * mask_i_perm
                if mask_mix.sum() == 0:
                    continue
                idx += 1
                cosine = self.compute_cosine(zi, zj)            # diff modality, same subj, should be different
                cosine_mix = self.compute_cosine(zi, zi_perm)   # same modality, diff subj, should be similar
                loss += (mask_mix * torch.max(torch.zeros_like(mask[:,0]), margin - cosine_mix + cosine)).sum() / mask_mix.sum()
        if idx == 0:
            return loss
        return loss / idx

    def compute_adversarial_loss(self, si_list, mask):
        if len(si_list) == 2:
            i, j = 0, 1
        else:
            # raise ValueError('Adversarial training only support two contrasts')
            idx_sel = np.random.choice(len(si_list), 2, replace=False)
            i, j = idx_sel[0], idx_sel[1]

        d_0 = self.discrim_s(si_list[i]).squeeze(1)
        d_1 = self.discrim_s(si_list[j]).squeeze(1)
        if mask[:,i].sum() == 0:
            d_loss_0 = torch.tensor(0., device=self.device)
            g_loss_0 = torch.tensor(0., device=self.device)
        else:
            d_loss_0 = (mask[:,i] * F.binary_cross_entropy_with_logits(d_0, torch.zeros_like(d_0), reduction='none')).sum() / mask[:,i].sum()
            g_loss_0 = (mask[:,i] * F.binary_cross_entropy_with_logits(d_0, torch.ones_like(d_0), reduction='none')).sum() / mask[:,i].sum()
        if mask[:,j].sum() == 0:
            d_loss_1 = torch.tensor(0., device=self.device)
            g_loss_1 = torch.tensor(0., device=self.device)
        else:
            d_loss_1 = (mask[:,j] * F.binary_cross_entropy_with_logits(d_1, torch.ones_like(d_1), reduction='none')).sum() / mask[:,j].sum()
            g_loss_1 = (mask[:,j] * F.binary_cross_entropy_with_logits(d_1, torch.ones_like(d_1), reduction='none')).sum() / mask[:,j].sum()
            # g_loss_1 = (mask[:,1] * F.binary_cross_entropy_with_logits(d_1, torch.zeros_like(d_1), reduction='none')).sum() / mask[:,1].sum()
        d_loss = 0.5 * (d_loss_0 + d_loss_1)
        g_loss = 0.5 * (g_loss_0 + g_loss_1)
        # print(d_loss_0.item(), d_loss_1.item(), g_loss_0.item(), g_loss_1.item())
        # print(((d_0 < 0).sum()/8.).item(), ((d_1 > 0).sum()/8.).item())
        # print(F.sigmoid(d_0).detach().cpu().numpy(), F.sigmoid(d_1).detach().cpu().numpy())
        return d_loss, g_loss
