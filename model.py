import numpy as np
from skimage.morphology import white_tophat, square
from skimage.transform import rotate
from skimage.filters import threshold_otsu
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from utils import initialize_weights, get_upsampling_weight


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=5, stride=2, output_padding=1, dilation=3),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, raw_img_channel):
        super(UNet, self).__init__()
        # self.thresh = torch.tensor([0.5])
        # self.enc1 = _EncoderBlock(3, 64)
        self.enc1 = _EncoderBlock(raw_img_channel, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear')


# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8s, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # print(vgg)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        # print(classifier[0].weight.data.view(4096, 512, 7, 7).shape)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(), nn.Dropout(), fc7, nn.ReLU(), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_( get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

        # print('done')

    def forward(self, x):
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)
        # print('x size:', x.shape)
        # print('pool3 size:', pool3.shape)
        # print('pool4 size:', pool4.shape)
        # print('pool5 size:', pool5.shape)

        score_fr = self.score_fr(pool5)
        # print('score_fr size:', score_fr.shape)

        upscore2 = self.upscore2(score_fr)
        # print('up_score2 size:', upscore2.shape)

        score_pool4 = self.score_pool4(0.01 * pool4)
        # print('score_pool4 size:', score_pool4.shape)
        upscore4 = self.upscore4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                 + upscore2)
        # print('upscore_poll4 size:', upscore4.shape)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        # print('score_pool3 size:', score_pool3.shape)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore4.size()[2]), 9: (9 + upscore4.size()[3])]
                                 + upscore4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()


class MBI:
    def __init__(self, img_arr, s_min, s_max, delta_s):
        # img_arr is N * C * W * H tensor
        self.img_arr = img_arr.numpy()
        self.s_min = s_min
        self.s_max = s_max
        self.delta_s = delta_s
        self.mbi = []
        self.set_mbi()

    def set_mbi(self):
        bsz = self.img_arr.shape[0]
        for ii in range(bsz):
            img = self.img_arr[ii, :, :, :]
            # print(img.shape)

            gray = np.max(img, 0)
            gray = np.pad(gray, ((self.s_min, self.s_min), (self.s_min, self.s_min)), 'constant',
                          constant_values=(0, 0))

            mp_lis = []
            for i in range(self.s_min, self.s_max, 2 * self.delta_s):
                se_inter = square(i)
                se_inter[:int((i - 1) / 2), :] = 0
                se_inter[int((i - 1) / 2) + 1:, :] = 0
                for angle in range(0, 180, 45):
                    se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
                    mp = white_tophat(gray, selem=se_inter)
                    mp_lis.append(mp)

            dmp_lis = []
            for i in range(self.delta_s, len(mp_lis)):
                dmp = np.absolute(mp_lis[i] - mp_lis[i - self.delta_s])
                dmp_lis.append(dmp)

            mbi = np.sum(dmp_lis, axis=0) / (4 * ((self.s_max - self.s_min) / self.delta_s + 1))
            mbi = mbi[self.s_min: mbi.shape[0] - self.s_min, self.s_min: mbi.shape[1] - self.s_min]
            # print(np.unique(mbi))

            # bug existed here
            thres = threshold_otsu(mbi)
            gt_idx = mbi > thres
            le_idx = mbi <= thres
            mbi[gt_idx] = 1
            mbi[le_idx] = 0
            # print(np.unique(mbi))

            self.mbi.append(mbi)

        self.mbi = torch.tensor(np.array(self.mbi))

    def get_mbi(self):
        assert isinstance(self.mbi, torch.Tensor)
        n = self.mbi.shape[0]
        w = self.mbi.shape[1]
        h = self.mbi.shape[2]
        return self.mbi.reshape(n, 1, w, h)

# def MBI(img_arr, s_min, s_max, delta_s): refactor this to class
#     gray = np.max(img_arr, 0)
#     gray = np.pad(gray, ((s_min, s_min), (s_min, s_min)), 'constant', constant_values=(0, 0))
#     mp_lis = []
#     dmp_lis = []
#
#     for i in range(s_min, s_max + 1, 2 * delta_s):
#         se_inter = square(i)
#         se_inter[:int((i - 1) / 2), :] = 0
#         se_inter[int((i - 1) / 2) + 1:, :] = 0
#         for angle in range(0, 180, 45):
#             se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
#             mp = white_tophat(gray, selem=se_inter)
#             mp_lis.append(mp)
#
#     for i in range(4, len(mp_lis)):
#         dmp = np.absolute(mp_lis[i] - mp_lis[i-4])
#         dmp_lis.append(dmp)
#
#     mbi = np.sum(dmp_lis, axis=0) / (4 * ( (s_max - s_min) / 4 + 1))
#     mbi = mbi[s_min: mbi.shape[0] - s_min, s_min: mbi.shape[1] - s_min]
#
#     thres = threshold_otsu(mbi)
#     mbi[mbi > thres] = 255
#     mbi[mbi <= thres] = 0
#
#     return mbi

class FusionFCN(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(FusionFCN, self).__init__()
        self.img0_size = img0_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(img0_size, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2, 1)

        #  左右上下
        # self.replication_pad = nn.ReplicationPad2d((0, 1, 0, 1))
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.AvgPool2d(2, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 + img1_size, 1, kernel_size=1),
            nn.ReLU()
        )
        initialize_weights(self)

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]
        conv1 = self.conv1(x0)
        conv1 = self.zero_pad(conv1)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv2 = self.zero_pad(conv2)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv3 = self.zero_pad(conv3)
        pool3 = self.pool3(conv3)

        # print(conv1.shape)
        # print(conv2.shape)
        # print(conv3.shape)
        # print(pool1.shape)
        # print(pool2.shape)
        # print(pool3.shape)

        merged_pool = pool1 + pool2 + pool3
        concat_data = torch.concat([merged_pool, x1], 1)

        conv4 = self.conv4(concat_data)

        return conv4


class BasicAtrousConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicAtrousConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiScaleAtrousConv2d(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(MultiScaleAtrousConv2d, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.conv1 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6 = BasicAtrousConv2d(in_channel=mid_channel, out_channel=mid_channel,
                                       kernel_size=3, stride=1, padding=5, dilation=5)
        self.conv7 = nn.Conv2d(in_channels=mid_channel * 3, out_channels=in_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        con1 = self.conv1(x)
        con1 = self.conv2(con1)
        con1 = self.conv3(con1)
        con2 = self.conv4(x)
        con2 = self.conv5(con2)
        con2 = self.conv6(con2)
        x = self.conv7(torch.concat([x, con1, con2], 1))
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class SpectralAttention(nn.Module):
    def __init__(self, feature_channel):
        super(SpectralAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_channel, 8),
            nn.ReLU(),
            nn.Linear(8, feature_channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgt = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        maxt = torch.max(x.view(x.size(0), x.size(1), -1), 2)
        weighted_avgt = self.mlp(avgt[0])
        weighted_maxt = self.mlp(maxt[0])
        return self.sigmoid(weighted_maxt + weighted_avgt).unsqueeze(2).unsqueeze(3).expand_as(x)


class MSMFFModule(nn.Module):
    def __init__(self, feature0_channel, feature1_channel, input_three_stream=False):
        """
        :param feature0_channel: PAN/RGB, default 32
        :param feature1_channel: spectral, default 32
        """
        super(MSMFFModule, self).__init__()
        self.feature0_channel = feature0_channel
        self.feature1_channel = feature1_channel
        sum_channel = feature0_channel + feature1_channel
        if input_three_stream:
            sum_channel += 32
        merged_channel = sum_channel // 3 if input_three_stream else sum_channel // 2
        self.conv0 = nn.Conv2d(sum_channel, merged_channel, 1)
        self.ms = MultiScaleAtrousConv2d(merged_channel, merged_channel // 2)
        self.pa = SpatialAttention()
        self.sa = SpectralAttention(feature1_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: concatenated features
        con0 = self.conv0(x)
        conms = self.ms(con0)
        spatial_att = self.pa(x[:, :self.feature0_channel, :, :])
        spectral_att = self.sa(x[:, self.feature0_channel:self.feature0_channel + self.feature1_channel, :, :])
        spatial_supple = self.relu(conms * spectral_att)
        spectral_supple = self.relu(conms * spatial_att)
        conms = self.relu(conms)
        return conms, spatial_supple, spectral_supple


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, relu=True):
        super(BasicConv, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn0 = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.1, affine=True)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.1, affine=True)
        self.relu_flag = relu
        if self.relu_flag:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        if self.relu_flag:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        return x


class MSMFFNet(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.fusion0 = MSMFFModule(32, 32)
        self.fusion1 = MSMFFModule(32, 32, True)
        self.fusion2 = MSMFFModule(32, 32, True)
        self.pconv0 = BasicConv(64, 32, 3, 1, 1)
        self.pconv1 = BasicConv(64, 32, 3, 1, 1)
        self.pconv2 = BasicConv(64, 32, 3, 1, 1)
        self.sconv0 = BasicConv(64, 32, 3, 1, 1)
        self.sconv1 = BasicConv(64, 32, 3, 1, 1)
        self.sconv2 = BasicConv(64, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def mbi(self, img_arr):
        ret_lis = []
        bsz = img_arr.shape[0]
        for ii in range(bsz):
            img = img_arr[ii, :, :, :].numpy()
            # print(img.shape)

            gray = np.max(img, 0)
            gray = np.pad(gray, ((self.s_min, self.s_min), (self.s_min, self.s_min)), 'constant',
                          constant_values=(0, 0))

            mp_lis = []
            for i in range(self.s_min, self.s_max, 2 * self.delta_s):
                se_inter = square(i)
                se_inter[:int((i - 1) / 2), :] = 0
                se_inter[int((i - 1) / 2) + 1:, :] = 0
                for angle in range(0, 180, 45):
                    se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
                    mp = white_tophat(gray, selem=se_inter)
                    mp_lis.append(mp)

            dmp_lis = []
            for i in range(self.delta_s, len(mp_lis)):
                dmp = np.absolute(mp_lis[i] - mp_lis[i - self.delta_s])
                dmp_lis.append(dmp)

            mbi = np.sum(dmp_lis, axis=0) / (4 * ((self.s_max - self.s_min) / self.delta_s + 1))
            mbi = mbi[self.s_min: mbi.shape[0] - self.s_min, self.s_min: mbi.shape[1] - self.s_min]
            # print(np.unique(mbi))

            # bug existed here
            thres = threshold_otsu(mbi)
            gt_idx = mbi > thres
            le_idx = mbi <= thres
            mbi[gt_idx] = 1
            mbi[le_idx] = 0
            # print(np.unique(mbi))

            ret_lis.append(mbi)

        ret_lis = torch.tensor(np.array(ret_lis)).unsqueeze(1).to(torch.float)
        return ret_lis

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        # self.priori.append(self.mbi(x0))
        # self.priori.append(self.mbi(x1))

        priori0 = self.mbi((x0))
        # print(priori0.dtype)
        # print(priori0.shape)
        priori0 = self.encoder0(priori0)
        # print(priori0.shape)
        priori0 = self.decoder0(priori0)
        # print(priori0.shape)
        x0 = x0 * (priori0 + 1)
        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)

        priori1 = self.mbi(x1)
        priori1 = self.encoder1(priori1)
        priori1 = self.decoder1(priori1)
        x1 = x1 * (priori1 + 1)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, po0, sp0 = self.fusion0(torch.cat([x0, x1], dim=1))
        # print(cross0.shape)
        # print(po0.shape)
        # print(sp0.shape)
        # print(x0.shape)
        # print(x1.shape)
        p0 = self.pconv0(torch.cat([x0, po0], dim=1))
        s0 = self.sconv0(torch.cat([x1, sp0], dim=1))
        cross1, po1, sp1 = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(torch.cat([p0, po1], dim=1))
        s1 = self.sconv1(torch.cat([s0, sp1], dim=1))
        cross2, po2, sp2 = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(torch.cat([p1, po2], dim=1))
        s2 = self.sconv2(torch.cat([s1, sp2], dim=1))

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        if self.training:
            return sum_reduced_features,priori0, priori1
        else:
            return sum_reduced_features

        # decoding cross2, x0, x1


class MSMFFNetwithoutPrior(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.fusion0 = MSMFFModule(32, 32)
        self.fusion1 = MSMFFModule(32, 32, True)
        self.fusion2 = MSMFFModule(32, 32, True)
        self.pconv0 = BasicConv(64, 32, 3, 1, 1)
        self.pconv1 = BasicConv(64, 32, 3, 1, 1)
        self.pconv2 = BasicConv(64, 32, 3, 1, 1)
        self.sconv0 = BasicConv(64, 32, 3, 1, 1)
        self.sconv1 = BasicConv(64, 32, 3, 1, 1)
        self.sconv2 = BasicConv(64, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, po0, sp0 = self.fusion0(torch.cat([x0, x1], dim=1))
        p0 = self.pconv0(torch.cat([x0, po0], dim=1))
        s0 = self.sconv0(torch.cat([x1, sp0], dim=1))
        cross1, po1, sp1 = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(torch.cat([p0, po1], dim=1))
        s1 = self.sconv1(torch.cat([s0, sp1], dim=1))
        cross2, po2, sp2 = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(torch.cat([p1, po2], dim=1))
        s2 = self.sconv2(torch.cat([s1, sp2], dim=1))

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        return sum_reduced_features


class MSMFFNetwithoutModule(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.fusion0 = BasicConv(32, 32, 3, 1, 1)
        self.fusion1 = BasicConv(32, 32, 3, 1, 1)
        self.fusion2 = BasicConv(32, 32, 3, 1, 1)
        self.pconv0 = BasicConv(64, 32, 3, 1, 1)
        self.pconv1 = BasicConv(64, 32, 3, 1, 1)
        self.pconv2 = BasicConv(64, 32, 3, 1, 1)
        self.sconv0 = BasicConv(64, 32, 3, 1, 1)
        self.sconv1 = BasicConv(64, 32, 3, 1, 1)
        self.sconv2 = BasicConv(64, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def mbi(self, img_arr):
        ret_lis = []
        bsz = img_arr.shape[0]
        for ii in range(bsz):
            img = img_arr[ii, :, :, :].cpu().detach().numpy()
            # print(img.shape)

            gray = np.max(img, 0)
            gray = np.pad(gray, ((self.s_min, self.s_min), (self.s_min, self.s_min)), 'constant',
                          constant_values=(0, 0))

            mp_lis = []
            for i in range(self.s_min, self.s_max, 2 * self.delta_s):
                se_inter = square(i)
                se_inter[:int((i - 1) / 2), :] = 0
                se_inter[int((i - 1) / 2) + 1:, :] = 0
                for angle in range(0, 180, 45):
                    se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
                    mp = white_tophat(gray, selem=se_inter)
                    mp_lis.append(mp)

            dmp_lis = []
            for i in range(self.delta_s, len(mp_lis)):
                dmp = np.absolute(mp_lis[i] - mp_lis[i - self.delta_s])
                dmp_lis.append(dmp)

            mbi = np.sum(dmp_lis, axis=0) / (4 * ((self.s_max - self.s_min) / self.delta_s + 1))
            mbi = mbi[self.s_min: mbi.shape[0] - self.s_min, self.s_min: mbi.shape[1] - self.s_min]
            # print(np.unique(mbi))

            # bug existed here
            thres = threshold_otsu(mbi)
            gt_idx = mbi > thres
            le_idx = mbi <= thres
            mbi[gt_idx] = 1
            mbi[le_idx] = 0
            # print(np.unique(mbi))

            ret_lis.append(mbi)

        ret_lis = torch.tensor(np.array(ret_lis)).unsqueeze(1).to(torch.float)
        return ret_lis.to('cuda')

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        priori0 = self.mbi((x0))
        priori0 = self.encoder0(priori0)
        priori0 = self.decoder0(priori0)
        x0 = x0 * (priori0 + 1)
        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)

        priori1 = self.mbi(x1)
        priori1 = self.encoder1(priori1)
        priori1 = self.decoder1(priori1)
        x1 = x1 * (priori1 + 1)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, po0, sp0 = self.fusion0(torch.cat([x0, x1], dim=1))
        p0 = self.pconv0(torch.cat([x0, po0], dim=1))
        s0 = self.sconv0(torch.cat([x1, sp0], dim=1))
        cross1, po1, sp1 = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(torch.cat([p0, po1], dim=1))
        s1 = self.sconv1(torch.cat([s0, sp1], dim=1))
        cross2, po2, sp2 = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(torch.cat([p1, po2], dim=1))
        s2 = self.sconv2(torch.cat([s1, sp2], dim=1))

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        if self.training:
            return sum_reduced_features, priori0, priori1
        else:
            return sum_reduced_features


class MSMFFNetwithoutInter(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.fusion0 = MSMFFModule(32, 32)
        self.fusion1 = MSMFFModule(32, 32, True)
        self.fusion2 = MSMFFModule(32, 32, True)
        self.pconv0 = BasicConv(32, 32, 3, 1, 1)
        self.pconv1 = BasicConv(32, 32, 3, 1, 1)
        self.pconv2 = BasicConv(32, 32, 3, 1, 1)
        self.sconv0 = BasicConv(32, 32, 3, 1, 1)
        self.sconv1 = BasicConv(32, 32, 3, 1, 1)
        self.sconv2 = BasicConv(32, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def mbi(self, img_arr):
        ret_lis = []
        bsz = img_arr.shape[0]
        for ii in range(bsz):
            img = img_arr[ii, :, :, :].cpu().detach().numpy()
            # print(img.shape)

            gray = np.max(img, 0)
            gray = np.pad(gray, ((self.s_min, self.s_min), (self.s_min, self.s_min)), 'constant',
                          constant_values=(0, 0))

            mp_lis = []
            for i in range(self.s_min, self.s_max, 2 * self.delta_s):
                se_inter = square(i)
                se_inter[:int((i - 1) / 2), :] = 0
                se_inter[int((i - 1) / 2) + 1:, :] = 0
                for angle in range(0, 180, 45):
                    se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
                    mp = white_tophat(gray, selem=se_inter)
                    mp_lis.append(mp)

            dmp_lis = []
            for i in range(self.delta_s, len(mp_lis)):
                dmp = np.absolute(mp_lis[i] - mp_lis[i - self.delta_s])
                dmp_lis.append(dmp)

            mbi = np.sum(dmp_lis, axis=0) / (4 * ((self.s_max - self.s_min) / self.delta_s + 1))
            mbi = mbi[self.s_min: mbi.shape[0] - self.s_min, self.s_min: mbi.shape[1] - self.s_min]
            # print(np.unique(mbi))

            # bug existed here
            thres = threshold_otsu(mbi)
            gt_idx = mbi > thres
            le_idx = mbi <= thres
            mbi[gt_idx] = 1
            mbi[le_idx] = 0
            # print(np.unique(mbi))

            ret_lis.append(mbi)

        ret_lis = torch.tensor(np.array(ret_lis)).unsqueeze(1).to(torch.float)
        return ret_lis.to('cuda')

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        # self.priori.append(self.mbi(x0))
        # self.priori.append(self.mbi(x1))

        priori0 = self.mbi((x0))
        # print(priori0.dtype)
        # print(priori0.shape)
        priori0 = self.encoder0(priori0)
        # print(priori0.shape)
        priori0 = self.decoder0(priori0)
        # print(priori0.shape)
        x0 = x0 * (priori0 + 1)
        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)

        priori1 = self.mbi(x1)
        priori1 = self.encoder1(priori1)
        priori1 = self.decoder1(priori1)
        x1 = x1 * (priori1 + 1)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, _, _ = self.fusion0(torch.cat([x0, x1], dim=1))
        # print(cross0.shape)
        # print(po0.shape)
        # print(sp0.shape)
        # print(x0.shape)
        # print(x1.shape)
        p0 = self.pconv0(x0)
        s0 = self.sconv0(x1)
        cross1, _, _ = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(p0)
        s1 = self.sconv1(s0)
        cross2, _, _ = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(p1)
        s2 = self.sconv2(s1)

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        if self.training:
            return sum_reduced_features, priori0, priori1
        else:
            return sum_reduced_features

        # decoding cross2, x0, x1


class MSMFFNetonlyPrior(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.pconv0 = BasicConv(32, 32, 3, 1, 1)
        self.pconv1 = BasicConv(32, 32, 3, 1, 1)
        self.pconv2 = BasicConv(32, 32, 3, 1, 1)
        self.sconv0 = BasicConv(32, 32, 3, 1, 1)
        self.sconv1 = BasicConv(32, 32, 3, 1, 1)
        self.sconv2 = BasicConv(32, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def mbi(self, img_arr):
        ret_lis = []
        bsz = img_arr.shape[0]
        for ii in range(bsz):
            img = img_arr[ii, :, :, :].cpu().detach().numpy()
            # print(img.shape)

            gray = np.max(img, 0)
            gray = np.pad(gray, ((self.s_min, self.s_min), (self.s_min, self.s_min)), 'constant',
                          constant_values=(0, 0))

            mp_lis = []
            for i in range(self.s_min, self.s_max, 2 * self.delta_s):
                se_inter = square(i)
                se_inter[:int((i - 1) / 2), :] = 0
                se_inter[int((i - 1) / 2) + 1:, :] = 0
                for angle in range(0, 180, 45):
                    se_inter = rotate(se_inter, angle, order=0, preserve_range=True).astype('int16')
                    mp = white_tophat(gray, selem=se_inter)
                    mp_lis.append(mp)

            dmp_lis = []
            for i in range(self.delta_s, len(mp_lis)):
                dmp = np.absolute(mp_lis[i] - mp_lis[i - self.delta_s])
                dmp_lis.append(dmp)

            mbi = np.sum(dmp_lis, axis=0) / (4 * ((self.s_max - self.s_min) / self.delta_s + 1))
            mbi = mbi[self.s_min: mbi.shape[0] - self.s_min, self.s_min: mbi.shape[1] - self.s_min]
            # print(np.unique(mbi))

            # bug existed here
            thres = threshold_otsu(mbi)
            gt_idx = mbi > thres
            le_idx = mbi <= thres
            mbi[gt_idx] = 1
            mbi[le_idx] = 0
            # print(np.unique(mbi))

            ret_lis.append(mbi)

        ret_lis = torch.tensor(np.array(ret_lis)).unsqueeze(1).to(torch.float)
        return ret_lis.to('cuda')

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        priori0 = self.mbi((x0))
        priori0 = self.encoder0(priori0)
        priori0 = self.decoder0(priori0)
        x0 = x0 * (priori0 + 1)
        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)

        priori1 = self.mbi(x1)
        priori1 = self.encoder1(priori1)
        priori1 = self.decoder1(priori1)
        x1 = x1 * (priori1 + 1)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        p0 = self.pconv0(x0)
        s0 = self.sconv0(x1)
        p1 = self.pconv1(p0)
        s1 = self.sconv1(s0)
        p2 = self.pconv2(p1)
        s2 = self.sconv2(s1)

        sum_reduced_features = self.rcon0(p2 + s2)

        if self.training:
            return sum_reduced_features, priori0, priori1
        else:
            return sum_reduced_features


class MSMFFNetonlyModule(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.fusion0 = MSMFFModule(32, 32)
        self.fusion1 = MSMFFModule(32, 32, True)
        self.fusion2 = MSMFFModule(32, 32, True)
        self.pconv0 = BasicConv(32, 32, 3, 1, 1)
        self.pconv1 = BasicConv(32, 32, 3, 1, 1)
        self.pconv2 = BasicConv(32, 32, 3, 1, 1)
        self.sconv0 = BasicConv(32, 32, 3, 1, 1)
        self.sconv1 = BasicConv(32, 32, 3, 1, 1)
        self.sconv2 = BasicConv(32, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, _, _ = self.fusion0(torch.cat([x0, x1], dim=1))
        p0 = self.pconv0(x0)
        s0 = self.sconv0(x1)
        cross1, _, _ = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(p0)
        s1 = self.sconv1(s0)
        cross2, _, _ = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(p1)
        s2 = self.sconv2(s1)

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        return sum_reduced_features


class MSMFFNetonlyInter(nn.Module):
    def __init__(self, img0_size, img1_size):
        super(MSMFFNet, self).__init__()
        self.s_min = 3
        self.s_max = 20
        self.delta_s = 1
        # self.priori = []
        self.img0_size = img0_size
        self.img1_size = img1_size
        self.encoder0 = _EncoderBlock(1, 32)
        self.decoder0 = _DecoderBlock(32, 64, 1)
        self.encoder1 = _EncoderBlock(1, 32)
        self.decoder1 = _DecoderBlock(32, 64, 1)
        self.pre_conv00 = BasicConv(self.img0_size, 32, 3, 1, 1, False)
        self.pre_conv01 = BasicConv(32, 32, 3, 1, 1, False)
        self.pre_conv10 = BasicConv(self.img1_size, 32, 1, 1, 0, False)
        self.pre_conv11 = BasicConv(32, 32, 1, 1, 0, False)
        self.fusion0 = BasicConv(32, 32, 3, 1, 1)
        self.fusion1 = BasicConv(32, 32, 3, 1, 1)
        self.fusion2 = BasicConv(32, 32, 3, 1, 1)
        self.pconv0 = BasicConv(64, 32, 3, 1, 1)
        self.pconv1 = BasicConv(64, 32, 3, 1, 1)
        self.pconv2 = BasicConv(64, 32, 3, 1, 1)
        self.sconv0 = BasicConv(64, 32, 3, 1, 1)
        self.sconv1 = BasicConv(64, 32, 3, 1, 1)
        self.sconv2 = BasicConv(64, 32, 3, 1, 1)
        self.rcon0 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x0 = x[:, :self.img0_size, :, :]
        x1 = x[:, self.img0_size:, :, :]

        x0 = self.pre_conv00(x0)
        x0 = self.pre_conv01(x0)
        x1 = self.pre_conv10(x1)
        x1 = self.pre_conv11(x1)

        cross0, po0, sp0 = self.fusion0(torch.cat([x0, x1], dim=1))
        p0 = self.pconv0(torch.cat([x0, po0], dim=1))
        s0 = self.sconv0(torch.cat([x1, sp0], dim=1))
        cross1, po1, sp1 = self.fusion1(torch.cat([p0, s0, cross0], dim=1))
        p1 = self.pconv1(torch.cat([p0, po1], dim=1))
        s1 = self.sconv1(torch.cat([s0, sp1], dim=1))
        cross2, po2, sp2 = self.fusion2(torch.cat([p1, s1, cross1], dim=1))
        p2 = self.pconv2(torch.cat([p1, po2], dim=1))
        s2 = self.sconv2(torch.cat([s1, sp2], dim=1))

        sum_reduced_features = self.rcon0(p2 + s2 + cross2)

        return sum_reduced_features
