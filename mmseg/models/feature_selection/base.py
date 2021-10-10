import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmseg.ops import resize, Upsample
from ..builder import NECKS
from ..builder import FEATURE_SELECTION
import torch



class LeakyUnit(nn.Module):
    def __init__(self, n_fi, n_fo):
        super(LeakyUnit, self).__init__()
        self.W_r = nn.Conv2d(n_fi + n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.W = nn.Conv2d(n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.U = nn.Conv2d(n_fi, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.W_z = nn.Conv2d(n_fi + n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, f_m, f_n):
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n

        return f_m_out, r_mn, z_mn


class LeakyUnit_adaptive(nn.Module):
    def __init__(self, n_fi, n_fo, co_ch, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')): #dict(type='ReLU')
        super(LeakyUnit_adaptive, self).__init__()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        # self.co_ch = co_ch
        self.fit_m = ConvModule(n_fo, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        self.fit_n = ConvModule(n_fi, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        self.W_r = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.W = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.U = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.W_z = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.sigma = nn.Sigmoid()
        self.out = ConvModule(co_ch, n_fo, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )

    def forward(self, f_m, f_n):

        f_m = self.fit_m(f_m)
        f_n = self.fit_n(f_n)
        prev_shape = f_m.shape[2:]
        f_n = F.interpolate(f_n, size=prev_shape)
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        f_m_out = self.out(f_m_out)
        return f_m_out, r_mn, z_mn

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

class Cat_adaptive(nn.Module):
    def __init__(self, n_fi, n_fo, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(Cat_adaptive, self).__init__()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        # self.co_ch = co_ch

        self.out = ConvModule(n_fi + n_fo, n_fo, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )

        '''ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)'''

    def forward(self, f_m, f_n):

        prev_shape = f_m.shape[2:]
        f_n = F.interpolate(f_n, size=prev_shape)
        f_m_out = self.out(torch.cat((f_m, f_n), dim=1))

        return f_m_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpSampleFeatureFusionModel(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 bilinear=True,
                 align_corners=False
                 ):
        super(UpSampleFeatureFusionModel, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)  # 'nearest'/bilinear
            # self.conv1 = FusionLayer(in_channels,in_channels)
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv1 = FusionLayer(in_channels, in_channels)
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, higher_features, lower_features):
        higher_features = self.up(higher_features)
        # input is CHW
        diffY = torch.tensor([lower_features.size()[2] - higher_features.size()[2]])
        diffX = torch.tensor([lower_features.size()[3] - higher_features.size()[3]])
        higher_features = F.pad(higher_features, [diffX // 2, diffX - diffX // 2,
                                                  diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        fused_features = torch.cat([lower_features, higher_features], dim=1)
        # x = self.conv1(x)
        return self.conv(fused_features)

@FEATURE_SELECTION.register_module()
class Cat_wo_sf(nn.Module):
    def __init__(self,
                 in_channels,
                 inu_channels,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(Cat_wo_sf, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.inu_channels = inu_channels
        self.Cat_u_fpn = nn.ModuleList()
        self.Cat_fpn_u = nn.ModuleList()
        # self.LeakyUnit_adaptive=LeakyUnit_adaptive
        '''
        only the last 3 layers of FPN_seg and the first 3 layers of UNet_gan implement soft selection
        '''
        for i in range(0, len(inu_channels)):

            self.Cat_u_fpn.append(Cat_adaptive(self.inu_channels[i],
                                               self.in_channels[i]))
            self.Cat_fpn_u.append(Cat_adaptive(self.in_channels[i],
                                               self.inu_channels[i]))




    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs_a, inputs_b):
        # print(len(inputs_a[0]))
        # assert len(inputs_a[0]) == len(self.in_channels)

        out_a = []
        out_b = []

        for i in range(0, len(self.inu_channels)):
            a_feat_hat = self.Cat_u_fpn[i](inputs_a[i], inputs_b[i])
            b_feat_hat = self.Cat_fpn_u[i](inputs_b[i], inputs_a[i])
            # laterals_u.append(lateral_u)
            out_a.append(a_feat_hat)
            out_b.append(b_feat_hat)
        out_a.append(inputs_a[3])

        # if len(inputs_a[0]) == len(inputs_a[0]):
        return tuple(out_a), tuple(out_b)
        # else if len(inputs_a[0]) == len(inputs_a[0])
        # build outputs
        # part 1: from original levels
        # part 2: add extra levels




@FEATURE_SELECTION.register_module()
class Basic_fs(nn.Module):

    def __init__(self,
                 in_channels,
                 inu_channels,
                 adaptive_channel=256,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(Basic_fs, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.inu_channels = inu_channels
        self.leakyunit_u_fpn = nn.ModuleList()
        self.leakyunit_fpn_u = nn.ModuleList()
        self.adaptive_channel = adaptive_channel
        # self.LeakyUnit_adaptive=LeakyUnit_adaptive
        '''
        only the last 3 layers of FPN_seg and the first 3 layers of UNet_gan implement soft selection
        '''
        for i in range(0, len(inu_channels)):

            self.leakyunit_u_fpn.append(LeakyUnit_adaptive(self.inu_channels[i],
                                                           self.in_channels[i], self.adaptive_channel))
            self.leakyunit_fpn_u.append(LeakyUnit_adaptive(self.in_channels[i],
                                                           self.inu_channels[i], self.adaptive_channel))




    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs_a, inputs_b):

        out_a = []
        out_b = []

        for i in range(0, len(self.inu_channels)):
            a_feat_hat, _, _ = self.leakyunit_u_fpn[i](inputs_a[i], inputs_b[i])
            b_feat_hat, _, _ = self.leakyunit_fpn_u[i](inputs_b[i], inputs_a[i])

            out_a.append(a_feat_hat)
            out_b.append(b_feat_hat)
        out_a.append(inputs_a[3])

        return tuple(out_a), tuple(out_b)



