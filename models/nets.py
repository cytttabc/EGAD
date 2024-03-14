import torch
import torch.nn as nn
import torch.nn.functional as F
from layers4 import ResBlock
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, dim):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(dim * 2, dim, 1, padding=0)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        # res = self.conv2(res)
        res1 = self.calayer(res)
        res2 = self.palayer(res1)
        res3=torch.cat((res1,res2),dim=1)
        res3=self.conv3(res3)
        res3 += x
        return res3




class EGADBlocks(nn.Module):
    def __init__(self, dim):
        super(EGADBlocks, self).__init__()

        self.c1 =Block(dim)
        self.c2 = Block(dim)
        self.c3 = ResBlock(dim,dim)



    def forward(self, x):
        x= self.c1(x)
        x = self.c2(x)
        # print('0', x.shape)
        x = self.c3(x)
        # print('1', x.shape)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class EEM(nn.Module):
    def __init__(self, out_plane):
        super(EEM, self).__init__()

        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class EEM2(nn.Module):
    def __init__(self, out_plane):
        super(EEM2, self).__init__()

        self.main = nn.Sequential(

            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class ConvsOut(nn.Module):
    def __init__(self, dim):
        super(ConvsOut, self).__init__()

        self.main = nn.Sequential(

            BasicConv(dim, dim // 2, kernel_size=1, relu=True, stride=1),
            BasicConv(dim // 2, dim // 3, kernel_size=4, relu=True, stride=2, transpose=True),

            BasicConv(dim // 3, 3, kernel_size=4, relu=True, stride=2, transpose=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class CAT(nn.Module):
    def __init__(self, channel):
        super(CAT, self).__init__()

        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class Dehaze(nn.Module):
 def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24]):

        super(Dehaze, self).__init__()

        self.patch_size = 4


        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = EGADBlocks( dim=embed_dims[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = EGADBlocks( dim=embed_dims[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = EGADBlocks( dim=embed_dims[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = EGADBlocks( dim=embed_dims[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = EGADBlocks(dim=embed_dims[4])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        self.get_g_nopadding = Get_gradient_nopadding()

        self.conv_out= nn.Conv2d(6, 3, kernel_size=3, padding=1,bias=False)
        self.EEM1 = EEM(embed_dims[0])
        self.EEM2 = EEM(embed_dims[1])
        self.EEM3 = EEM(embed_dims[2])
        self.ConvsOut = ConvsOut(embed_dims[2])
        self.CAT=CAT(embed_dims[0])
        self.CAT1 = CAT(embed_dims[1])
        self.CAT2 = CAT(embed_dims[2])


 def check_image_size(self, x):
        _, _, h, w = x.size()
        patch_size = self.patch_size
        mod_pad_h = (patch_size - h % patch_size) % patch_size
        mod_pad_w = (patch_size - w % patch_size) % patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

 def forward(self, x):
        #edge

        x_grad = self.get_g_nopadding(x)
        c1=self.EEM1(x_grad)
        x2 = F.interpolate(x_grad, scale_factor=0.5)#torch.Size([2, 24, 32, 32])
        c2=self.EEM2(x2)
        x3 = F.interpolate(x2, scale_factor=0.5)
        c3 = self.EEM3(x3)

        feature = self.ConvsOut(c3)
        feature=torch.cat((feature,x_grad),dim=1)
        feature=self.conv_out(feature)
        #dehaze
        H, W = x.shape[2:]
        x_1 = self.check_image_size(x)
        x = self.patch_embed(x)
        x = self.layer1(x)
        x = self.CAT(x, c1)
        skip1 = x
        x = self.patch_merge1(x)
        x = self.layer2(x)
        x = self.CAT1(x, c2)
        skip2 = x

        x = self.patch_merge2(x)


        x = self.layer3(x)
        x = self.CAT2(x, c3)
        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        feat= self.patch_unembed(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x_1 = K * x_1 - B + x_1
        x_1 = x_1[:, :, :H, :W]
        return x_1,feature
# if __name__ == "__main__":
#
#     net=Dehaze()
#     print(net)
# from torchsummary import summary
#
# summary(net, input_size=(3, 64,64), batch_size=1)
