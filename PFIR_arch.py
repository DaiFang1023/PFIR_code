import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.module import *



class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)

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
            nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1, bias=False),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim * 2, kernel_size=2, stride=2))

    def forward(self, x):
        x = self.proj(x)
        return x





class MSLKP(nn.Module):
    def __init__(self, dim, lks1, lks2, sks, groups):
        super().__init__()
        # 通用设置
        self.dim = dim
        self.sks = sks
        self.groups = groups

        # 初始降维
        self.cv1 = Conv2d_BN(dim, dim)  # 先降维

        # 分支1：kernel size = lks1
        self.cv2_1 = Conv2d_BN(dim // 2, dim // 2, ks=lks1, pad=(lks1 - 1) // 2, groups=dim // 2)
        self.cv3_1 = Conv2d_BN(dim // 2, dim // 2)

        # 分支2：kernel size = lks2
        self.cv2_2 = Conv2d_BN(dim // 2, dim // 2, ks=lks2, pad=(lks2 - 1), dilation=2,groups=dim // 2)
        self.cv3_2 = Conv2d_BN(dim // 2, dim // 2)

        # 合并通道后进行卷积生成注意力权重
        self.cv4 = nn.Conv2d(dim , sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.act = nn.GELU()
        self.act1= nn.Tanh()
    def forward(self, x):
        x = self.act(self.cv1(x))  # b, dim//2, h, w

        # Split into two parts along channel dimension
        x1, x2 = torch.chunk(x, 2, dim=1)  # Each: b, dim//4, h, w

        # Process each part
        x1 = self.act(self.cv3_1(self.cv2_1(x1)))
        x2 = self.act(self.cv3_2(self.cv2_2(x2)))

        # Concatenate
        x = torch.cat([x1, x2], dim=1)  # b, dim//2, h, w

        # Output attention weights
        w = self.act1(self.norm(self.cv4(x)))  # b, sks^2 * dim // groups, h, w
        b, _, h, width = w.shape
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class MSLSConv(nn.Module):
    def __init__(self, dim,lks1=5,lks2=9,sks=3,groups=8):
        super(MSLSConv, self).__init__()
        self.lkp = MSLKP(dim, lks1=lks1,lks2=lks2, sks=sks, groups=groups)
        self.ska = SKA()


    def forward(self, x):
        return self.ska(x, self.lkp(x))+x

class PFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(PFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim
        # PW first or DW first?

        self.conv1_3 = nn.Sequential(
            MSLSConv(self.dim_sp,lks1=5,lks2=9,groups=8),
        )

        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.conv1_3(x)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x

class TokenMixer_SPA(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_SPA, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x



class TokenMixer_FFT(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_FFT, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.FFC = G_FFTAttention(self.dim,patch_size=32)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)

        return x

class SFFM(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(SFFM, self).__init__()
        self.dim = dim
        self.fe_local = TokenMixer_SPA(dim=self.dim,)
        # self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)
        self.fe_gloal = TokenMixer_FFT(dim=self.dim)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 1),
        )



    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.fe_local(x[0])
        x_gloal = self.fe_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
    ):
        super(Block, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.sffm = SFFM(dim=self.dim)
        self.pffn = PFFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.sffm(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.pffn(x)
        x = x * self.gamma + copy

        return x



# need drop_path?
class Stage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        super(Stage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                Block(
                    dim=in_channels,
                    norm_layer=nn.BatchNorm2d,
                )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output


class Backbone(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2], embed_kernel_size=3,):
        super(Backbone, self).__init__()

        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = Stage(depth=depth[0], in_channels=embed_dim[0],)
        self.skip1 = nn.Conv2d(embed_dim[1], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],)
        self.layer2 = Stage(depth=depth[1], in_channels=embed_dim[1],)
        self.skip2 = nn.Conv2d(embed_dim[2], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],)
        self.layer3 = Stage(depth=depth[2], in_channels=embed_dim[2],)
        self.upsample3 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2],
                                                   out_dim=embed_dim[3])
        self.layer8 = Stage(depth=depth[3], in_channels=embed_dim[3],)
        self.upsample4 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.layer9 = Stage(depth=depth[4], in_channels=embed_dim[4],)
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)

    def forward(self, x):
        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.upsample3(x)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer8(x)
        x = self.upsample4(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer9(x)
        x = self.patch_unembed(x)
        x = copy0 + x
        return x


@ARCH_REGISTRY.register()
def SFHformer_l():
    return Backbone(
        patch_size=1,
        embed_dim=[32, 64, 128, 64, 32],
        depth=[6, 6, 10, 6, 6],
        embed_kernel_size=3
    )
