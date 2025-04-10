import torch

# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

# from pdb import set_trace as stx
import numbers

from einops import rearrange
from timm.models.layers import trunc_normal_, get_padding


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class InceptionDWConv2d(nn.Module):
    """ Inception depthwise convolution
    """

    def __init__(
            self,
            in_chs,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=0.125,
            dilation=1,
    ):
        super().__init__()

        gc = int(in_chs * branch_ratio)  # channel numbers of a convolution branch
        square_padding = get_padding(square_kernel_size, dilation=dilation)
        band_padding = get_padding(band_kernel_size, dilation=dilation)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size,
            padding=square_padding, dilation=dilation, groups=gc)
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size),
            padding=(0, band_padding), dilation=(1, dilation), groups=gc)
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1),
            padding=(band_padding, 0), dilation=(dilation, 1), groups=gc)
        self.split_indexes = (in_chs - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((
            x_id,
            self.dwconv_hw(x_hw),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
            ), dim=1,
        )
    
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, shuffle):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = ShuffleConv1x1(dim, hidden_features * 2) if shuffle else nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.gate = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = F.gelu(x1) * self.gate(x2)
        x = self.project_out(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.groups
        Cg = C // G  # Channels per group

        # Reshape into (N, G, Cg, H, W), permute to shuffle, then reshape back
        x = x.view(N, G, Cg, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(N, C, H, W)
        return x
    
class ShuffleConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        self.group_conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False)
        self.shuffle = ChannelShuffle(groups)

    def forward(self, x):
        x = self.shuffle(x)
        x = self.group_conv(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.q_dwconv = InceptionDWConv2d(dim, branch_ratio=0.125)
        self.k_dwconv = InceptionDWConv2d(dim, branch_ratio=0.125)
        self.v_dwconv = InceptionDWConv2d(dim, branch_ratio=0.125)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.q_dwconv(q)
        k = self.k_dwconv(k)
        v = self.v_dwconv(v)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, shuffle):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, shuffle)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class LiteRAWFormer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=8,
        transposed_attn_heads=8,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias", 
    ):
        super().__init__()
        shuffle_list = [1, 1, 1, 1, 1, 1, 0, 0]
        self.conv_in = nn.Conv2d(inp_channels, int(dim * 2**2), 3, 1, 1)
        self.decoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=transposed_attn_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    shuffle=shuffle_list[i]
                )
                for i in range(num_blocks)
            ]
        )
        self.up = Upsample(int(dim * 2**2), out_channels)
        self.skip = Upsample(inp_channels, out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, inp_img, noise_emb=None):
        return self.up(self.decoder(self.conv_in(inp_img))) + self.skip(inp_img)
