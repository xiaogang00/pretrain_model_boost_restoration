import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2



from models.archs.transformer.Models import Encoder_patch66

from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class refinement2(nn.Module):
    def __init__(self, num_blocks=[4,6,6,8],num_refinement_blocks = 4,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(refinement2, self).__init__()

        nf = 16
        self.conv_last2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.pool=nn.AdaptiveAvgPool2d(1)
        kh=3
        self.kh=kh
        kw=kh
        
        self.mapp_space1=nn.Linear(512, 256,bias=False)
        self.mapp_space2=nn.Linear(256, nf*4,bias=False)
        self.mapp_space11=nn.Linear(512, 256,bias=False)
        self.mapp_space22=nn.Linear(256, nf*4,bias=False)
        
        self.mapp_position1=nn.Conv2d(2, nf, 3, 1, 1, bias=False)
        self.mapp_position2=nn.Conv2d(nf, nf, 1, 1, 0, bias=False)
        self.mapp_position11=nn.Conv2d(2, nf, 3, 1, 1, bias=False)
        self.mapp_position22=nn.Conv2d(nf, nf, 1, 1, 0, bias=False)

        self.mapping11=nn.Linear(nf*4+nf, nf*1,bias=False)
        self.mapping12=nn.Linear(nf*1, nf*1,bias=False)
        self.mapp11=nn.Conv2d(nf*5+nf, nf*kh*kw, 3,1,1,bias=False)
        self.mapp12=nn.Conv2d(nf*kh*kw, nf*kh*kw,3,1,1,bias=False)
        self.mapp1=nn.Conv2d(nf, 1, 3, 1, 1, bias=False)
        self.refine11=nn.Conv2d(nf*2, nf, 1, 1, 0, bias=False)
        
        self.r1=nn.Conv2d(3+3, nf, 3, 1, 1, bias=False)
        self.r2=nn.Conv2d(nf, nf, 3, 2, 1, bias=False)
        self.r22=nn.Conv2d(nf, nf, 3, 2, 1, bias=False)
        
        self.r1_decoder=nn.Conv2d(nf, nf*4, 3, 1, 1, bias=False)
        self.r2_decoder=nn.Conv2d(nf, nf*4, 3, 1, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.r1norm=nn.InstanceNorm2d(nf, affine=True)
        self.r2norm=nn.InstanceNorm2d(nf, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.r1_mask=nn.Conv2d(nf+nf*5, nf, 3, 1, 1, bias=False)
        self.r2_mask=nn.Conv2d(nf, 1, 3, 1, 1, bias=False)
        
        self.transformer = nn.Sequential(*[TransformerBlock(dim=nf, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.resnet=ResidualBlock_noBN(nf)

    def forward(self, clip_condition_list, out_dec_level1, inp_img):
        feat_clip_normal=clip_condition_list[0].float()
        
        out_noise2=self.r22(self.r2norm(self.r2(self.lrelu(self.r1norm(self.r1(torch.cat([out_dec_level1,inp_img],dim=1)))))))
        batch_size=out_noise2.shape[0]
        height=out_noise2.shape[2]
        width=out_noise2.shape[3]

        ##### position embedding
        xs = torch.linspace(-1, 1, steps=height)
        ys = torch.linspace(-1, 1, steps=width)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        grid_x=grid_x.unsqueeze(dim=0).unsqueeze(dim=0)
        grid_y=grid_y.unsqueeze(dim=0).unsqueeze(dim=0)
        grid=torch.cat([grid_x,grid_y],dim=1).repeat(batch_size,1,1,1).to(out_noise2.device)
        position_embedding1=self.lrelu(self.mapp_position1(grid))
        position_embedding1=self.mapp_position2(position_embedding1)
        

        ### fusion1
        feat_clip_normal1=self.lrelu(self.mapp_space1(feat_clip_normal))
        feat_clip_normal1=self.mapp_space2(feat_clip_normal1)
        
        mask_input1=feat_clip_normal1.view(batch_size,-1,1,1).repeat(1,1,height,width)
        mask_input=torch.cat([out_noise2, mask_input1, position_embedding1],dim=1)
        mask_input=self.r2_mask(self.lrelu(self.r1_mask(mask_input)))
        mask_input=nn.Sigmoid()(mask_input)

        feature_long=self.transformer(out_noise2)
        feature_short=self.resnet(out_noise2)
        out_noise3 = feature_long * mask_input + feature_short * (1-mask_input)

        height=out_noise3.shape[2]
        width=out_noise3.shape[3]
        batch_size=out_noise3.shape[0]
        feap=self.pool(out_noise3).view(batch_size,-1)

        ##### position embedding
        position_embedding2=self.lrelu(self.mapp_position11(grid))
        position_embedding2=self.mapp_position22(position_embedding2)

        ### fusion2
        feat_clip_normal2=self.lrelu(self.mapp_space11(feat_clip_normal))
        feat_clip_normal2=self.mapp_space22(feat_clip_normal2)
        
        f1=self.mapping11(torch.cat([feap, feat_clip_normal2],dim=1))
        f1=self.mapping12(self.lrelu(f1)).view(batch_size,-1,1,1)
        feature1=nn.Sigmoid()(f1)
        feature1=feature1*out_noise3
        
        input_condition2=feat_clip_normal2.view(batch_size,-1,1,1).repeat(1,1,height,width)
        input_condition=torch.cat([out_noise3, input_condition2, position_embedding2],dim=1)
        
        g1=self.mapp11(input_condition)
        g1=self.mapp12(self.lrelu(g1))
        fea1=kernel2d_conv(out_noise3, g1, self.kh)
        fea1=self.mapp1(self.lrelu(fea1))
        fea1=nn.Sigmoid()(fea1)*out_noise3
        feature1=self.refine11(torch.cat([feature1,fea1],dim=1))+out_noise3
        
        feature1=self.r1_decoder(feature1)
        feature1=self.lrelu(self.pixel_shuffle(feature1))
        feature1=self.r2_decoder(feature1)
        feature1=self.lrelu(self.pixel_shuffle(feature1))
        
        feature_noise=self.conv_last2(feature1)
        feature_noise=self.conv_last3(self.lrelu(feature_noise))
        out_noise4=torch.clamp(out_dec_level1,min=0.0,max=1.0)+feature_noise
        return out_noise4


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


class low_light_transformer2(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer2, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=4*4*nf, d_inner=4*4*nf*2, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        
        num_blocks = [4,6,6,8]
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        
        self.refinement11=refinement2(num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                                    heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias = bias, LayerNorm_type=LayerNorm_type)
        

    def forward(self, x, mask=None, clip_condition_list=None):
        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        
        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask
        
        out_noise = self.recon_trunk(fea)
        
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise2 = self.conv_last(out_noise)
        out_noise2 = out_noise2 + x_center
        
        out_noise4=self.refinement11(clip_condition_list, out_noise2, x_center)
        
        return out_noise4, out_noise2
        
