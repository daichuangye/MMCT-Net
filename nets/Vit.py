import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1,padding_mode="reflect")
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
class Reconstruct_cat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct_cat, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,padding_mode="reflect")
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.channel_change=ConvBatchNorm(in_channels=3*in_channels,out_channels=out_channels)
    def forward(self, x,cat_x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor,mode='bilinear')(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        result1=out+cat_x
        result2 = torch.cat([result1,out, cat_x], dim=1)
        result=self.channel_change(result2)
        return result
class x_Number_vit_Reconstruct_add(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(x_Number_vit_Reconstruct_add, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,padding_mode="reflect")
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
    def forward(self, x,add_x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        result1=out+add_x
        return result1
class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,padding_mode="reflect")
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
    def forward(self, x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
class Embeddings(nn.Module):
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size,padding_mode="reflect")
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)
    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class ConvTransBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
class VisionTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.vis = vis
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)
    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x+self.CTBN3(text)
            x = self.Encoder_blocks(x)
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)
            x = self.CTBN(x)
            x = x.transpose(1, 2)
            y = torch.cat([x, skip_x], dim=2)
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)
            y = x+skip_x
            return y
class VisionTransformer_y1_huan(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer_y1_huan, self).__init__()
        self.config = config
        self.vis = vis
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)
    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x+self.CTBN3(text)
            x = self.Encoder_blocks(x)
            x=torch.cat([x,skip_x],dim=2)
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)
            x = self.CTBN(x)
            x = x.transpose(1, 2)
            y = torch.cat([x, skip_x], dim=2)
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)
            y = x+skip_x
            return y
class VisionTransformer_yNumber_huan(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer_yNumber_huan, self).__init__()
        self.config = config
        self.vis = vis
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)
    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x+self.CTBN3(text)
            x = self.Encoder_blocks(x)
            x=torch.cat([x,skip_x],dim=2)
            return x
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)
            x = self.CTBN(x)
            x = x.transpose(1, 2)
            y = torch.cat([x, skip_x], dim=2)
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)
            y = x+skip_x
            return y
class VisionTransformer_xNumber(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer_xNumber, self).__init__()
        self.config = config
        self.vis = vis
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])
    def forward(self, x, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            x = self.Encoder_blocks(x)
            return x
        else:
            x = self.Encoder_blocks(x)