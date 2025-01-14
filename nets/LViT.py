import torch
import torch.nn as nn
from .Vit import VisionTransformer, Reconstruct, Reconstruct_cat, VisionTransformer_y1_huan, VisionTransformer_yNumber_huan
from .Vit import VisionTransformer_xNumber,x_Number_vit_Reconstruct_add
from .pixlevel import PixLevelModule,self_PixLevelModule
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)
class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1,padding_mode='reflect')
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.downSample=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=2,padding=1,
                                        padding_mode="reflect",bias=False)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.act1=nn.LeakyReLU()
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    def forward(self, x):
        out=self.downSample(x)
        out= self.bn1(out)
        out=self.act1(out)
        return self.nConvs(out)
class DownBlockUseMax(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlockUseMax, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Upblock_cnn_cat(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    def forward(self, x, skip_x):
        up = self.up(x)
        up=self.nConvs(up)
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension
        return x
class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)
class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit_y1_huan = VisionTransformer_y1_huan(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit_y2_huan = VisionTransformer_yNumber_huan(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit_y3_huan = VisionTransformer_yNumber_huan(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.x3_vit = VisionTransformer_xNumber(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.x4_vit = VisionTransformer_xNumber(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.x3_vit_add = x_Number_vit_Reconstruct_add(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.x4_vit_add = x_Number_vit_Reconstruct_add(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlockUseMax(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlockUseMax(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.Up_cat1=Upblock_cnn_cat(in_channels=512,out_channels=256,nb_Conv=2)
        self.Up_cat2 = Upblock_cnn_cat(in_channels=512, out_channels=128, nb_Conv=2)
        self.Up_cat3 = Upblock_cnn_cat(in_channels=256, out_channels=64, nb_Conv=2)
        self.final_con=ConvBatchNorm(in_channels=128,out_channels=in_channels)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()
        self.multi_activation = nn.Softmax()
        self.attention_x1_vit_add = self_PixLevelModule(64)
        self.attention_x2_vit_add = self_PixLevelModule(128)
        self.attention_x3_vit_add = self_PixLevelModule(256)
        self.attention_x4_vit_add = self_PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.y1_reconstruct_cat=Reconstruct_cat(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.y2_reconstruct_cat = Reconstruct_cat(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.y3_reconstruct_cat = Reconstruct_cat(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.y4_reconstruct_cat =Reconstruct_cat(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
    def forward(self, x, text):
        x = x.float()
        x1 = self.inc(x)
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        y1 = self.downVit(x1, x1, text1)
        y1_huan = self.y1_reconstruct_cat(y1, x1)
        x2=self.down1(y1_huan)
        y2 = self.downVit_y1_huan(y1_huan, y1, text1)
        y2_huan=self.y2_reconstruct_cat(y2,x2)
        x3=self.down2(y2_huan)
        y3=self.downVit_y2_huan(y2_huan,y2,text1)
        y3_huan = self.y3_reconstruct_cat(y3, x3)
        x4=self.down3(y3_huan)
        y4=self.downVit_y3_huan(y3_huan,y3,text1)
        y4_huan = self.y4_reconstruct_cat(y4, x4)
        x1_vit= x1
        x2_vit = x2
        x3_vit = self.x3_vit(x3)
        x4_vit = self.x4_vit(x4)
        x1_vit_add= x1_vit
        x2_vit_add = x2_vit
        x3_vit_add = self.x3_vit_add(x3_vit, x3)
        x4_vit_add = self.x4_vit_add(x4_vit, x4)
        attention_x1_vit_add=self.attention_x1_vit_add(x1_vit_add)
        attention_x2_vit_add = self.attention_x2_vit_add(x2_vit_add)
        attention_x3_vit_add = self.attention_x3_vit_add(x3_vit_add)
        attention_x4_vit_add = self.attention_x4_vit_add(x4_vit_add)
        x4=y4_huan+attention_x4_vit_add
        x3 = y3_huan + attention_x3_vit_add
        x2 = y2_huan + attention_x2_vit_add
        x1 = y1_huan + attention_x1_vit_add
        x4_cat_x3=self.Up_cat1(x4,x3)
        x3_cat_x2=self.Up_cat2(x4_cat_x3,x2)
        x2_cat_x1 = self.Up_cat3(x3_cat_x2, x1)
        x = self.final_con(x2_cat_x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        return logits