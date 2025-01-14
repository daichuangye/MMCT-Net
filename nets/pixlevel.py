import torch
import torch.nn as nn
class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2 
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max+x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        y = x_output * x
        return y
class self_PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(self_PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_H = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn_H=nn.BatchNorm2d(in_channels)
        self.relu_H = nn.ReLU(inplace=True)
        self.conv_W = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn_W = nn.BatchNorm2d(in_channels)
        self.relu_W = nn.ReLU(inplace=True)
        self.mlp=nn.Linear(in_features=in_channels,out_features=2*in_channels,bias=False)
        self.middle_ac=nn.ReLU(inplace=True)
        self.self_con=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,bias=False)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        B,C,H,W=x.shape
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max + x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        x_H=self.conv_H(x)
        x_H=self.bn_H(x_H)
        x_H=self.relu_H(x_H)
        x_W=self.conv_W(x)
        x_W=self.bn_W(x_W)
        x_W=self.relu_W(x_W)
        avg_W = nn.AdaptiveAvgPool2d((H,1))(x_W)
        avg_H = nn.AdaptiveAvgPool2d((1, W))(x_H)
        weight_self=avg_W @ avg_H
        weight_self=weight_self.transpose(1,3)
        weight_self=self.mlp(weight_self)
        weight_self=self.middle_ac(weight_self)
        weight_self=weight_self.transpose(1,3)
        weight_self=self.self_con(weight_self)
        weight_self=self.sig(weight_self)
        out_self=x*weight_self
        out_yuan = x_output * x
        y=out_yuan+out_self
        return y