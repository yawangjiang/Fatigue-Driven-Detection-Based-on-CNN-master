import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init


# import Config  <-- 如果这行报错可以注释掉，因为我们下面不再依赖它判断设备

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10

        # 【修改重点】：自动检测设备，不再硬性调用 .cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化权重并自动移动到可用设备（CPU或GPU）
        self.weight = nn.Parameter(torch.Tensor(self.n_channels).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # 确保计算在同一设备上进行
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # 这里的 self.weight 已经通过 __init__ 移动到了正确的设备
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out