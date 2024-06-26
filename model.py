# 模型总体架构

from torchinfo import summary
import torch
from torch import nn

from utils_model import Bottleneck1D, Bottleneck1D_M, CMGxSensorEncoder

class DCF(nn.Module):
    def __init__(self, num_cmg: int=5):
        super().__init__()
        self.num_cmg = num_cmg

        self.EncoderI1 = Bottleneck1D(num_cmg, num_cmg)
        self.EncoderV1 = Bottleneck1D(num_cmg, num_cmg)

        self.EncoderI2 = Bottleneck1D(num_cmg, num_cmg)
        self.EncoderV2 = Bottleneck1D(num_cmg, num_cmg)

    def forward(self, current: torch.Tensor, voltage: torch.Tensor):
        current1 = self.EncoderI1(current)
        voltage1 = self.EncoderV1(voltage)

        current2 = self.EncoderI2(current1)
        voltage2 = self.EncoderV2(voltage1)

        return current1, voltage1, current2, voltage2
    
class DCF_M(nn.Module):
    def __init__(self, num_cmg: int=5):
        super().__init__()
        self.num_cmg = num_cmg

        self.EncoderI1 = Bottleneck1D_M(num_cmg, num_cmg)
        self.EncoderV1 = Bottleneck1D_M(num_cmg, num_cmg)

        self.EncoderI2 = Bottleneck1D_M(num_cmg, num_cmg)
        self.EncoderV2 = Bottleneck1D_M(num_cmg, num_cmg)

    def forward(self, current: torch.Tensor, voltage: torch.Tensor):
        current1 = self.EncoderI1(current)
        voltage1 = self.EncoderV1(voltage)

        current2 = self.EncoderI2(current1)
        voltage2 = self.EncoderV2(voltage1)

        return current1, voltage1, current2, voltage2

class DCF_T(nn.Module):
    def __init__(self, num_cmg: int=4):
        super().__init__()
        self.num_cmg = num_cmg

        self.EncoderI1 = CMGxSensorEncoder(patch_size=128, num_patch=int(1024/128))
        self.EncoderV1 = CMGxSensorEncoder(patch_size=128, num_patch=int(1024/128))

        self.EncoderI2 = CMGxSensorEncoder(patch_size=128, num_patch=int(1024/128))
        self.EncoderV2 = CMGxSensorEncoder(patch_size=128, num_patch=int(1024/128))

    def forward(self, current: torch.Tensor, voltage: torch.Tensor):
        current1 = self.EncoderI1(current)
        voltage1 = self.EncoderV1(voltage)

        current2 = self.EncoderI2(current1)
        voltage2 = self.EncoderV2(voltage1)

        return current1, voltage1, current2, voltage2

if __name__=="__main__":
    model = DCF_T()
    summary(model)
