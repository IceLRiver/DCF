# 模型的基础模块/类

from torch.utils.data import DataLoader
import time
from data_loader import LoadPretrain
from torchinfo import summary

import torch
from torch import nn


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv1d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.Tanh()

        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.Tanh()

        self.avgpool = nn.AvgPool1d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv1d(planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu3 = nn.Tanh()

        self.stride = stride

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu3(out)
        return out
    
class Bottleneck1D_M(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv1d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.tanh1 = nn.Tanh()

        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.tanh2 = nn.Tanh()

        self.conv3 = nn.Conv1d(planes, planes, 7, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.tanh3 = nn.Tanh()

        self.conv4 = nn.Conv1d(planes, planes, 15, padding=7, bias=False)
        self.bn4 = nn.BatchNorm1d(planes)
        self.tanh4 = nn.Tanh()

        self.conv5 = nn.Conv1d(planes, planes, 7, padding=3, bias=False)
        self.bn5 = nn.BatchNorm1d(planes)
        self.tanh5 = nn.Tanh()

        self.conv6 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm1d(planes)
        self.tanh6 = nn.Tanh()

        self.conv7 = nn.Conv1d(planes, planes, 1, bias=False)
        self.bn7 = nn.BatchNorm1d(planes)
        self.tanh7 = nn.Tanh()

        self.stride = stride

    def forward(self, x: torch.Tensor):
        identity1 = x

        out = self.tanh1(self.bn1(self.conv1(x)))
        out = self.tanh2(self.bn2(self.conv2(out)))
        out = self.tanh3(self.bn3(self.conv3(out+identity1)))
        identity2 = out

        out = self.tanh4(self.bn4(self.conv4(out)))
        out = self.tanh5(self.bn5(self.conv5(out+identity2)))
        identity3 = out

        out = self.tanh6(self.bn6(self.conv6(out)))
        out = self.tanh7(self.bn7(self.conv7(out+identity3)))

        return out

# 将输入的四通道信号（batch*num_cmg*sensor*len）取窗切分，得到（batch*num_cmg*num_window*len_window
class Patches(nn.Module):
    def __init__(self, patch_sizeT=100):
        super(Patches, self).__init__()
        self.patch_size = patch_sizeT

    def forward(self, cmgx):
        batch_size, channels, height, width = cmgx.size()  # batch*num_cmg*sensor*len
        # 将图像按照 patch_size 进行切分
        patches = cmgx.view(batch_size, channels, self.patch_size, height*width // self.patch_size)
        return patches.transpose(2, 3)

# patch encoder,得到Transformer encoder的输入  
class PatchEncoder(nn.Module):
    def __init__(self, num_patchesT, patch_sizeT, projection_dimT=64):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patchesT
        self.projection_dimT = projection_dimT
        self.projection = nn.Linear(patch_sizeT, self.projection_dimT)
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_patches, embedding_dim=self.projection_dimT
        )

    def forward(self, patch: torch.Tensor):
        positions = torch.arange(0, self.num_patches, device='cuda')
        positions = positions.unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(patch.size(0), patch.size(1), 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
       
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        return x
    
class ModeEncoder(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2):
        super(ModeEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.size()
        x = self.transformer(x.view(batch_size * channels, height, width))
        x = x.view(batch_size, channels, height, width)
        return x.view(batch_size, channels, height*width)
    
class CMGxSensorEncoder(nn.Module):
    def __init__(self, patch_size, num_patch):
        super(CMGxSensorEncoder, self).__init__()
        self.Patches = Patches(patch_size)
        self.project_dim = patch_size # 暂且保持整个encoder输入输出大小相同
        self.PatchEncoder = PatchEncoder(num_patch, patch_size, self.project_dim)
        self.TransformerEncoder = ModeEncoder(self.project_dim)

    def forward(self, x:torch.Tensor):
        x = self.Patches(x.unsqueeze(3))
        x = self.PatchEncoder(x)
        x = self.TransformerEncoder(x)
        return x

    
if __name__=="__main__":
    t0 = time.time()
    data_folder = r' '
    train_dataset = LoadPretrain(data_folder)
    print('Load time: ', time.time()-t0, ' s')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)

    # model = Bottleneck1D(inplanes=5, planes=5)
    model = CMGxSensorEncoder(patch_size=128, num_patch=int(1024/128), project_dim=64)
    summary(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    i = 0
    for current, voltage in train_loader:
        current, voltage = current.to(device), voltage.to(device)
        output1 = model(current)
        output2 = model(voltage)

        print(i, '/', len(train_loader))
        i+=1

    print('Total time: ', time.time()-t0, ' s')

