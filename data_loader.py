# 所有用于训练的数据加载与预处理类/函数

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time, os

    
# 用于训练/测试的数据生成器---------------------------------
# 取电流与电压通道，并分段、批量化
# data_folder: 数据存储文件夹
# train: 表示训练/测试状态，True/False
# implicit: 表示故障隐式/显示，True/False
# 使用cmg0123
class LoadPretrainV2(Dataset):
    def __init__(self, data_folder, train=True, implicit=True, data_len=1024, gap=1/3):
        self.len = data_len
        self.gap = gap

        self.num_cmg = 4
        self.train_len = 839310
        self.imfault_len = int(839310*0.33)
        self.exfault_len = self.train_len - self.imfault_len
        if train:
            self.data = np.zeros((self.num_cmg, 2, self.train_len))
            for num in range(self.num_cmg):
                data_path = os.path.join(data_folder, 'cmg'+str(num)+'.csv')
                cmgx = pd.read_csv(data_path)
                self.data[num, :, :] = cmgx[['rotor_current', 'rotor_voltage']][0:self.train_len].values.T
                self.items =list(range(0, int(self.train_len-self.len), int(self.len*(1-self.gap))))
        elif implicit:
            self.data = np.zeros((self.num_cmg, 2, self.imfault_len))
            for num in range(self.num_cmg):
                data_path = os.path.join(data_folder, 'cmg'+str(num)+'.csv')
                cmgx = pd.read_csv(data_path)
                self.data[num, :, :] = cmgx[['rotor_current', 'rotor_voltage']][self.train_len:self.train_len+self.imfault_len].values.T
                self.items =list(range(0, int(self.imfault_len-self.len), int(self.len*(1-self.gap))))
        else:
            self.data = np.zeros((self.num_cmg, 2, self.exfault_len))
            for num in range(self.num_cmg):
                data_path = os.path.join(data_folder, 'cmg'+str(num)+'.csv')
                cmgx = pd.read_csv(data_path)
                self.data[num, :, :] = cmgx[['rotor_current', 'rotor_voltage']][self.train_len+self.imfault_len:].values.T
                self.items =list(range(0, int(self.exfault_len-self.len), int(self.len*(1-self.gap))))
        self.len = data_len
        self.gap = gap
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, item):
        num = self.items[item]
        i = np.array(self.data[:, 0, num:num+self.len])
        v = np.array(self.data[:, 1, num:num+self.len])
        return torch.tensor(i, dtype=torch.float32), torch.tensor(v, dtype=torch.float32)


if __name__=="__main__":
    t0 = time.time()
    data_folder = r''
    train_dataset = LoadPretrainV2(data_folder, train=False, implicit=False)
    print('Load time: ', time.time()-t0, ' s')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)
    
    i = 0
    
    for x, y in train_loader:
        print(i, '/', len(train_loader))
        i+=1

    print('Total time: ', time.time()-t0, ' s')