# 计算融合诊断准确率

import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from data_loader import LoadPretrainV2
from model import DCF_T
from utils_function import PredictLoss, smooth_columns, window_back_overlap, calculate_metrics, plot_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    data_folder = r' '
    exper_save_folder = r' '
    os.makedirs(os.path.join(exper_save_folder, 'predict'), exist_ok=True)
    os.makedirs(os.path.join(exper_save_folder, 'predict', 'implicit_data'), exist_ok=True)

    criterion = PredictLoss()
    # 创建模型实例
    model = DCF_T()
    model.load_state_dict(torch.load(os.path.join(exper_save_folder, 'checkpoint/model_best.pth')))
    model.to(device)

    train= False# 训练集
    implicit= True # 隐式故障与显式故障
    test_dataset = LoadPretrainV2(data_folder, train=train, implicit=implicit)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)

    losses_i = []
    losses_v = []
    i_batch = 0
    for current, voltage in test_loader:
        current, voltage = current.to(device), voltage.to(device)
        with torch.no_grad():
            current1, voltage1, current2, voltage2 = model(current, voltage)
        loss_i = criterion(current1)
        loss_v = criterion(voltage1)
        losses_i.append(loss_i)
        losses_v.append(loss_v)
        i_batch+=1
        print(f"{i_batch} / {len(test_loader)}", end='\r')

    losses_i = smooth_columns(torch.cat(losses_i, dim=0), 20).cpu().numpy()
    losses_v = smooth_columns(torch.cat(losses_v, dim=0), 20).cpu().numpy()
    losses_iv = 0.5*losses_i + 0.5*losses_v

    confusion_matrix_iv=np.zeros((2,2))
    num_cmg = losses_iv.shape[1]
    for i_cmg in range(num_cmg):
        mean = np.mean(np.delete(losses_iv, i_cmg, axis=1))
        var = np.std(np.delete(losses_iv, i_cmg, axis=1))
        min = np.min(np.delete(losses_iv, i_cmg, axis=1), axis=1)
        # print('min: ', np.min(min))
        faults_iv = (losses_iv[:,i_cmg]<(mean-3*var)) & (losses_iv[:,i_cmg]<min)
        if i_cmg==2:
            confusion_matrix_iv[0, 0] = confusion_matrix_iv[0, 0]+np.sum(faults_iv)
            confusion_matrix_iv[1, 0] = confusion_matrix_iv[1, 0]+(losses_iv.shape[0]-np.sum(faults_iv))
        else:
            confusion_matrix_iv[0, 1] = confusion_matrix_iv[0, 1]+np.sum(faults_iv)
            confusion_matrix_iv[1, 1] = confusion_matrix_iv[1, 1]+(losses_iv.shape[0]-np.sum(faults_iv))

        print('cmg ', i_cmg, ' fusion diagnosis accuracy: ', np.sum(faults_iv)/losses_iv.shape[0])
        faults_iv0 = window_back_overlap(faults_iv, data_len=1024, gap=1/3)
        np.save(os.path.join(exper_save_folder, 'predict', 'implicit_data', 'cmg'+str(i_cmg)+'_fusion_faults.npy'), faults_iv0)

        a=1

    print(confusion_matrix_iv)
    accuracy, f1_score, MCC = calculate_metrics(confusion_matrix_iv)
    os.makedirs(os.path.join(exper_save_folder, 'predict', 'confusion_matrix', ''), exist_ok=True)
    plot_confusion_matrix(confusion_matrix_iv.astype(int), os.path.join(exper_save_folder, 'predict', 'confusion_matrix', 'implicit_cmg'+str(i_cmg)+'_fusion.png'))
    print("Accuracy: ", accuracy, " F1_score: ", f1_score, " MCC: ", MCC)
    a=1

if __name__ == "__main__":
    main()

