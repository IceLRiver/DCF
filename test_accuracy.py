# 计算诊断准确率

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

    criterion = PredictLoss()
    # 创建模型实例
    model = DCF_T()
    model.load_state_dict(torch.load(os.path.join(exper_save_folder, 'checkpoint/model_best.pth')))
    model.to(device)

    train=False # 训练集
    implicit=True # 隐式故障与显式故障
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

    confusion_matrix_i=np.zeros((2,2))
    confusion_matrix_v=np.zeros((2,2))
    num_cmg = losses_i.shape[1]
    for i_cmg in range(num_cmg):
        mean = np.mean(np.delete(losses_i, i_cmg, axis=1))
        var = np.std(np.delete(losses_i, i_cmg, axis=1))
        min = np.min(np.delete(losses_i, i_cmg, axis=1), axis=1)
        faults_i = (losses_i[:,i_cmg]<(mean-3*var)) & (losses_i[:,i_cmg]<min)

        if i_cmg==2:
            confusion_matrix_i[0, 0] = confusion_matrix_i[0, 0]+np.sum(faults_i)
            confusion_matrix_i[1, 0] = confusion_matrix_i[1, 0]+(losses_i.shape[0]-np.sum(faults_i))
        else:
            confusion_matrix_i[0, 1] = confusion_matrix_i[0, 1]+np.sum(faults_i)
            confusion_matrix_i[1, 1] = confusion_matrix_i[1, 1]+(losses_i.shape[0]-np.sum(faults_i))

        print('cmg ', i_cmg, ' current diagnosis accuracy: ', np.sum(faults_i)/losses_i.shape[0])
        faults_i0 = window_back_overlap(faults_i, data_len=1024, gap=1/3)
        os.makedirs(os.path.join(exper_save_folder, 'predict', 'implicit_data'), exist_ok=True)
        np.save(os.path.join(exper_save_folder, 'predict', 'implicit_data', 'cmg'+str(i_cmg)+'_current_faults.npy'), faults_i0)

        mean = np.mean(np.delete(losses_v, i_cmg, axis=1))
        var = np.std(np.delete(losses_v, i_cmg, axis=1))
        min = np.min(np.delete(losses_v, i_cmg, axis=1), axis=1)
        faults_v = (losses_v[:,i_cmg]<(mean-3*var)) & (losses_v[:,i_cmg]<min)
        if i_cmg==2:
            confusion_matrix_v[0, 0] = confusion_matrix_v[0, 0]+np.sum(faults_v)
            confusion_matrix_v[1, 0] = confusion_matrix_v[1, 0]+(losses_v.shape[0]-np.sum(faults_v))
        else:
            confusion_matrix_v[0, 1] = confusion_matrix_v[0, 1]+np.sum(faults_v)
            confusion_matrix_v[1, 1] = confusion_matrix_v[1, 1]+(losses_v.shape[0]-np.sum(faults_v))
        print('cmg ', i_cmg, ' voltage diagnosis accuracy: ', np.sum(faults_v)/losses_v.shape[0])
        faults_v0 = window_back_overlap(faults_v, data_len=1024, gap=1/3)
        np.save(os.path.join(exper_save_folder, 'predict', 'implicit_data', 'cmg'+str(i_cmg)+'_voltage_faults.npy'), faults_v0)
        a=1

    print(confusion_matrix_i)
    accuracy, f1_score, MCC = calculate_metrics(confusion_matrix_i)
    os.makedirs(os.path.join(exper_save_folder, 'predict', 'confusion_matrix'), exist_ok=True)
    plot_confusion_matrix(confusion_matrix_i.astype(int), os.path.join(exper_save_folder, 'predict', 'confusion_matrix', 'implicit_cmg'+str(i_cmg)+'_current.png'))
    print("Current Accuracy: ", accuracy, " F1_score: ", f1_score, " MCC: ", MCC)
    print(confusion_matrix_v)
    accuracy, f1_score, MCC = calculate_metrics(confusion_matrix_v.astype(int))
    plot_confusion_matrix(confusion_matrix_v.astype(int), os.path.join(exper_save_folder, 'predict', 'confusion_matrix', 'implicit_cmg'+str(i_cmg)+'_voltage.png'))
    print("Voltage Accuracy: ", accuracy, " F1_score: ", f1_score, " MCC: ", MCC)
    a=1

if __name__ == "__main__":
    main()

