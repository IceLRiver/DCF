
import os
import torch
from torch.utils.data import DataLoader

from data_loader import LoadPretrainV2
from model import DCF_T
from utils_function import PredictLoss, smooth_columns, plot_predict_loss_compare


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    data_folder = r' '
    exper_save_folder = r' '
    os.makedirs(os.path.join(exper_save_folder, 'predict'), exist_ok=True)

    train=False # 训练集
    implicit=True # 隐式故障与显式故障

    test_dataset = LoadPretrainV2(data_folder, train=train, implicit=implicit)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)

    criterion = PredictLoss()
    # 创建模型实例
    model = DCF_T()
    model.load_state_dict(torch.load(os.path.join(exper_save_folder, 'checkpoint/model_best.pth')))
    model.to(device)

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

    plot_predict_loss_compare(losses_i, 'current', os.path.join(exper_save_folder, 'predict'), train, implicit)
    plot_predict_loss_compare(losses_v, 'voltage', os.path.join(exper_save_folder, 'predict'), train, implicit)

    a=1

if __name__ == "__main__":
    main()

