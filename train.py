import numpy as np
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import LoadPretrainV2
from model import DCF_T
from utils_function import MyLoss, plot_training_loss, smooth_columns, calculate_metrics, PredictLoss

def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses_batch = []
    losses_i = []
    losses_v = []
    criterion_test = PredictLoss()
    i_batch = 0
    for current, voltage in test_loader:
        current, voltage = current.to(device), voltage.to(device)
        with torch.no_grad():
            current1, voltage1, current2, voltage2 = model(current, voltage)
            loss11, loss12, loss2 = criterion(current1, voltage1, current2, voltage2)
            loss = loss11*0.25+loss12*0.25+loss2*0.5
        losses_batch.append([loss11.item(), loss12.item(), loss2.item(), loss.item()])
        loss_i = criterion_test(current1)
        loss_v = criterion_test(voltage1)
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
        faults_iv = (losses_iv[:,i_cmg]<(mean-3*var)) & (losses_iv[:,i_cmg]<min)
        if i_cmg==2:
            confusion_matrix_iv[0, 0] = confusion_matrix_iv[0, 0]+np.sum(faults_iv)
            confusion_matrix_iv[1, 0] = confusion_matrix_iv[1, 0]+(losses_iv.shape[0]-np.sum(faults_iv))
        else:
            confusion_matrix_iv[0, 1] = confusion_matrix_iv[0, 1]+np.sum(faults_iv)
            confusion_matrix_iv[1, 1] = confusion_matrix_iv[1, 1]+(losses_iv.shape[0]-np.sum(faults_iv))
        print('cmg ', i_cmg, ' fusion diagnosis accuracy: ', np.sum(faults_iv)/losses_iv.shape[0])
        a=1
    print(confusion_matrix_iv)
    accuracy, f1_score, MCC = calculate_metrics(confusion_matrix_iv)
    print("Accuracy: ", accuracy, " F1_score: ", f1_score, " MCC: ", MCC)

    return losses_batch, [accuracy, f1_score, MCC]

def pretrain(model, train_loader, test_loader, save_path, learning_rate=0.001, weight_decay=0.001, num_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Device: ', device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MyLoss() # -----------------------------------------------------
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    checkpoint_filepath = os.path.join(save_path, 'checkpoint', 'model_best.pth')
    
    best_loss = 0.0
    losses = []
    losses_test = []
    metrics = []
    for epoch in range(num_epochs):
        model.train()
        losses_batch = []
        i_batch = 1
        for current, voltage in train_loader:
            current, voltage = current.to(device), voltage.to(device)
            optimizer.zero_grad()
            current1, voltage1, current2, voltage2 = model(current, voltage)
            loss11, loss12, loss2 = criterion(current1, voltage1, current2, voltage2)
            loss = loss11*0.25+loss12*0.25+loss2*0.5
            losses_batch.append([loss11.item(), loss12.item(), loss2.item(), loss.item()])
            loss.backward()
            optimizer.step()
            print(f"{i_batch} / {len(train_loader)}", end='\r')
            i_batch+=1

        scheduler.step()
        print('testing')
        loss_test, metric = test(model, test_loader, criterion)
        
        losses.append([sum(col)/len(train_loader.dataset) for col in zip(*losses_batch)])
        losses_test.append([sum(col)/len(test_loader.dataset) for col in zip(*loss_test)])
        metrics.append(metric)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses[epoch]}")

        if epoch==0:
            best_loss=losses[epoch][-1]
        if losses[epoch][-1] < best_loss:
            best_loss = losses[epoch][-1]
            torch.save(model.state_dict(), checkpoint_filepath)

    print(f"Best Train Loss: {best_loss:.4f}")
    
    # 保存模型结构和参数
    torch.save(model, os.path.join(save_path, 'model/model_last.pth'))
    losses_np = np.array(losses)
    np.save('{0}/loss/losses.npy'.format(save_path), losses_np)
    losses_test_np = np.array(losses_test)
    np.save('{0}/loss/losses_test.npy'.format(save_path), losses_test_np)

    np.save('{0}/loss/metrics_test.npy'.format(save_path), np.array(metrics))

    return losses_np

def main():
    print('Torch ', torch.__version__)
    data_folder = r' '

    exper_save_folder = r' '
    # 如果不存在，则创建文件夹
    os.makedirs(exper_save_folder, exist_ok=True)
    os.makedirs(os.path.join(exper_save_folder, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(exper_save_folder, 'model'), exist_ok=True)
    os.makedirs(os.path.join(exper_save_folder, 'loss'), exist_ok=True)

    train_dataset = LoadPretrainV2(data_folder)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=8)

    # 用于测试的配置
    train=False # 训练集
    implicit=True # 隐式故障与显式故障
    test_dataset = LoadPretrainV2(data_folder, train=train, implicit=implicit)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)

    # 创建模型实例
    dcf = DCF_T()

    # 训练
    loss = pretrain(model=dcf, train_loader=train_loader, test_loader=test_loader, save_path=exper_save_folder)
    plot_training_loss(loss, ['Loss11', 'Loss12', 'Loss2', 'Loss'],
                       os.path.join(exper_save_folder, 'loss'))

if __name__ == "__main__":
    main()