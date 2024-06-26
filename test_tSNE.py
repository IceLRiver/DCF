# 计算诊断准确率

import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from data_loader import LoadPretrainV2
from model import DCF_T
from utils_function import PredictLoss

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# 使用t-SNE进行降维
def apply_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)
    return tsne_features

# 可视化
def plot_tsne(tsne_features, labels, save_path):
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels, legend='full', palette=palette)
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

# 可视化 三维
def plot_tsne_3d(tsne_features, labels, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    colors = [palette[label] for label in labels]
    scatter = ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], c=colors, marker='o')
    # 添加图例
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in unique_labels]
    ax.legend(handles, unique_labels, title="Classes")
    ax.set_title('t-SNE Visualization of Features (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    data_folder = r' '
    exper_save_folder = r' '
    os.makedirs(os.path.join(exper_save_folder, 'visual_t-SNE'), exist_ok=True)

    # 创建模型实例
    model = DCF_T()
    model.load_state_dict(torch.load(os.path.join(exper_save_folder, 'checkpoint/model_best.pth')))
    model.to(device)

    train=False # 训练集
    implicit=True # 隐式故障与显式故障
    test_dataset = LoadPretrainV2(data_folder, train=train, implicit=implicit)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=8)

    criterion = PredictLoss()
    losses_i = []
    losses_v = []
    currents = []
    voltages = []
    feature_i1 = []
    feature_v1 = []
    feature_i2 = []
    feature_v2 = []
    i_batch = 0
    for current, voltage in test_loader:
        current, voltage = current.to(device), voltage.to(device)
        with torch.no_grad():
            current1, voltage1, current2, voltage2 = model(current, voltage)
        loss_i = criterion(current1)
        loss_v = criterion(voltage1)
        losses_i.append(loss_i)
        losses_v.append(loss_v)

        currents.append(current)
        voltages.append(voltage)
        feature_i1.append(current1)
        feature_v1.append(voltage1)
        feature_i2.append(current2)
        feature_v2.append(voltage2)

        i_batch+=1
        print(f"{i_batch} / {len(test_loader)}", end='\r')

    data_name = 'implicit'

    # # 原始输入
    # currents = torch.cat(currents).cpu().numpy()
    # l, num_cmg, c = currents.shape
    # currents = currents.transpose(1, 0, 2).reshape(-1, c)
    # labels = np.repeat(np.arange(num_cmg), l)
    # tsne_features = apply_tsne(currents)
    # save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'I_'+data_name+'.png')
    # plot_tsne(tsne_features, labels, save_path)

    # voltages = torch.cat(voltages).cpu().numpy()
    # l, num_cmg, c = voltages.shape
    # voltages = voltages.transpose(1, 0, 2).reshape(-1, c)
    # labels = np.repeat(np.arange(num_cmg), l)
    # tsne_features = apply_tsne(voltages)
    # save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'V_'+data_name+'.png')
    # plot_tsne(tsne_features, labels, save_path)

    ## 特征
    feature_i1 = torch.cat(feature_i1).cpu().numpy()
    l, num_cmg, c = feature_i1.shape
    feature_i1 = feature_i1.transpose(1, 0, 2).reshape(-1, c)
    labels = np.repeat(np.arange(num_cmg), l)
    tsne_features = apply_tsne(feature_i1)
    save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'i1_'+data_name+'.png')
    plot_tsne(tsne_features, labels, save_path)

    feature_v1 = torch.cat(feature_v1).cpu().numpy()
    l, num_cmg, c = feature_v1.shape
    feature_v1 = feature_v1.transpose(1, 0, 2).reshape(-1, c)
    labels = np.repeat(np.arange(num_cmg), l)
    tsne_features = apply_tsne(feature_v1)
    save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'v1_'+data_name+'.png')
    plot_tsne(tsne_features, labels, save_path)

    feature_i2 = torch.cat(feature_i2).cpu().numpy()
    l, num_cmg, c = feature_i2.shape
    feature_i2 = feature_i2.transpose(1, 0, 2).reshape(-1, c)
    labels = np.repeat(np.arange(num_cmg), l)
    tsne_features = apply_tsne(feature_i2)
    save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'i2_'+data_name+'.png')
    plot_tsne(tsne_features, labels, save_path)

    feature_v2 = torch.cat(feature_v2).cpu().numpy()
    l, num_cmg, c = feature_v2.shape
    feature_v2 = feature_v2.transpose(1, 0, 2).reshape(-1, c)
    labels = np.repeat(np.arange(num_cmg), l)
    tsne_features = apply_tsne(feature_v2)
    save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'v2_'+data_name+'.png')
    plot_tsne(tsne_features, labels, save_path)


    # loss
    losses_i = torch.cat(losses_i).cpu().numpy()
    losses_v = torch.cat(losses_v).cpu().numpy()
    losses_i = np.expand_dims(losses_i, axis=2)
    losses_v = np.expand_dims(losses_v, axis=2)
    losses_iv = np.concatenate((losses_i, losses_v), axis=2)
    l, num_cmg, c = losses_iv.shape
    losses_iv = losses_iv.transpose(1, 0, 2).reshape(-1, c)
    labels = np.repeat(np.arange(num_cmg), l)
    tsne_features = losses_iv
    save_path = os.path.join(exper_save_folder, 'visual_t-SNE', 'ivLoss_'+data_name+'.png')
    plot_tsne(tsne_features, labels, save_path)  

    a=1

if __name__ == "__main__":
    main()

