# 自用函数库

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

    
# 相似度损失
class SimSelfLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, sensor, ):
        sensor = sensor / sensor.norm(dim=2, keepdim=True)
        similarity = torch.bmm(sensor, sensor.transpose(1, 2))
        similarity[:, torch.arange(similarity.shape[2]), torch.arange(similarity.shape[2])] = 0
        loss = torch.mean(torch.std(similarity, dim=(0)))*25/16
        return loss*100


class ContraSelfLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.labels = {}

    def get_logits(self, sensor):
        sensor_list = [sensor[i, :, :] for i in range(sensor.size(0))]
        sensor = torch.cat(sensor_list)
        logits_sensor = sensor @ sensor.T
        return logits_sensor

    def get_ground_truth(self, device, num_cmg, batch_size) -> torch.Tensor:
        if device not in self.labels:
            labels = torch.arange(num_cmg, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]
        return labels.repeat(batch_size)

    def forward(self, sensor):
        device = sensor.device
        logits_sensor = self.get_logits(sensor)
        labels = self.get_ground_truth(device, sensor.size(1), sensor.size(0))
        loss = F.cross_entropy(logits_sensor, labels)
        return loss

    
class ContrastiveLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.labels = {}

    def get_logits(self, current, voltage):
        current_list = [current[i, :, :] for i in range(current.size(0))]
        voltage_list = [voltage[i, :, :] for i in range(voltage.size(0))]
        current = torch.cat(current_list)
        voltage = torch.cat(voltage_list)

        logits_current = current @ voltage.T
        logits_voltage = logits_current.transpose(0, 1)
        return logits_current, logits_voltage

    def get_ground_truth(self, device, num_cmg, batch_size) -> torch.Tensor:
        if device not in self.labels:
            labels = torch.arange(num_cmg, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]
        return labels.repeat(batch_size)

    def forward(self, current, voltage):
        device = current.device
        logits_current, logits_voltage = self.get_logits(current, voltage)
        labels = self.get_ground_truth(device, current.size(1), current.size(0))
        loss = (
            F.cross_entropy(logits_current, labels) +
            F.cross_entropy(logits_voltage, labels)
        ) / 2

        return loss


class MyLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.SimSelfLoss = SimSelfLoss()
        self.ContrastiveLoss = ContrastiveLoss()

    def forward(self, current1, voltage1, current2, voltage2):
        loss11 = self.SimSelfLoss(current1)
        loss12 = self.SimSelfLoss(voltage1)
        loss2 = self.ContrastiveLoss(current2, voltage2)
        return loss11, loss12, loss2
    
class MyLossAblation2c(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.SimSelfLoss = SimSelfLoss()
        self.ContrastiveLoss = ContrastiveLoss()

    def forward(self, current1, voltage1, current2, voltage2):
        loss11 = self.SimSelfLoss(current1)
        loss12 = self.SimSelfLoss(voltage1)
        loss21 = self.ContrastiveLoss(current1, voltage1)
        loss22 = self.ContrastiveLoss(current2, voltage2)
        return loss11, loss12, loss21, loss22
    
class MyLoss1level2(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.SimSelfLoss = SimSelfLoss()
        self.ContrastiveLoss = ContrastiveLoss()

    def forward(self, current1, voltage1, current2, voltage2):
        loss11 = self.SimSelfLoss(current2)
        loss12 = self.SimSelfLoss(voltage2)
        loss2 = self.ContrastiveLoss(current2, voltage2)
        return loss11, loss12, loss2
    
class MyLossPlus(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.SimSelfLoss = SimSelfLoss()
        self.ContrastiveLoss = ContrastiveLoss()

    def forward(self, current1, voltage1, current2, voltage2):
        loss11 = self.SimSelfLoss(current1)
        loss12 = self.SimSelfLoss(voltage1)
        loss20 = self.ContrastiveLoss(current2, voltage2)
        loss21 = self.ContrastiveLoss(current2, current2)
        loss22 = self.ContrastiveLoss(voltage2, voltage2)
        return loss11, loss12, loss20, loss21, loss22


def plot_training_loss(loss, title, save_folder):
    num_plots = loss.shape[1]
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(9, 6*num_plots))
    for i in range(num_plots):
        axes[i].plot(loss[:, i])
        axes[i].set_title(title[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid(True)
        axes[i].legend(['Loss'])
    plt.tight_layout()
    plt.savefig(save_folder+'/training_loss.png', bbox_inches='tight')
    plt.show()
    plt.close()


# 测试
class PredictLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, sensor, ):
        sensor = sensor / sensor.norm(dim=2, keepdim=True)
        similarity = torch.bmm(sensor, sensor.transpose(1, 2))
        similarity[:, torch.arange(similarity.shape[1]), torch.arange(similarity.shape[2])] = 0
        loss = torch.mean(similarity, dim=1) * similarity.shape[1] / (similarity.shape[1]-1)
        return loss


def smooth_columns(tensor, smooth_factor):
    tensor_t = tensor.t()
    kernel_size = smooth_factor * 2 + 1
    kernel = torch.ones(1, 1, kernel_size, device='cuda') / kernel_size
    padding = int((kernel.size(2) - 1)/2)
    # 在输入向量的两端进行边缘填充
    padded_tensor = F.pad(tensor_t.unsqueeze(1), (padding, padding), mode='replicate')
    # 对每一列进行卷积
    smoothed_tensor_t = F.conv1d(padded_tensor, kernel, stride=1)[:, 0, :]
    # 返回平滑后的张量（转置回原始形状）
    return smoothed_tensor_t.t()


def plot_predict_loss(loss, sensor, save_folder, train, implicit):
    if train:
        data_name = 'train'
    elif implicit:
        data_name = 'implicit'
    else:
        data_name = 'explicit'
    num_plots = loss.shape[1]
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(9, 6*num_plots))
    for i in range(num_plots):
        axes[i].plot(loss[:, i])
        axes[i].set_title('cmg'+str(i+1))
        axes[i].set_xlabel('time')
        axes[i].set_ylabel(sensor)
        axes[i].set_ylim(0.75, 0.95)
        axes[i].grid(True, )
        # axes[i].legend(['Loss'])
    plt.tight_layout()
    plt.savefig(save_folder+'/'+sensor+'_predict_loss_'+data_name+'.png', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_predict_loss_compare(loss, sensor, save_folder, train, implicit):
    if train:
        data_name = 'train'
    elif implicit:
        data_name = 'implicit'
    else:
        data_name = 'explicit'
    num_plots = loss.shape[1]
    plt.figure(figsize=(5, 4))
    for i in range(num_plots):
        if i==1:
            plt.plot(loss[:, i], color='#2ca02c')
        elif i==3:
            plt.plot(loss[:, i], color='#1f77b4')
        elif i==0:
            plt.plot(loss[:, i], color='#ff7f0e')
        elif i==2:
            plt.plot(loss[:, i], color='#d62728')

    plt.xlabel('time')
    plt.ylabel(sensor)
    plt.ylim(0.75, 0.95)
    plt.grid(True, )
    plt.legend(['cmg0 healthy', 'cmg1 healthy', 'cmg2 near failure', 'cmg3 healthy'])
    plt.tight_layout()
    plt.savefig(save_folder+'/'+sensor+'_predict_loss_'+data_name+'_compare.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def split_and_merge(input_tensor, overlap_ratio=1/3):
    batch_size, num, length = input_tensor.size()
    overlap_length = int(length * overlap_ratio)
    # 拆解张量为 batch_size*(num, len)
    split_tensors = torch.split(input_tensor, 1, dim=0)
    # 沿着维度1拼接，并重叠0.2*len
    merged_tensor = torch.cat([tensor.squeeze(0)[:, :-overlap_length] if i > 0 else tensor.squeeze(0) 
                               for i, tensor in enumerate(split_tensors)], dim=1)
    return merged_tensor


# 将诊断结果一维向量还原至输入尺寸
def window_back_overlap(vector, data_len, gap):
    len = int(data_len*(1-gap))
    matrix = np.tile(vector, (len, 1)).T # 将一维向量复制为二维矩阵
    overlapped_vector = matrix.flatten() # 展平为一维向量
    repeated_elements = np.tile(overlapped_vector[-1], data_len-len)  # 复制最后一个元素n次
    overlapped_vector = np.append(overlapped_vector, repeated_elements)  # 将复制的元素附加到向量末尾
    return overlapped_vector


# 根据混淆矩阵计算accuracy, F1-score, AUC
# 适用于二分类问题，即2*2混淆矩阵
def calculate_metrics(confusion_matrix):
    # Extract values from confusion matrix
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]
    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    # Calculate F1-score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    MCC = (TP*TN-FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return accuracy, f1_score, MCC


def plot_confusion_matrix(cm, save_path, class_names=['near failure', 'healthy']):
    """
    绘制混淆矩阵图。
    参数:
    cm (numpy.ndarray): 混淆矩阵。
    class_names (list): 类别名称列表。
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()
    