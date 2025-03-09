from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import random
import os
import json
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False,
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=4, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=100, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=2, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.2, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float,
                    help='step size for scribble loss')
parser.add_argument('--mode', metavar='MODE', default='train', type=str,
                    help='Choose mode: train or evaluate')
args = parser.parse_args()


# 自注意力模块 (SEBlock)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        if channel // reduction == 0:
            reduction = max(1, channel // 2)  # 动态调整 reduction
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)  # 挤压
        excitation = self.fc2(F.relu(self.fc1(avg_out)))  # 激励
        excitation = self.sigmoid(excitation)  # 激励
        return x * excitation


# 多尺度特征融合模块
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(3 * out_channels)
        self.compress = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn(out)
        out = self.compress(out)
        return out


# 注意力引导的边缘感知模块 (Edge-Aware Attention Block)
class EdgeAwareAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(EdgeAwareAttentionBlock, self).__init__()
        self.sobel_x = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.attention_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始化 Sobel 滤波器核
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(channels, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        # 对输入特征的边缘进行计算
        edge_x = self.sobel_x(x.detach())
        edge_y = self.sobel_y(x.detach())
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_attention = self.sigmoid(edge)

        # 全局注意力
        global_attention = self.sigmoid(self.attention_conv(x))

        # 加权融合边缘和全局注意力
        alpha = 0.3
        out = x * (alpha * edge_attention + (1 - alpha) * global_attention)
        return out


# CNN模型
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.se1 = SEBlock(args.nChannel)  # 添加SEBlock
        self.multi_scale = MultiScaleBlock(args.nChannel, args.nChannel)  # 多尺度模块
        self.edge_attention = EdgeAwareAttentionBlock(args.nChannel)  # 添加Edge-Aware模块
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.multi_scale(x)
        x = self.edge_attention(x)  # 使用边缘感知模块
        x = self.bn3(self.conv3(x))
        return x


# 添加一个函数来计算评价指标
def compute_evaluation_metrics(features, labels, sample_size=10000):
    N = features.shape[0]
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        sampled_features = features[indices]
        sampled_labels = labels[indices]
    else:
        sampled_features = features
        sampled_labels = labels

    if len(np.unique(sampled_labels)) > 1:
        silhouette_avg = silhouette_score(sampled_features, sampled_labels)
        dbi = davies_bouldin_score(sampled_features, sampled_labels)
        chi = calinski_harabasz_score(sampled_features, sampled_labels)
    else:
        silhouette_avg = -1
        dbi = -1
        chi = -1

    metrics = {
        'Silhouette Score': silhouette_avg,
        'Davies-Bouldin Index': dbi,
        'Calinski-Harabasz Index': chi
    }
    return metrics


# 加载图片
im_sourse = cv2.imread(args.input)

# 统一处理图像分辨率
target_resolution = (3500, 3500)  # 设定统一分辨率，或者动态读取训练图像的分辨率
sourse_resolution = (im_sourse.shape[1], im_sourse.shape[0])
im = cv2.resize(im_sourse, target_resolution, interpolation=cv2.INTER_LINEAR)

data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

# 加载scribble，默认false
if args.scribble:
    mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    inds_sim = torch.from_numpy(np.where(mask == 255)[0])
    inds_scr = torch.from_numpy(np.where(mask != 255)[0])
    target_scr = torch.from_numpy(mask.astype(np.int))
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable(target_scr)
    # set minLabels
    args.minLabels = len(mask_inds)

# 模型实例化
model = MyNet(data.size(1))

if use_cuda:
    model.cuda()

# 相似性损失定义
loss_fn = torch.nn.CrossEntropyLoss()

# scribble损失定义
loss_fn_scr = torch.nn.CrossEntropyLoss()

# 连续性损失定义
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

# 动态调整 HPy_target 和 HPz_target
im_h, im_w = im.shape[:2]
HPy_target = torch.zeros(im_h - 1, im_w, args.nChannel)
HPz_target = torch.zeros(im_h, im_w - 1, args.nChannel)

if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

if args.mode == 'train':
    print("start train")
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # 初始化存储评价指标的列表
    metrics_history = {
        "Silhouette Score": [],
        "Davies-Bouldin Index": [],
        "Calinski-Harabasz Index": []
    }

    for batch_idx in range(args.maxIter):
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        # outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
        outputHP = output.view(data.size(2), data.size(3), args.nChannel)
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]

        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)
        ignore, target = torch.max(output, 1)

        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # 计算评价指标
        features = output.data.cpu().numpy()
        labels = im_target
        metrics = compute_evaluation_metrics(features, labels)

        # 存储当前迭代的指标
        metrics_history["Silhouette Score"].append(metrics["Silhouette Score"])
        metrics_history["Davies-Bouldin Index"].append(metrics["Davies-Bouldin Index"])
        metrics_history["Calinski-Harabasz Index"].append(metrics["Calinski-Harabasz Index"])

        # 根据最多类别数值添加数字标签
        class_labels = {}
        for c in np.unique(im_target):
            for i in range(args.nChannel):
                class_labels[str(i)] = str(i)

        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
            cv2.imshow("output", cv2.resize(im_target_rgb, (512, 512)))
            cv2.waitKey(1)

        if args.scribble:
            loss = (args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) +
                    args.stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr]) +
                    args.stepsize_con * (lhpy + lhpz))
        else:
            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()

        # 打印损失和评价指标
        print(f"Iteration {batch_idx + 1}/{args.maxIter} | Labels: {nLabels} | Loss: {loss.item():.4f} | "
              f"Silhouette: {metrics['Silhouette Score']:.4f} | DBI: {metrics['Davies-Bouldin Index']:.4f} | "
              f"CHI: {metrics['Calinski-Harabasz Index']:.4f}")

        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break
    # torch.save(model.state_dict(), 'trained_model_Mos2_test.pth')

    # 将 metrics_history 的所有值转换为 Python 的 float 类型
    metrics_history = {
        metric: [float(value) for value in values]
        for metric, values in metrics_history.items()
    }

    with open('metrics_without_modules.json', 'w') as f:
        json.dump(metrics_history, f)

    # 绘制评价指标折线图（折断的 y 轴）
    iterations = range(1, len(metrics_history["Silhouette Score"]) + 1)

    # 创建主图框架
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})

    # 绘制较小范围的指标（Silhouette 和 DBI）在上半部分
    ax1.plot(iterations, metrics_history["Silhouette Score"], label="Silhouette Score", marker="o", color="blue")
    ax1.plot(iterations, metrics_history["Davies-Bouldin Index"], label="Davies-Bouldin Index", marker="s",
             color="orange")
    ax1.set_ylabel("Small Range Metrics")
    ax1.set_ylim(-0.1, 1)  # 设置上半部分 y 轴范围（适合较小指标值）
    ax1.legend(loc="upper right")
    ax1.grid()

    # 绘制较大范围的指标（CHI）在下半部分
    ax2.plot(iterations, metrics_history["Calinski-Harabasz Index"], label="Calinski-Harabasz Index", marker="^",
             color="green")
    ax2.set_ylabel("CHI Index")
    ax2.set_ylim(0, max(metrics_history["Calinski-Harabasz Index"]) * 1.1)  # 设置下半部分 y 轴范围
    ax2.set_xlabel("Iteration")
    ax2.grid()
    ax2.legend(loc="upper right")

    # 在图的上下部分间添加断裂符号
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(labeltop=False)  # 关闭上半部分的 x 轴标签
    ax2.tick_params(labeltop=False)  # 关闭下半部分的 x 轴标签

    # 添加折断线
    d = 0.015  # 折断线的尺寸
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # 左侧折断线
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右侧折断线

    kwargs.update(transform=ax2.transAxes)  # 复制下半部分的折断线
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左侧折断线
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右侧折断线

    # 显示图像
    plt.suptitle("Evaluation Metrics Over Training Iterations (Broken Y-Axis)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('不加入三模块指标.png')
    plt.show()

    # 保存类别标签映射字典
    with open('class_labels.json', 'w', encoding='UTF-8') as f:
        json.dump(class_labels, f)

    # 保存颜色映射表
    # np.save('label_colours_test.npy', label_colours)

    # 对分类进行颜色记忆
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

    im_target_rgb = cv2.resize(im_target_rgb, sourse_resolution, interpolation=cv2.INTER_LINEAR)

    # cv2.imwrite("train_output_image_test.jpeg", im_target_rgb)


elif args.mode == 'evaluate':
    print("start evaluate")
    model.load_state_dict(torch.load('trained_model_Mos2_test.pth'))
    # model.eval()

    # 加载颜色映射表
    label_colours = np.load('label_colours_test.npy')  # 使用训练时保存的颜色映射表

    with torch.no_grad():
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        # im_target = np.load('train_output_classes.npy')

        # 加载类别标签映射字典
        with open('class_labels.json', 'r', encoding='UTF-8') as f:
            class_labels = json.load(f)

        # 计算评价指标
        features = output.data.cpu().numpy()
        labels = im_target
        metrics = compute_evaluation_metrics(features, labels)

        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

        # 输出类别统计信息
        unique_classes, counts = np.unique(im_target, return_counts=True)
        for c, count in zip(unique_classes, counts):
            print(f"类别 {c}（{class_labels[str(c)]}）的像素数量: {count}")

        print("Unique classes and counts:", dict(zip(unique_classes, counts)))

        # 打印评价指标
        print(f"Silhouette Score: {metrics['Silhouette Score']:.4f}")
        print(f"Davies-Bouldin Index: {metrics['Davies-Bouldin Index']:.4f}")
        print(f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Index']:.4f}")

        im_target_rgb = cv2.resize(im_target_rgb, sourse_resolution, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite("evaluate_output_image_20_4_test.jpeg", im_target_rgb)

# 保存输出图片
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

# output_file = os.path.splitext(args.input)[0] + '_output' + '.png'
# cv2.imwrite(output_file, im_target_rgb)
