# -*- coding: utf-8 -*-
# @Time : 2023/2/22 15:07 
# @Author : Mingzheng 
# @File : draw.py
# @desc :
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils.dataloader import USRADataset,USRADataset_collate,USRADataset_CR,USRADataset_collate_CR
from torch.utils.data import DataLoader
from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from tqdm import tqdm
from nets.general_net import *
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
class result_show():
    def __init__(self,labels,outputs,R2,RMSE,MSE,MAE):
        self.labels = labels
        self.outputs = outputs
        self.R2 = R2
        self.RMSE = RMSE
        self.MSE = MSE
        self.MAE = MAE
    def draw(self):
        test_labels = self.labels
        predictions = self.outputs
        # 绘图(大论文)
        x2 = np.linspace(-16, 16)
        y2 = x2
        def f_1(x, A, B):
            return A * x + B
        A1, B1 = optimize.curve_fit(f_1, test_labels, predictions)[0]
        y3 = A1 * test_labels + B1
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        point = plt.scatter(test_labels, predictions, edgecolors=None, c='k', s=16, marker='s')
        ax.plot(x2, y2, color='k', linewidth=1.5, linestyle='--')
        ax.plot(test_labels, y3, color='r', linewidth=2, linestyle='-')
        fontdict1 = {"size": 15, "color": "k", "family": "SimSun"}
        ax.set_xlabel("真实降雨强度", fontdict=fontdict1)
        ax.set_ylabel("估计降雨强度", fontdict=fontdict1)
        ax.grid(False)
        ax.set_xlim((0, 16.0))
        ax.set_ylim((0, 16.0))
        ax.set_xticks(np.arange(0, 16, step=1))
        ax.set_yticks(np.arange(0, 16, step=1))
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('k')
        ax.tick_params(left=True, bottom=True, direction='in', labelsize=14)
        titlefontdict = {'size': 16, 'color': 'k', 'family': 'SimSun'}
        ax.set_title('降雨估计散点图', titlefontdict, pad=20)
        fontdict = {'size': 14, 'color': 'k', 'family': 'Times New Roman'}
        ax.text(0.5, 15, r'$R^2=$' + str(round(self.R2, 3)), fontdict=fontdict)
        ax.text(0.5, 14, r'RMSE=' + str(round(self.RMSE, 3)), fontdict=fontdict)
        ax.text(0.5, 13, r'MSE=' + str(round(self.MSE, 3)), fontdict=fontdict)
        ax.text(0.5, 12, r'MAE=' + str(round(self.MAE, 3)), fontdict=fontdict)
        ax.text(0.5, 11, r'$y=$' + str(round(A1, 3)) + '$x$' + " + " + str(round(B1, 3)), fontdict=fontdict)
        ax.text(0.5, 10, r'$N=$' + str(len(test_labels)), fontdict=fontdict)
        # plt.scatter(test_labels,predictions,s=2,c=predictions,cmap='coolwarm')
        plt.show()
        # 绘图(小论文)
        x2 = np.linspace(-16, 16)
        y2 = x2


        def f_1(x, A, B):
            return A * x + B


        A1, B1 = optimize.curve_fit(f_1, test_labels, predictions)[0]
        y3 = A1 * test_labels + B1
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        point = plt.scatter(test_labels, predictions, edgecolors=None, c='k', s=16, marker='s')
        ax.plot(x2, y2, color='k', linewidth=1.5, linestyle='--')
        ax.plot(test_labels, y3, color='r', linewidth=2, linestyle='-')
        fontdict1 = {"size": 17, "color": "k", "family": "Times New Roman"}
        ax.set_xlabel("True Values", fontdict=fontdict1)
        ax.set_ylabel("Estimated Values", fontdict=fontdict1)
        ax.grid(False)
        ax.set_xlim((0, 16.0))
        ax.set_ylim((0, 16.0))
        ax.set_xticks(np.arange(0, 16, step=1))
        ax.set_yticks(np.arange(0, 16, step=1))
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('k')
        ax.tick_params(left=True, bottom=True, direction='in', labelsize=14)
        titlefontdict = {'size': 20, 'color': 'k', 'family': 'Times New Roman'}
        ax.set_title('Scatter plot of True data and Model Estimated', titlefontdict, pad=20)
        fontdict = {'size': 16, 'color': 'k', 'family': 'Times New Roman'}
        ax.text(0.5, 15, r'$R^2=$' + str(round(self.R2, 3)), fontdict=fontdict)
        ax.text(0.5, 14, r'RMSE=' + str(round(self.RMSE, 3)), fontdict=fontdict)
        ax.text(0.5, 13, r'MSE=' + str(round(self.MSE, 3)), fontdict=fontdict)
        ax.text(0.5, 12, r'MAE=' + str(round(self.MAE, 3)), fontdict=fontdict)
        ax.text(0.5, 11, r'$y=$' + str(round(A1, 3)) + '$x$' + " + " + str(round(B1, 3)), fontdict=fontdict)
        ax.text(0.5, 10, r'$N=$' + str(len(test_labels)), fontdict=fontdict)
        # plt.scatter(test_labels,predictions,s=2,c=predictions,cmap='coolwarm')
        # Estimate the 2D histogram
        nbins = 70
        H, xedges, yedges = np.histogram2d(test_labels, predictions, bins=nbins)
        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)
        # Mask zeros
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero
        # 开始绘图
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=plt.cm.get_cmap('jet'), vmin=0, vmax=40)
        cbar = plt.colorbar(ax=ax, ticks=[0, 10, 20, 30, 40], drawedges=False)
        # cbar.ax.set_ylabel('Frequency',fontdict=colorbarfontdict)
        colorbarfontdict = {'size': 16, 'color': 'k', 'family': 'Times New Roman'}
        cbar.ax.set_title('Counts', fontdict=colorbarfontdict, pad=8)
        cbar.ax.tick_params(labelsize=12, direction='in')
        cbar.ax.set_yticklabels(['0', '10', '20', '30', '>40'], family='Times New Roman')
        plt.style.use('seaborn-darkgrid')
        plt.show()
