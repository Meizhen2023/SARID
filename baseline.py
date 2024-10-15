import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils.dataloader import USRADataset,USRADataset_collate,USRADataset_CR,USRADataset_collate_CR
from torch.utils.data import DataLoader
from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
from nets.general_net import BaseCNN_Conv
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from tqdm import tqdm
from nets.general_net import *
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import r2_score  # R square
from utils.draw import result_show
import torchinfo

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Hyper parameters
num_epochs = 100
batch_size = 64
learning_rate = 0.001

train_features_path = f'{data features train}.npy'
train_labels_path = f'{label training}.csv'
test_features_path = f'{data features test}.npy'
test_labels_path = f'{label test}.csv'
train_dataset = USRADataset(train_labels_path, train_features_path)
val_dataset = USRADataset(test_labels_path, test_features_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
 collate_fn=USRADataset_collate)
test_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
 collate_fn=USRADataset_collate)

# For the data setting and model training:
# Please notice that the current code is for the paper settings, but due to the different features dimensions and model structure,
# you need to adjust the feature dimension to make sure that the code can be run correctly
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BaseCNN_Conv(173, 128, kernel_size=3, padding=2, dilation=1))
        self.layer2 = nn.Sequential(
            BaseCNN_Conv(128, 256, kernel_size=3, padding=2, dilation=1))
        self.layer3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze()
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.LSTM(input_size=1025, hidden_size=256))
        self.layer2 = nn.Sequential(
            nn.LSTM(input_size=256, hidden_size=256))
        self.layer3 = nn.Sequential(
            nn.Linear(256,512),nn.ReLU(),nn.BatchNorm1d(173))
        self.linear = nn.Sequential(
            nn.Linear(512, 128),nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(173, 1)

    def forward(self, x):
        out1, state1 = self.layer1(x)
        out2, state2 = self.layer2(out1)
        out = self.layer3(out2)
        out = self.linear(out).squeeze()
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=5, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.layer1 = nn.Sequential(
            self.transformer_encoder)
        self.layer4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x).transpose(2,1)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity


model = Transformer().to(device)
# -------------------------------------------------------------------#
#   Determine the current batch_size and adaptively adjust the learning rate
# -------------------------------------------------------------------#
Init_lr             = 5e-4
Min_lr              = Init_lr * 0.01
optimizer_type      = "adam"
momentum            = 0.9
weight_decay        = 0.0003
lr_decay_type       = 'cos'
nbs = 64
lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# Loss and optimizer
criterion_r = nn.SmoothL1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
R2_max = 0.6
R2_list = []
MAE_list = []
MSE_list = []
RMSE_list = []
total_step = len(train_loader)
# # ---------------------------------------#
# #   Optimizer selection based on optimizer_type
# # ---------------------------------------#
optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                     weight_decay=weight_decay)
}[optimizer_type]
#
# # ---------------------------------------#
# #   Formula for obtaining a decrease in the learning rate
# # ---------------------------------------#
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, num_epochs)
for epoch in range(num_epochs):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    model.train()
    for i, (images,labels_intensity) in enumerate(train_loader):
        images = images.to(device)

        labels_intensity = labels_intensity.to(torch.float32)
        labels_intensity = labels_intensity.to(device)
        # Forward pass
        rainfall_intensity = model(images)
        r_loss = criterion_r(rainfall_intensity, labels_intensity.view([-1, 1]))
        loss = r_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    torch.save(model.state_dict(), 'model_epoch_R.ckpt')

    model.load_state_dict(torch.load('model_epoch_R.ckpt'))
    model.eval()

    acoustic_feaure = val_dataset.feature.transpose(0, 2, 1)
    outputs = []
    step = 16
    with torch.no_grad():
        acoustic_feaure = torch.tensor(acoustic_feaure).cuda()
        for index in tqdm(range(0,acoustic_feaure.shape[0],step)):
            if index ==0:
                rainfall_intensity = model(acoustic_feaure[index:step].to(torch.float))
                outputs = rainfall_intensity
            elif index>0 and index != acoustic_feaure.shape[0]-acoustic_feaure.shape[0]%step:
                rainfall_intensity = model(acoustic_feaure[index:index+step].to(torch.float))
                outputs = torch.cat((outputs,rainfall_intensity))
            elif index == acoustic_feaure.shape[0]-acoustic_feaure.shape[0]%step:
                rainfall_intensity = model(acoustic_feaure[index:acoustic_feaure.shape[0]].to(torch.float))
                outputs = torch.cat((outputs, rainfall_intensity))
    outputs = np.array(outputs.squeeze().cpu(),dtype=float)
    labels = val_dataset.label['RAINFALL INTENSITY'].to_numpy()
    MSE = mean_squared_error(labels, outputs)
    RMSE = np.sqrt(mean_squared_error(labels, outputs))
    MAE = mean_absolute_error(labels, outputs)
    R2 = r2_score(labels, outputs)
    print(R2)
    R2_list.append(R2)
    RMSE_list.append(RMSE)
    MSE_list.append(MSE)
    MAE_list.append(MAE)
    if R2>R2_max:
        torch.save(model.state_dict(), 'model_epoch_best_R.ckpt')
        R2_max=R2
        result_draw = result_show(labels,outputs,R2,RMSE,MSE,MAE)
        result_draw.draw()
