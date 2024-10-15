# -*- coding: utf-8 -*-
# @Time : 2023/1/8 19:51 
# @Author : Mingzheng 
# @File : dataloader.py
# @desc :

import math
import random

import numpy as np
import torch
import torchaudio.transforms
from torch.utils.data.dataset import Dataset
import librosa
import os
import pandas as pd
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class USRADataset(Dataset):
    def __init__(self,label_path,feature_path):
        super(USRADataset,self).__init__()
        self.label = pd.read_csv(label_path)
        self.feature = np.load(feature_path)
        self.length = self.label.shape[0]
        self.num_rows = 40
        self.num_columns = 173
        self.num_channels = 1
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # pay attention, for the code "self.feature[index].reshape({dimension})", you need to matches the acoustic features dimension to model input dimension
        # for example, if feature is MFCC, network is Transformer, you need to check the "n_model" and other setting of the Transformerencoder, to make sure that
        # the MFCC feature array could be correctly send to model and do a forward calculation.
        feature_item = self.feature[index].reshape({dimension})
        rainfall_intensity = self.label.iloc[index]['RAINFALL INTENSITY']
        return feature_item,rainfall_intensity

def USRADataset_collate(batch):
    features,batch_rainfall_intensities = [],[]

    for feature,batch_rainfall_intensity in batch:
        features.append(feature)
        batch_rainfall_intensities.append(batch_rainfall_intensity)
    features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    batch_rainfall_intensities = torch.from_numpy(np.array(batch_rainfall_intensities)).type(torch.FloatTensor)

    return features,batch_rainfall_intensities
