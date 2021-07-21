import numpy as np
import pandas as pd
import collections
import os
import functools
import json
import sys
import logging
import pickle
import re
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import time
import scipy.signal
import math

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import random 
from transformers import *
from transformers.modeling_bert import BertConfig,BertLayerNorm
from transformers.activations import gelu, gelu_new, swish

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class sampleDataset(Data.Dataset):
    def __init__(self,day2action):
        super(sampleDataset,self).__init__()
#         self.user_idx=torch.tensor([user2idx[e[0].split("_")[0]] for day in day2action for e in day]).long()
        self.labels=torch.tensor([e[1]['label'] for day in day2action for e in day]).long()

        self.location_feature=[ torch.tensor(e[1]['masked_location_feature']).float() for day in day2action for e in day]
        self.location_fre_feature=[ torch.tensor(e[1]['masked_location_fre_feature']).float() for day in day2action for e in day]
        self.mouse_feature=[ torch.tensor(e[1]['masked_mouse_feature']).float() for day in day2action for e in day]
        self.mouse_fre_feature=[ torch.tensor(e[1]['masked_mouse_fre_feature']).float() for day in day2action for e in day]
        
        self.location_token = [ torch.tensor(e[1]['masked_location_token']).long() for day in day2action for e in day]
        self.location_dis = [ torch.tensor(e[1]['masked_location_dis']).long() for day in day2action for e in day]
        self.location_f_token = [ torch.tensor(e[1]['masked_location_f_token']).long() for day in day2action for e in day]
        self.location_f_dis = [ torch.tensor(e[1]['masked_location_f_dis']).long() for day in day2action for e in day]

        self.mouse_token = [ torch.tensor(e[1]['masked_mouse_token']).long() for day in day2action for e in day]
        self.mouse_dis = [ torch.tensor(e[1]['masked_mouse_dis']).long() for day in day2action for e in day]
        self.mouse_f_token = [ torch.tensor(e[1]['masked_mouse_f_token']).long() for day in day2action for e in day]
        self.mouse_f_dis = [ torch.tensor(e[1]['masked_mouse_f_dis']).long() for day in day2action for e in day]
        idx = 0
        for i in range(len(day2action)):
            if len(day2action[i])>10:
                idx = i
                break
        if 'map_idx' in day2action[idx][0][1]:
            self.map_idx = torch.tensor([e[1]['map_idx'] for day in day2action for e in day]).long()
            self.weekday_idx = torch.tensor([e[1]['weekday_idx']  for day in day2action for e in day]).long()
            self.time_idx = torch.tensor([e[1]['time_idx']  for day in day2action for e in day]).long()
        if 'masked_location_xy' in day2action[idx][0][1]:
            self.location_xy = [ torch.tensor(e[1]['masked_location_xy']).long() for day in day2action for e in day] 
        if 'masked_mouse_xy' in day2action[idx][0][1]:
            self.mouse_xy = [ torch.tensor(e[1]['masked_mouse_xy']).long() for day in day2action for e in day] 
        if 'masked_location_f_xy' in day2action[idx][0][1]:
            self.location_f_xy = [ torch.tensor(e[1]['masked_location_f_xy']).long() for day in day2action for e in day] 
        if 'masked_mouse_f_xy' in day2action[idx][0][1]:
            self.mouse_f_xy = [ torch.tensor(e[1]['masked_mouse_f_xy']).long() for day in day2action for e in day] 
    def __getitem__(self,idx):

        val = {"location_feature":self.location_feature[idx],"location_fre_feature":self.location_fre_feature[idx],\
             "mouse_feature":self.mouse_feature[idx],"mouse_fre_feature":self.mouse_fre_feature[idx],\
             'location_token':self.location_token[idx],'location_dis':self.location_dis[idx],\
             'location_f_token':self.location_f_token[idx],'location_f_dis':self.location_f_dis[idx],\
             'mouse_token':self.mouse_token[idx],'mouse_dis':self.mouse_dis[idx],\
             'mouse_f_token':self.mouse_f_token[idx],'mouse_f_dis':self.mouse_f_dis[idx],\
             'labels': self.labels[idx]}
        if 'map_idx' in self.__dict__:
            val['map_idx'] = self.map_idx[idx]
            val['weekday_idx'] = self.weekday_idx[idx]
            val['time_idx'] = self.time_idx[idx]
        if 'location_xy' in self.__dict__:
            val['location_xy'] = self.location_xy[idx]
            val['mouse_xy'] = self.mouse_xy[idx]
        if 'location_f_xy' in self.__dict__:
            val['location_f_xy'] = self.location_f_xy[idx]
            val['mouse_f_xy'] = self.mouse_f_xy[idx]
        return val
    def __len__(self):
        return len(self.labels)
def collate_fn(train_data,use_token=False,fil=False):
    location_xy = None
    mouse_xy = None
    if not use_token:
        location_feature = nn.utils.rnn.pad_sequence([e['location_feature'][:8000] for e in train_data],batch_first=True)
        location_length=torch.tensor([len(e['location_feature'][:8000]) for e in train_data])
        location_dis =  nn.utils.rnn.pad_sequence([e['location_dis'][:8000] for e in train_data],batch_first=True)

        mouse_feature = nn.utils.rnn.pad_sequence([e['mouse_feature'][:8000] for e in train_data],batch_first=True)
        mouse_length=torch.tensor([len(e['mouse_feature'][:8000]) for e in train_data])
        mouse_dis = nn.utils.rnn.pad_sequence([e['mouse_dis'][:8000] for e in train_data],batch_first=True)
    else:
        if not fil:
            location_feature = nn.utils.rnn.pad_sequence([e['location_token'][:8000] for e in train_data],batch_first=True)
            location_length=torch.tensor([len(e['location_token'][:8000]) for e in train_data])
            location_dis =  nn.utils.rnn.pad_sequence([e['location_dis'][:8000] for e in train_data],batch_first=True)
            if 'location_xy' in train_data[0]:
                location_xy =  nn.utils.rnn.pad_sequence([e['location_xy'][:8000] for e in train_data],batch_first=True)
            mouse_feature = nn.utils.rnn.pad_sequence([e['mouse_token'][:8000] for e in train_data],batch_first=True)
            mouse_length=torch.tensor([len(e['mouse_token'][:8000]) for e in train_data])
            mouse_dis = nn.utils.rnn.pad_sequence([e['mouse_dis'][:8000] for e in train_data],batch_first=True)
            if 'mouse_xy' in train_data[0]:
                mouse_xy =  nn.utils.rnn.pad_sequence([e['mouse_xy'][:8000] for e in train_data],batch_first=True)
        else:
            location_feature = nn.utils.rnn.pad_sequence([e['location_f_token'][:2000] for e in train_data],batch_first=True)
            location_length=torch.tensor([len(e['location_f_token'][:2000]) for e in train_data])
            location_dis =  nn.utils.rnn.pad_sequence([e['location_f_dis'][:2000] for e in train_data],batch_first=True)
            if 'location_f_xy' in train_data[0]:
                location_xy =  nn.utils.rnn.pad_sequence([e['location_f_xy'][:2000] for e in train_data],batch_first=True)
            mouse_feature = nn.utils.rnn.pad_sequence([e['mouse_f_token'][:2000] for e in train_data],batch_first=True)
            mouse_length=torch.tensor([len(e['mouse_f_token'][:2000]) for e in train_data])
            mouse_dis = nn.utils.rnn.pad_sequence([e['mouse_f_dis'][:2000] for e in train_data],batch_first=True)
            if 'mouse_f_xy' in train_data[0]:
                mouse_xy =  nn.utils.rnn.pad_sequence([e['mouse_f_xy'][:2000] for e in train_data],batch_first=True)
            
    location_fre_feature = torch.stack([e['location_fre_feature'] for e in train_data]).float()
    mouse_fre_feature = torch.stack([e['mouse_fre_feature'] for e in train_data]).float()
    
    labels = torch.stack([e['labels'] for e in train_data]).long()
    if location_feature.shape[1]%8:
        pad_len = 8-location_feature.shape[1]%8
        location_feature = F.pad(location_feature,(0,pad_len))
        if location_xy is not None:
            location_xy  =F.pad(location_xy.permute(0,2,1),(0,pad_len)).permute(0,2,1)
    if mouse_feature.shape[1]%8:
        pad_len = 8-mouse_feature.shape[1]%8
        mouse_feature = F.pad(mouse_feature,(0,pad_len))
        if mouse_xy is not None:
            mouse_xy  =F.pad(mouse_xy.permute(0,2,1),(0,pad_len)).permute(0,2,1)
    val = {'location_feature':location_feature,'mouse_feature':mouse_feature,\
             'location_length':location_length,'mouse_length':mouse_length,\
              'location_dis':location_dis,'mouse_dis':mouse_dis,\
             'location_fre_feature':location_fre_feature,'mouse_fre_feature':mouse_fre_feature,\
             'labels': labels}
    if 'map_idx' in train_data[0]:
        map_idxes = torch.stack([e['map_idx'] for e in train_data]).long()
        weekday_idxes = torch.stack([e['weekday_idx'] for e in train_data]).long()
        time_idxes = torch.stack([e['time_idx'] for e in train_data]).long()
        val['map_idx'] = map_idxes
        val['weekday_idx'] = weekday_idxes
        val['time_idx'] = time_idxes
    if location_xy is not None:
        val['location_xy'] = location_xy
    if mouse_xy is not None:
        val['mouse_xy'] = mouse_xy
    return val