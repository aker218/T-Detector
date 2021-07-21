#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm
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
import argparse

# In[126]:

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
    parser.add_argument("-M","--mouse_dir",type=str,help="鼠标轨迹样本文件夹",default = "./data/04_mouse_trajectory_anti_cheat_dataset/mouse_trajectory_dataset/")
    parser.add_argument("-ML","--mouse_label_dir",type=str,help="鼠标轨迹样本标签文件夹",default = "./data/04_mouse_trajectory_anti_cheat_dataset/")
    parser.add_argument("-L","--location_dir",type=str,help="人物轨迹样本文件夹",default = "./data/02_movement_trajectory_anti_cheat_dataset/movement_trajectory_dataset/")
    parser.add_argument("-LL","--location_label_dir",type=str,help="人物轨迹样本标签文件夹",default = "./data/02_movement_trajectory_anti_cheat_dataset/")
    parser.add_argument("-P0","--p0",type=float,default = 0.351 ) 
    parser.add_argument("-P1","--p1",type=float,default = 0.157)

    args = parser.parse_args()
    # args.output_data_dir = "new_processed_data"
    # args.mouse_dir = "new_dataset/mouse"
    # args.location_dir = "new_dataset/move"
    # args.mouse_label_dir = "new_dataset"
    # args.location_label_dir = "new_dataset"
    # args.p0 = 0
    # args.p1 = 0
    output_data_dir = args.output_data_dir
    mouse_dir = args.mouse_dir
    mouse_label_dir = args.mouse_label_dir
    location_dir = args.location_dir
    location_label_dir = args.location_label_dir


    # In[52]:


    def process_raw_data(data_dir,label_dir):
        label_map=dict()
        if os.path.exists(os.path.join(label_dir,"label_neg.json")):
            label_neg=json.load(open(os.path.join(label_dir,"label_neg.json"),'r'))
            for e in label_neg:
                label_map[e]=0
            label_pos=json.load(open(os.path.join(label_dir,"label_pos.json"),'r'))
            for e in label_pos:
                label_map[e]=1
        else:
            tmp = pd.read_csv(os.path.join(label_dir,"label.csv"))
            for e in tmp.values:
                label_map[e[0]] = e[1]
        data_files=os.listdir(data_dir)
        samples=[[] for i in range(31)]
        for file in tqdm(data_files):
            sample=np.asarray([[e['x'],e['y'],e['tm']] for e in json.load(open(os.path.join(data_dir,file),'r'))])
            begin=sample[:,2][0]
            end=sample[:,2][-1]
            user_idx,map_idx,_=file.split("_")
            map_idx=int(map_idx)
            time_dis=np.concatenate([[0],sample[1:,2]-sample[:-1,2]])
            abnormal_point=np.arange(sample.shape[0]-1)[(np.abs((sample[1:,2]-sample[:-1,2])-400)>=10)]+1
            pre_sample=sample.copy()
            sample[:,2]=time_dis
            label=label_map[file.strip(".json")]
            full_sample=[sample,[user_idx,map_idx,begin,end,file.strip(".json")],label]
            day_idx=(time.localtime(begin//1000).tm_mday%31)
            samples[day_idx].append(full_sample)
        return samples


    # In[56]:



    if not os.path.exists(os.path.join(output_data_dir,"mouse_samples.pickle")):
        data_dir=mouse_dir
        label_dir=mouse_label_dir
        mouse_samples=process_raw_data(data_dir,label_dir)
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir,exist_ok=True)
        pickle.dump(mouse_samples,open(os.path.join(output_data_dir,"mouse_samples.pickle"),"wb"))
    else:
        print("loading mouse samples...")
        mouse_samples=pickle.load(open(os.path.join(output_data_dir,"mouse_samples.pickle"),"rb"))


    # In[66]:


    if not os.path.exists(os.path.join(output_data_dir,"location_samples.pickle")):
        data_dir=location_dir
        label_dir=location_label_dir
        location_samples=process_raw_data(data_dir,label_dir)
        pickle.dump(location_samples,open(os.path.join(output_data_dir,"location_samples.pickle"),"wb"))
    else:
        print("loading location_samples...")
        location_samples=pickle.load(open(os.path.join(output_data_dir,"location_samples.pickle"),"rb"))


    # In[88]:


    if not os.path.exists(os.path.join(output_data_dir,"action_sequence.pickle")):
        user2action=collections.defaultdict(dict)
        for idx,samples in enumerate(location_samples):
            for sample in tqdm(samples):

                user=sample[1][0]
                map_idx=sample[1][1]
                begin_idx,end_idx=sample[1][2:4]
                user_idx=sample[1][4]
                label=sample[2]
                location_data=sample[0]
                time_idx=f"{begin_idx}_{end_idx}"
                user2action[user][user_idx]={"label":label,"location_data":location_data,"day":idx,                                         "map_idx":map_idx,"location_time_duration":time_idx}
        for idx,samples in enumerate(mouse_samples):
            for sample in tqdm(samples):
                user=sample[1][0]
                map_idx=sample[1][1]
                begin_idx,end_idx=sample[1][2:4]
                user_idx=sample[1][4]
                label=sample[2]
                mouse_data=sample[0]
                time_idx=f"{begin_idx}_{end_idx}" 
                if user in user2action and user_idx in user2action[user]:
                    assert user2action[user][user_idx]["label"]==label 
    #                 assert user2action[user][user_idx]["day"]==idx
                    assert user2action[user][user_idx]["map_idx"]==map_idx
                    user2action[user][user_idx]['mouse_data']=mouse_data
                    user2action[user][user_idx]['mouse_time_duration']=time_idx
                else:
                    user2action[user][user_idx]={"label":label,"mouse_data":location_data,"day":idx,                                             "map_idx":map_idx,"mouse_time_duration":time_idx}
        idx2user=list(user2action.keys())
        day2action=[[ ] for i in range(31)]
        for user in user2action:
            for key in user2action[user]:
                idx=user2action[user][key]['day']
                day2action[idx].append((key,user2action[user][key]))
        user2idx={user:idx for idx,user in enumerate(idx2user)}
        print("dumping ...")
        pickle.dump([idx2user,user2idx,user2action,day2action],open(os.path.join(output_data_dir,"action_sequence.pickle"),"wb"))
    else:
        print("loading sample...")
        idx2user,user2idx,user2action,day2action=pickle.load(open(os.path.join(output_data_dir,"action_sequence.pickle"),"rb"))



    # In[129]:


    p0,p1 =args.p0,args.p1

    if not os.path.exists(os.path.join(output_data_dir,"mask_idx.pickle")):
        mask_idxs = []
        for day in tqdm(day2action):
            mask_idxs.append([])
            for i in tqdm(range(len(day))):
                label = day[i][1]['label']
                if label==1: 
                    if np.random.binomial(1,p1):
                        mask_idxs[-1].append(0)
                    else:
                        mask_idxs[-1].append(1)
                else:
                    if np.random.binomial(1,p0):
                        mask_idxs[-1].append(2)
                    else:
                        mask_idxs[-1].append(3)
        pickle.dump(mask_idxs,open(os.path.join(output_data_dir,"mask_idx.pickle"),"wb"))
    else:
        print("loading...")
        mask_idxs = pickle.load(open(os.path.join(output_data_dir,"mask_idx.pickle"),"rb"))

