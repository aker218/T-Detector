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
from tqdm import  tqdm
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


# In[3]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
parser.add_argument('-M',"--use_merge_split", default=False, action='store_true')

args = parser.parse_args()


# In[4]:


data_dir = args.output_data_dir
# data_dir = "pkdd_data"
# data_dir = "ms_data"
use_merge_split = args.use_merge_split
suffix = ""
if use_merge_split:
    suffix = "_merge_split"
# if use_cluster:
#     if with_weight:
#         suffix = "_cluster_w"
#     else:
#         suffix = "_cluster"
if use_merge_split:
    word2vec_config = {'location':[0,50000,0,50000,20,200,200],'mouse':[-5000,50000,-1000,35000,10,200,200]} #merge_and_split 专用
else:
    word2vec_config = {'location':[0,50000,0,50000,20,100,100],'mouse':[-5000,50000,-1000,35000,10,100,100]}


# In[5]:


print("loading sample...")
idx2user,user2idx,user2action,day2action=pickle.load(open(os.path.join(data_dir,"action_sequence.pickle"),"rb"))


# In[6]:


def raw_xy2token(sample,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    return ((x2-x1)//width) *((sample[:,1]-y1)//height)+(sample[:,0]-x1)//width
def token2raw_xy(token_sequence,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    x_y_sequence = get_x_y(token_sequence,x1,x2,y1,y2,threshold,width,height)
    x_y_sequence[:,0] *=width
    x_y_sequence[:,0] +=x1
    x_y_sequence[:,1] *=height
    x_y_sequence[:,1] +=y1
    return x_y_sequence
def process(sample,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    sample = sample.copy()
    sample[:,-1]=np.cumsum(sample[:,-1])
    sample[:,0][sample[:,0]>=x2]=x2
    sample[:,0][sample[:,0]<x1]=x1
    sample[:,1][sample[:,1]>=y2]=y2
    sample[:,1][sample[:,1]<y1]=y1 
    dis = abs(sample[1:,]-sample[:-1])
    abs_dis = np.sqrt(dis[:,0]**2+dis[:,1]**2)
    filtered_sample = sample[np.concatenate([[threshold],abs_dis])>=threshold]
    filtered_sample[1:,-1] = filtered_sample[1:,-1] - filtered_sample[:-1,-1]
    token_sequence = ((x2-x1)//width) *((filtered_sample[:,1]-y1)//height)+(filtered_sample[:,0]-x1)//width
    raw_token_sequence = ((x2-x1)//width) *((sample[:,1]-y1)//height)+(sample[:,0]-x1)//width
    return filtered_sample,token_sequence,raw_token_sequence
def get_idx(sample,itype = 'location'):
    if itype == 'location':
        map_idx = sample[1]['map_idx']
        weekday_idx = int(time.localtime(float(sample[1]['location_time_duration'].split("_")[0])//1000).tm_wday<=4)
        time_idx = time.localtime(float(sample[1]['location_time_duration'].split("_")[0])//1000).tm_hour
    else:
        map_idx = sample[1]['map_idx']
        weekday_idx = int(time.localtime(float(sample[1]['mouse_time_duration'].split("_")[0])//1000).tm_wday<=4)
        time_idx = time.localtime(float(sample[1]['mouse_time_duration'].split("_")[0])//1000).tm_hour
    return [map_idx,weekday_idx,time_idx]
def get_x_y(token_sequence,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    x = token_sequence%((x2-x1)//width)
    y = token_sequence//((x2-x1)//width)
    return np.stack([x,y],axis=1)
def idx2xy(idx,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    x = idx%((x2-x1)//width)
    y = idx//((x2-x1)//width)
    return (x,y)
def xy2idx(x,y,x1,x2,y1,y2,threshold=20,width = 100,height = 100):
    return ((x2-x1)//width) *y+x


# In[7]:


location_datas = []

location_idxs = []



dis_grid = 400
# dis_grid = 300
alpha = 0.8
for day in day2action:
    for sample in day:
        if 'location_data' in sample[1]:
            location_datas.append(sample[1]['location_data'].copy())
            location_idxs.append(get_idx(sample,itype = 'location'))
        if 'location_raw_token' in sample[1]:
            location_tokens.append(sample[1]['location_raw_token'])
            location_time_dis.append(np.cumsum(sample[1]['location_data'][:,-1]/dis_grid))
            
filtered_location_datas = []
location_tokens = []
location_xys = []
location_time_dis = []
if len(location_tokens)==0:
    for sample in tqdm(location_datas):
        filtered_sample,token_sequence,raw_token_sequence = process(sample,*word2vec_config['location'])
        filtered_location_datas.append(filtered_sample)
        location_tokens.append(token_sequence)
#         location_xys.append(get_x_y(token_sequence,*word2vec_config['location']))
        location_time_dis.append(np.cumsum(filtered_sample[:,-1]/dis_grid))


# In[8]:


mouse_datas = []
mouse_idxs = []
filtered_mouse_datas = []
mouse_xys = []
mouse_tokens = []
mouse_time_dis = []
for day in day2action:
    for sample in day:
        if 'mouse_data' in sample[1]:
            mouse_datas.append(sample[1]['mouse_data'].copy())
            mouse_idxs.append(get_idx(sample,itype = 'mouse'))
            
for sample in tqdm(mouse_datas):
    filtered_sample,token_sequence,raw_token_sequence = process(sample,*word2vec_config['mouse'])
    mouse_tokens.append(token_sequence)
    filtered_mouse_datas.append(filtered_sample)
#     mouse_xys.append(get_x_y(token_sequence,*word2vec_config['mouse']))
    mouse_time_dis.append(np.cumsum(filtered_sample[:,-1]/400))
# location_metadata = [a+[b] for a,b in zip(location_idxs,location_time_dis)]
# mouse_metadata = [a+[b] for a,b in zip(mouse_idxs,mouse_time_dis)]


# In[9]:


def word2vec_preprocess(raw_dataset,dis_dataset,subsample = False):
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 1, counter.items()))
    idx_to_token = ['pad']+[tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
               for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])

    def discard(idx):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / counter[idx_to_token[idx]] * num_tokens)

    subsampled_dataset,subsampled_dis_dataset = [],[]
    for st,dis in zip(dataset,dis_dataset):
        subsampled_dataset.append([])
        subsampled_dis_dataset.append([])
        for tk,dis_tk in zip(st,dis):
            if not subsample or (not discard(tk)):
                subsampled_dataset[-1].append(tk)
                subsampled_dis_dataset[-1].append(dis_tk)
    return dataset,subsampled_dataset,subsampled_dis_dataset,idx_to_token,token_to_idx,counter


# In[12]:


def get_labels(data_idx_list,label_list):
    n = len(label_list)
    def dfs(idx):
        if (idx==n-1):
            return label_list[idx]
        next_labels = dfs(idx+1)
        cur_label = np.zeros(len(data_idx_list[idx]))
        next_idxs = data_idx_list[idx+1]
        for idxs,label in zip(next_idxs,next_labels):
            cur_label[idxs] = label
        return cur_label
    return dfs(0)

def merge_and_split(location_token2raw_xy,min_count = 25,restrict_count = 2000,restrict_step = 2,itype = "location"):
    new_location_token2raw_xy = collections.defaultdict(set)
    token2new_token = {}
    n = 0
    for i,t in tqdm(enumerate(location_token2raw_xy)):
        if t in token2new_token:
            continue
        x1 = word2vec_config[itype][0]
        x2 = word2vec_config[itype][1]
        y1 = word2vec_config[itype][2]
        y2 = word2vec_config[itype][3]
        threshold= word2vec_config[itype][4]
        width= word2vec_config[itype][5]
        height= word2vec_config[itype][6]
        cur_w = width
        cur_h = height
        tmp = []
        bx,by = token2raw_xy(np.array([t]),x1,x2,y1,y2,threshold,cur_w,cur_h)[0]
        ex = bx + width
        ey = by + height
        tmp.append(location_token2raw_xy[t])
        count = np.asarray(([len(e) for e in tmp])).mean()
        e_t = t
        res = tmp[0]
        if count<min_count:
            n +=1
            t1 = raw_xy2token(np.array([[bx+width,by]]),x1,x2,y1,y2,threshold,cur_w,cur_h)[0]
            t2 = raw_xy2token(np.array([[bx,by+height]]),x1,x2,y1,y2,threshold,cur_w,cur_h)[0]
            t3 = raw_xy2token(np.array([[ex,ey]]),x1,x2,y1,y2,threshold,cur_w,cur_h)[0]
            res = tmp[0]
            new_res = res.copy()
            for e in [t1,t2,t3]:
                if e in location_token2raw_xy:
                    new_res |=location_token2raw_xy[e]
            if len(new_res) >restrict_count:
                token2new_token[t] = t
                new_location_token2raw_xy[str(t)+"_"+str(e_t)] = res
                continue
            for e in [t,t1,t2,t3]:
                if e in location_token2raw_xy:
                    token2new_token[e] = t
            e_t = t3
            res = new_res
        new_location_token2raw_xy[str(t)+"_"+str(e_t)] = res

    final_location_token2raw_xy = collections.defaultdict(set)
    for i,t_et in tqdm(enumerate(new_location_token2raw_xy)):
        t,et = [int(e) for e in t_et.split("_")]
        if(t!=et):
            final_location_token2raw_xy[str(t)+"_0"] = new_location_token2raw_xy[t_et]
            continue
        x1 = word2vec_config[itype][0]
        x2 = word2vec_config[itype][1]
        y1 = word2vec_config[itype][2]
        y2 = word2vec_config[itype][3]
        threshold= word2vec_config[itype][4]
        width= word2vec_config[itype][5]
        height= word2vec_config[itype][6]
        cur_w = width
        cur_h = height
        tmp = []
        bx,by = token2raw_xy(np.array([t]),x1,x2,y1,y2,threshold,cur_w,cur_h)[0]
        ex = bx + width
        ey = by + height
        tmp.append(location_token2raw_xy[t])
        count = np.asarray(([len(e) for e in tmp])).mean()
        length = 1
        step = 0
        while count>restrict_count and step<restrict_step:
            cur_w /=2
            cur_h /=2
            count = count/4
            step +=1
            length *=4
        new_tmp = [set() for i in range(length)]
#         print("===")
#         print(t)
#         print(length)
#         print(tmp)
#         print(cur_w,cur_h)
        for token_set in tmp:
            for x,y in token_set:
                token_idx = ((int)(((y-by)//cur_h)*(ex-bx)//cur_w+(x-bx)//cur_w))%len(new_tmp)
#                 print(token_idx)
#                 print(y,by,cur_h,x,bx,cur_w)
                new_tmp[token_idx].add((x,y))
        tmp = new_tmp

        for j,e in enumerate(tmp):
            final_location_token2raw_xy[str(t)+"_"+str(j)] = e
    return final_location_token2raw_xy


# In[13]:


if args.use_merge_split:
    if not os.path.exists(os.path.join(data_dir,"merge_split.pk")):
        location_raw_xy_vocab = np.unique(np.concatenate(filtered_location_datas)[:,:2],axis=0)
        location_raw_token = raw_xy2token(location_raw_xy_vocab,*word2vec_config['location'])
        location_token2raw_xy = collections.defaultdict(set)
        for t,(x,y) in tqdm(zip(location_raw_token,location_raw_xy_vocab)):
            location_token2raw_xy[t].add((x,y))
        mouse_raw_xy_vocab = np.unique(np.concatenate(filtered_mouse_datas)[:,:2],axis=0)
        mouse_raw_token = raw_xy2token(mouse_raw_xy_vocab,*word2vec_config['mouse'])
        mouse_token2raw_xy = collections.defaultdict(set)
        for t,(x,y) in tqdm(zip(mouse_raw_token,mouse_raw_xy_vocab)):
            mouse_token2raw_xy[t].add((x,y))
        #min count 25 2.1
        final_location_token2raw_xy = merge_and_split(location_token2raw_xy,min_count = 25,restrict_count = 2000,restrict_step = 2,itype = "location")
        final_mouse_token2raw_xy = merge_and_split(mouse_token2raw_xy,min_count = 2.1,restrict_count = 100,restrict_step = 2,itype = "mouse")

        location_raw_xy2token = collections.defaultdict(int)
        for k,v in tqdm(final_location_token2raw_xy.items()):
            for e in v:
                location_raw_xy2token[e] = k
        mouse_raw_xy2token = collections.defaultdict(int)
        for k,v in tqdm(final_mouse_token2raw_xy.items()):
            for e in v:
                mouse_raw_xy2token[e] = k
        pickle.dump([location_raw_xy2token,mouse_raw_xy2token],open(os.path.join(data_dir,"merge_split.pk"),"wb"))
    else:
        print("loading...")
        location_raw_xy2token,mouse_raw_xy2token = pickle.load(open(os.path.join(data_dir,"merge_split.pk"),"rb"))


# In[14]:



subsample = True
flag = "" if not subsample else "subsampled_"
os.makedirs(os.path.join(data_dir,"time-dis-vec/"),exist_ok=True)
if use_merge_split:
    vocab_name = "time-dis-vec/vocab%s.pk"%(suffix)
    location_tokens = [[location_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_location_datas)]
    mouse_tokens = [[mouse_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_mouse_datas)]


# In[17]:


def get_centers_and_contexts(dataset,dis_dataset,idx_dataset, max_window_size):
    centers, contexts,dis_list,idx_list = [], [],[],[]
    for st,dis,id_list in tqdm(zip(dataset,dis_dataset,idx_dataset)):
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            idx_center = dis[center_i]
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
            dis_list.append([abs(dis[idx]-idx_center) for idx in indices])
            idx_list.append(id_list)
    return centers, contexts,dis_list,idx_list
def get_negatives(all_contexts, sampling_weights, K, ntype = 'skip_gram'):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    if ntype =='skip_gram':
        for contexts in all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                    # 为了高效计算，可以将k设得稍大一点
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
    else:
        for center in tqdm(all_contexts):
            negatives = []
            while len(negatives) <  K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                    # 为了高效计算，可以将k设得稍大一点
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词
                if neg != center:
                    negatives.append(neg)
            all_negatives.append(negatives)
    return all_negatives


# In[15]:


if not os.path.exists(os.path.join(data_dir,"time-dis-vec/"+flag+"location_vocab.pickle"+suffix)):

    tokens = location_tokens
    dataset,subsampled_dataset,subsampled_dis_dataset,idx_to_token,token_to_idx,counter = word2vec_preprocess(tokens,location_time_dis,subsample = subsample)

    pickle.dump([dataset,subsampled_dataset,subsampled_dis_dataset],open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_dataset.pickle"+suffix),"wb"))
    pickle.dump([idx_to_token,token_to_idx,counter],open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_vocab.pickle"+suffix),"wb"))
else:
    print("loading...")
    dataset,subsampled_dataset,subsampled_dis_dataset = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_dataset.pickle"+suffix),"rb"))
    idx_to_token,token_to_idx,counter = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_vocab.pickle"+suffix),"rb"))


# In[18]:


ntype = 'cbow'
if not os.path.exists(os.path.join(data_dir,"time-dis-vec/"+flag+"location_pair.pickle"+suffix)):
    all_centers, all_contexts,all_dis,all_idx = get_centers_and_contexts(subsampled_dataset,subsampled_dis_dataset,location_idxs, 20)

    sampling_weights = [counter[w]**0.75 for w in idx_to_token if w!='pad']
    if ntype =='skip_gram':
        all_negatives = get_negatives(all_contexts, sampling_weights, 5)
    else:
        all_negatives = get_negatives(all_centers, sampling_weights, 5,ntype = ntype)
    pickle.dump([all_centers,all_idx],open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_pair.pickle"+suffix),"wb"))
    pickle.dump(all_negatives,open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_negative.pickle"+suffix),"wb"))
    pickle.dump(all_contexts,open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_context.pickle"+suffix),"wb"))
    new_all_dis = []
    for e in tqdm(all_dis):
        new_all_dis.append(np.asarray(e))
    pickle.dump(new_all_dis,open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_dis.pickle"+suffix),"wb"))
else:
    print("loading...")
    all_centers,all_idx = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_pair.pickle"+suffix),"rb"))
    all_contexts = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_context.pickle"+suffix),"rb"))
    all_dis = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_dis.pickle"+suffix),"rb"))
    all_negatives = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"location_negative.pickle"+suffix),"rb"))
    


# In[19]:


if not os.path.exists(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_vocab.pickle"+suffix)):

    tokens = mouse_tokens
    dataset,subsampled_dataset,subsampled_dis_dataset,idx_to_token,token_to_idx,counter = word2vec_preprocess(tokens,mouse_time_dis,subsample = subsample)
    os.makedirs(os.path.join(data_dir,"time-dis-vec/"),exist_ok=True)
    pickle.dump([dataset,subsampled_dataset,subsampled_dis_dataset],open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_dataset.pickle"+suffix),"wb"))
    pickle.dump([idx_to_token,token_to_idx,counter],open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_vocab.pickle"+suffix),"wb"))
else:
    print("loading...")
    dataset,subsampled_dataset,subsampled_dis_dataset = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_dataset.pickle"+suffix),"rb"))
    idx_to_token,token_to_idx,counter = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_vocab.pickle"+suffix),"rb"))


# In[20]:


ntype = 'cbow'
if not os.path.exists(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_pair.pickle"+suffix)):
    all_centers, all_contexts,all_dis,all_idx = get_centers_and_contexts(subsampled_dataset,subsampled_dis_dataset,mouse_idxs, 20)

    sampling_weights = [counter[w]**0.75 for w in idx_to_token if w!='pad']
    if ntype =='skip_gram':
        all_negatives = get_negatives(all_contexts, sampling_weights, 5)
    else:
        all_negatives = get_negatives(all_centers, sampling_weights, 5,ntype = ntype)
    pickle.dump([all_centers,all_idx],open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_pair.pickle"+suffix),"wb"))
    pickle.dump(all_negatives,open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_negative.pickle"+suffix),"wb"))
    pickle.dump(all_contexts,open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_context.pickle"+suffix),"wb"))
    new_all_dis = []
    for e in tqdm(all_dis):
        new_all_dis.append(np.asarray(e))
    pickle.dump(new_all_dis,open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_dis.pickle"+suffix),"wb"))
else:
    print("loading...")
    all_centers,all_idx = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_pair.pickle"+suffix),"rb"))
    all_contexts = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_context.pickle"+suffix),"rb"))
    all_dis = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_dis.pickle"+suffix),"rb"))
    all_negatives = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+"mouse_negative.pickle"+suffix),"rb"))
    

