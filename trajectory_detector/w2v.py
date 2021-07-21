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


# In[2]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
parser.add_argument('-M',"--use_merge_split", default=False, action='store_true')

args = parser.parse_args()
# args.use_merge_split = True
# args.output_data_dir="./data/new_processed_data/"


# In[3]:


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


# In[4]:


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


# In[28]:


from gensim.models.word2vec import Word2Vec
window = 20
mincount = 1
def word2vec(se_feature_sequence, n_dim):
    sentences = []
    for a_user in se_feature_sequence:
#         print(a_user)
        sentence = []
        for item_id in a_user:
            sentence.append(str(item_id))
        sentences.append(sentence)
#     print(sentences)
    wv_model= Word2Vec(sentences, min_count=mincount, size = n_dim, window = window, sg = 1,workers=16)
    
    x_wv_list = []
#     for i, sentence in enumerate(sentences):
#         x_wv_list.append(wv_model[sentence])

    return x_wv_list, wv_model
token_trans = []
idx_embeddings = [0,0]

use_cluster = False
if use_merge_split:
    vocab_name = "w2v/vocab%s.pk"%(suffix)
    if not os.path.exists(os.path.join(data_dir,vocab_name)):
        location_tokens = [[location_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_location_datas)]
        mouse_tokens = [[mouse_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_mouse_datas)]

model_dir = data_dir.strip("/").split("/")[-1]+"_w2v/"
vocab_name = "w2v/vocab%s.pk"%(suffix)
if not os.path.exists(os.path.join(data_dir,vocab_name)):
    os.makedirs(os.path.join(data_dir,"w2v"),exist_ok=True)
    token_lists = [location_tokens,mouse_tokens]
else:
    token_lists = [0,0]


# In[30]:






keys = ['location','mouse']
wv_models = []

for i,(key,tokens) in enumerate(zip(keys,token_lists)):
    model_name = '%s_window%d_dim%d_mincount%d%s.pickle' % (key, window,100, mincount, suffix)
    if not os.path.exists('./models/'+model_dir+model_name):

        os.makedirs('./models/'+model_dir,exist_ok=True)
        logging.info('start %s wv' % key)
        if use_cluster:
            tokens = [[token_trans[i][token] for token in token_list] for token_list in tqdm(tokens)]
        wv, model = word2vec(tokens,100)

        pickle.dump(model, open('./models/'+model_dir+model_name, 'wb'))
        logging.info('finish %s wv' % key)
    else:
        model=pickle.load(open('./models/'+model_dir+model_name,'rb'))
        wv_models.append(model)

if not os.path.exists(os.path.join(data_dir,vocab_name)):
    if use_cluster:
        location_cluster = np.unique(list(location_trans.values()))
        location_cluster2idx = collections.defaultdict(int)
        for idx,token in enumerate(location_cluster):
            location_cluster2idx[int(token)] = idx+1
        location_vocab = np.unique(np.concatenate(location_tokens))
        location_token2idx = collections.defaultdict(int)
        for idx,token in enumerate(location_vocab):
            location_token2idx[int(token)] = location_cluster2idx[location_trans[token]]
        mouse_cluster = np.unique(list(mouse_trans.values()))
        mouse_cluster2idx = collections.defaultdict(int)
        for idx,token in enumerate(mouse_cluster):
            mouse_cluster2idx[int(token)] = idx+1
        mouse_vocab = np.unique(np.concatenate(mouse_tokens))
        mouse_token2idx = collections.defaultdict(int)
        for idx,token in enumerate(mouse_vocab):
            mouse_token2idx[int(token)] = mouse_cluster2idx[mouse_trans[token]]
    else:
        mouse_vocab = np.unique(np.concatenate(mouse_tokens))
        mouse_token2idx = collections.defaultdict(int)
        for idx,token in enumerate(mouse_vocab):
            if "_" in token:
                mouse_token2idx[token] = idx+1
            else:
                mouse_token2idx[int(token)] = idx+1
        location_vocab = np.unique(np.concatenate(location_tokens))
        location_token2idx = collections.defaultdict(int)
        for idx,token in enumerate(location_vocab):
            if "_" in token:
                location_token2idx[token] = idx+1
            else:
                location_token2idx[int(token)] = idx+1
    pickle.dump([mouse_vocab,mouse_token2idx,location_vocab,location_token2idx],open(os.path.join(data_dir,vocab_name),"wb"))
else:
    print("loading...")
    mouse_vocab,mouse_token2idx,location_vocab,location_token2idx =pickle.load(open(os.path.join(data_dir,vocab_name),"rb"))
vocabs = [location_vocab,mouse_vocab]
vocab2idxs = [location_token2idx,mouse_token2idx]
wv_embeddings=[]
for i,wv_model in enumerate(wv_models):
    if not use_cluster:
        wv_embedding=np.zeros((len(vocab2idxs[i])+1,100))
    else:
        wv_embedding=np.zeros((np.unique(list(token_trans[i].values())).shape[0]+1,100))        
    for v in vocabs[i]:
        if not use_cluster:
            wv_embedding[vocab2idxs[i][v]]=wv_model.wv[str(v)]
        else:
            wv_embedding[vocab2idxs[i][v]]=wv_model.wv[str(token_trans[i][v])]
    wv_embeddings.append(wv_embedding)

