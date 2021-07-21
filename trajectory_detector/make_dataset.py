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
parser.add_argument('-T',"--use_time_dis", default=False, action='store_true')


# In[3]:


args = parser.parse_args()

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
# In[4]:


if not args.use_time_dis:
    token_trans = []
    idx_embeddings = [0,0]
    window = 20
    mincount = 1
    use_cluster = False
    if use_merge_split:
        vocab_name = "w2v/vocab%s.pk"%(suffix)
        if not os.path.exists(os.path.join(data_dir,vocab_name)):
            location_tokens = [[location_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_location_datas)]
            mouse_tokens = [[mouse_raw_xy2token[tuple(list(e))] for e in t[:,:2]] for t in tqdm(filtered_mouse_datas)]

    model_dir = data_dir.strip("/").split("/")[-1]+"_w2v/"
    vocab_name = "w2v/vocab%s.pk"%(suffix)
    if not os.path.exists(os.path.join(data_dir,vocab_name)):
        os.makedirs(os.path.join(data_dir,"w2v"),exists_ok=True)
        token_lists = [location_tokens,mouse_tokens]
    else:
        token_lists = [0,0]

    keys = ['location','mouse']
    wv_models = []

    for i,(key,tokens) in enumerate(zip(keys,token_lists)):
        model_name = '%s_window%d_dim%d_mincount%d%s.pickle' % (key, window,100, mincount, suffix)
        model=pickle.load(open('./models/'+model_dir+model_name,'rb'))
        wv_models.append(model)


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
else:
    subsample = True
    flag = "" if not subsample else "subsampled_"
    keys = ['location','mouse']
    vocabs = []
    vocab2idxs = []
    wv_embeddings = []
    idx_embeddings = []
    embed_size = 100
    epoches = [10,10]

    model_dir =data_dir.strip("/").split("/")[-1]

    for key,epoch in zip(keys,epoches):
        idx_to_token,token_to_idx,counter = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+key+"_vocab.pickle"+suffix),"rb"))

        net = nn.Sequential(
            nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
            nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
        )
        idx_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
            nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
            nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
        )
        net.load_state_dict(torch.load("./models/"+model_dir+"_time_dis_w2v_"+flag+key+suffix+"/w2v_"+str(epoch-1)+".pt"))
        idx_embedding.load_state_dict(torch.load("./models/"+model_dir+"_time_dis_w2v_"+flag+key+suffix+"/idx_embedding_"+str(epoch-1)+".pt"))
        embed = (net[0].weight.detach().numpy()+net[1].weight.detach().numpy())/2
        map_embedding = idx_embedding[0].weight.detach()
        weekday_embedding = idx_embedding[1].weight.detach()
        time_embedding = idx_embedding[2].weight.detach()
        wv_embeddings.append(embed)
        vocabs.append(idx_to_token)
        vocab2idxs.append(token_to_idx)
        idx_embeddings.append([map_embedding,weekday_embedding,time_embedding])

# In[5]:


def test_fft(data,sampling_rate=100,use_phase=False,denoise=True):
    if denoise:
        temp=data-data.mean()
    else:
        temp=data
    sampling_rate = sampling_rate   #采样率
    fft_size =data.shape[0]      #FFT长度
    xs = temp
    
    xf = np.fft.rfft(xs) / fft_size  #返回fft_size/2+1 个频率
    freqs = np.linspace(0, int(sampling_rate/2), int(fft_size/2+1))   #表示频率
    xfp = np.abs(xf) *2   #代表信号的幅值，即振幅
    angle=np.angle(xf)
    if np.sum(xfp)-0<=1e-3:
        avg_freq,variance=0,0
    else:
        avg_freq=np.sum(xfp*freqs)/np.sum(xfp)
        variance=np.sum(((freqs-avg_freq)**2)*xfp)/np.sum(xfp)
    if use_phase:
        return xfp,angle,avg_freq,variance #返回平均频率  
    else:
        return xfp,avg_freq,variance
def get_frequency_map(a,flag=False,sampling_rate=100,denoise=True,use_phase=False,dim=3):
    a=a.reshape(-1,dim)
    if flag:
        a/=np.max(abs(a),axis=0)
    a=a.transpose()
    xfps=[]
    angles=[]
    for i in range(dim):
        if use_phase:
            xfp,angle,avg_freq,variance=test_fft(a[i],sampling_rate=sampling_rate,use_phase=use_phase,denoise=denoise)
            xfps.append(xfp)
            angles.append(angle)
        else:
            xfp,avg_freq,variance=test_fft(a[i],sampling_rate=sampling_rate,use_phase=use_phase,denoise=denoise)
            xfps.append(xfp)
    if use_phase:
        return np.concatenate([np.array(xfps).transpose(),np.array(angles).transpose()],axis=-1)
    else:
        return np.array(xfps).transpose()
def get_feature(sample=None):
    if sample is None:
        return np.zeros((1,9)),np.zeros((int(1000//40/2)*2,9))
    res = sample[:,:2]
    time_dis = sample[:,2:]
    dis = (res[1:]-res[:-1])
    degree = ((np.arctan2(dis[:,1],dis[:,0])*180)/np.pi)[:,np.newaxis]
    abs_dis = np.sqrt(dis[:,0:1]**2+dis[:,1:2]**2)
    begin_idx=np.argmin((abs_dis[:,0]==0)&(sample[1:,2]<350))
    v_xy = dis/(time_dis[1:]+1e-6)
    v_abs_full = np.sqrt(v_xy[:,0:1]**2+v_xy[:,1:2]**2)
    second_v_xy = np.concatenate([np.zeros((1,2)),res[2:]-res[:-2],res[-1:]-res[-2:-1]],axis=0)/(time_dis+1e-6)
    a_xy = (second_v_xy[1:]-second_v_xy[:-1])/(time_dis[1:]+1e-6)
    a_abs_full = np.sqrt(a_xy[:,0:1]**2+a_xy[:,1:2]**2)
    feature=np.concatenate([degree,abs_dis,v_xy,v_abs_full,a_xy,a_abs_full,time_dis[1:]],axis=-1)
    fre_feature=get_frequency_map(feature[begin_idx:],sampling_rate=1000//40,dim=feature.shape[-1])
    fre_feature=scipy.signal.resample(fre_feature,num=int(1000//40/2)*2,axis=0)
    return feature,fre_feature
def resample(sample
             ,down_num = 4):
    if sample.shape[0]<=8:
        return sample
    new_sample,new_t = scipy.signal.resample(sample[:,:2],sample.shape[0]//down_num,t=np.cumsum(sample[:,2]),axis=0)
    new_t[1:]=new_t[1:]-new_t[:-1]
    new_sample=np.concatenate([new_sample,new_t[:,np.newaxis]],axis=-1)
    return new_sample
import functools
def quantile(df,q):
    return df.quantile(q=q)
func_50 = functools.partial(quantile,q=0.5)
func_75 = functools.partial(quantile,q=0.75)
func_25 = functools.partial(quantile,q=0.25)
cols = ['degree','abs_dis','vx','vy','v_full','ax','ay','a_full','time_dis']
numeric_cols_func=dict([(col,['mean','max','min','std','skew',func_50,func_75,func_25]) for col in cols])
def agg_feature(data):
#     df = pd.DataFrame(data,columns = ['degree','abs_dis','vx','vy','v_full','ax','ay','a_full','time_dis'])
#     fea = df.agg(numeric_cols_func).values.reshape(-1)
#     filter_df = df[df['abs_dis']!=0]
#     if filter_df.shape[0]==0:
#         temp = np.zeros(72)
#     else:
#         temp = filter_df.agg(numeric_cols_func).values.reshape(-1)
#     feature = np.concatenate([fea,temp])
    def make_feature(data):
        mean_data = np.mean(data,axis=0)
        max_data = np.max(data,axis=0)
        min_data = np.min(data,axis=0)
        std_data = np.std(data,axis=0)
        skew_data = ((((data - mean_data)/(std_data+1e-6)))**3).mean(axis = 0)
        data_25 = np.quantile(data,q=0.25,axis=0)
        data_50 = np.quantile(data,q=0.5,axis=0)
        data_75 = np.quantile(data,q=0.75,axis=0)
        feature = np.concatenate([mean_data,max_data,min_data,std_data,skew_data,data_50,data_75,data_25],axis=0)
        return feature
    feature_a = make_feature(data)
    filter_data = data[data[:,0]!=0]
    if(filter_data.shape[0]==0):
        feature_b = np.zeros(72)
    else:
        feature_b = make_feature(filter_data)
    feature = np.concatenate([feature_a,feature_b],axis=0)
    return feature
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


# In[6]:





# In[7]:


location_token2idx = vocab2idxs[0]
mouse_token2idx = vocab2idxs[1]
use_mouse = True
# use_mouse = False

if not args.use_time_dis:
    file_name = "action_sequence_with_feature.pickle"
else:
    file_name = "action_sequence_with_feature(time-dis).pickle"
if not os.path.exists(os.path.join(data_dir,file_name+suffix)):
    new_day2action= []
    for j,day in tqdm(enumerate(day2action)):
        new_day = [[e[0],{'label':e[1]['label'],'map_idx':e[1]['map_idx']}]  for e in day]
        for i in tqdm(range(len(day))):
            if 'location_data' in day[i][1]:
                tmp1 = get_idx(day[i],itype = 'location')
            if 'mouse_data' in day[i][1]:
                tmp2 = get_idx(day[i],itype = 'mouse')
                if not 'location_data' in day[i][1]:
                    tmp1 = tmp2
            if 'location_data' in day[i][1] or  'mouse_data' in day[i][1]:
                new_day[i][1]['weekday_idx'] = tmp1[1]
                new_day[i][1]['time_idx'] = tmp1[2]
            else:
                new_day[i][1]['weekday_idx'] = 100
                new_day[i][1]['time_idx'] = 100                
            sample=day[i][1]['location_data'] if 'location_data' in day[i][1] else None
            location_feature,location_fre_feature=np.zeros((1,9)),np.zeros((int(1000//40/2)*2,9))
#             get_feature(sample)
            if sample is not None:
                if 'location_raw_token' in day[i][1]:
                    token_sequence = raw_token_sequence = day[i][1]['location_raw_token']
                    filtered_sample = sample
                else:
                    filtered_sample,token_sequence,raw_token_sequence = process(sample,*word2vec_config['location'])
                if 'location_xy' in day[i][1]:
                    new_day[i][1]['location_xy'] = day[i][1]['location_xy']
                    new_day[i][1]['location_f_xy'] = day[i][1]['location_xy']
                else:

                    new_day[i][1]['location_xy'] = get_x_y(raw_token_sequence,*word2vec_config['location'])
                    new_day[i][1]['location_f_xy'] = get_x_y(token_sequence,*word2vec_config['location'])
                if not use_merge_split:
                    raw_token_sequence = np.asarray([location_token2idx[t] if t in location_token2idx else location_token2idx['pad']                                                      for t in raw_token_sequence])
                    token_sequence = np.asarray([location_token2idx[t] if t in location_token2idx else location_token2idx['pad']                                                  for t in token_sequence])
                else:
                    raw_token_sequence =  [location_raw_xy2token[tuple(list(e))] for e in sample[:,:2]]
                    raw_token_sequence =  np.asarray([location_token2idx[t] if t in location_token2idx else location_token2idx['pad']                                                      for t in raw_token_sequence])
                    token_sequence =  [location_raw_xy2token[tuple(list(e))] for e in filtered_sample[:,:2]]
                    token_sequence = np.asarray([location_token2idx[t] if t in location_token2idx else location_token2idx['pad']                                                  for t in token_sequence])
                filtered_time_dis = filtered_sample[:,-1]
                time_dis = sample[:,-1]
            else:
                filtered_time_dis ,token_sequence,raw_token_sequence =                 np.asarray([0]),np.asarray([location_token2idx['pad']]),np.asarray([location_token2idx['pad']])
                time_dis = np.asarray([0])
                new_day[i][1]['location_xy'] = np.zeros([1,2]);
                new_day[i][1]['location_f_xy'] = np.zeros([1,2]);
            new_day[i][1]['location_f_token'] = token_sequence
            new_day[i][1]['location_token'] = raw_token_sequence
            new_day[i][1]['location_f_dis'] = filtered_time_dis
            new_day[i][1]['location_dis'] = time_dis

            
            sample=day[i][1]['mouse_data'] if 'mouse_data' in day[i][1] else None
            if use_mouse:
                if sample is not None:
                    filtered_sample,token_sequence,raw_token_sequence = process(sample,*word2vec_config['mouse'])
                    new_day[i][1]['mouse_xy'] = get_x_y(raw_token_sequence,*word2vec_config['mouse'])
                    new_day[i][1]['mouse_f_xy'] = get_x_y(token_sequence,*word2vec_config['mouse'])

                    if not use_merge_split:
                        raw_token_sequence = np.asarray([mouse_token2idx[t] if t in mouse_token2idx else mouse_token2idx['pad']                                                          for t in raw_token_sequence])
                        token_sequence = np.asarray([mouse_token2idx[t] if t in mouse_token2idx else mouse_token2idx['pad']                                                      for t in token_sequence])
                    else:
                        raw_token_sequence =  [mouse_raw_xy2token[tuple(list(e))] for e in sample[:,:2]]
                        raw_token_sequence = np.asarray([mouse_token2idx[t] if t in mouse_token2idx else mouse_token2idx['pad']                                                          for t in raw_token_sequence])
                        token_sequence =  [mouse_raw_xy2token[tuple(list(e))] for e in filtered_sample[:,:2]]
                        token_sequence = np.asarray([mouse_token2idx[t] if t in mouse_token2idx else mouse_token2idx['pad']                                                      for t in token_sequence])
                    filtered_time_dis = filtered_sample[:,-1]
                    time_dis = sample[:,-1]
                else:
                    filtered_time_dis ,token_sequence,raw_token_sequence =                      np.asarray([0]), np.asarray([mouse_token2idx['pad']]), np.asarray([mouse_token2idx['pad']])
                    time_dis = np.asarray([0])
                    new_day[i][1]['mouse_xy'] = np.zeros([1,2]);
                    new_day[i][1]['mouse_f_xy'] = np.zeros([1,2]);
                mouse_feature,mouse_fre_feature=np.zeros((1,9)),np.zeros((int(1000//40/2)*2,9))
#                 get_feature(sample)
            else:
                filtered_time_dis ,token_sequence,raw_token_sequence =                  np.asarray([0]), np.asarray([mouse_token2idx['pad']]), np.asarray([mouse_token2idx['pad']])
                time_dis = np.asarray([0])
                new_day[i][1]['mouse_xy'] = np.zeros([1,2]);
                new_day[i][1]['mouse_f_xy'] = np.zeros([1,2]);     
                mouse_feature,mouse_fre_feature=location_feature,location_fre_feature
            new_day[i][1]['mouse_f_token'] = token_sequence
            new_day[i][1]['mouse_token'] = raw_token_sequence
            new_day[i][1]['mouse_f_dis'] = filtered_time_dis
            new_day[i][1]['mouse_dis'] = time_dis
            
            new_day[i][1]['location_feature']=location_feature
            new_day[i][1]['location_fre_feature']=location_fre_feature
            new_day[i][1]['mouse_feature']=mouse_feature
            new_day[i][1]['mouse_fre_feature']=mouse_fre_feature

        new_day2action.append(new_day)
    pickle.dump(new_day2action,open(os.path.join(data_dir,file_name+suffix),"wb"))
    day2action = new_day2action
else:
    print("loading feature...")
    tmp = pickle.load(open(os.path.join(data_dir,file_name+suffix),"rb"))
    if len(tmp)==3:
        idx2user,user2idx,day2action = tmp
    else:
        day2action = tmp


# In[8]:


if not args.use_time_dis:
    file_name = "masked_action_sequence_with_feature.pickle"
else:
    file_name = "masked_action_sequence_with_feature(time-dis).pickle"

if not os.path.exists(os.path.join(data_dir,file_name+suffix)):
    mask_idxs = pickle.load(open(os.path.join(data_dir,"mask_idx.pickle"),"rb"))
    for day,mask_idx in tqdm(zip(day2action,mask_idxs)):
        for i in tqdm(range(len(day))):
            label = day[i][1]['label']
#             day[i][1]['location_agg_feature'] = agg_feature(day[i][1]['location_feature'])
#             day[i][1]['mouse_agg_feature'] = agg_feature(day[i][1]['mouse_feature'])

            if mask_idx[i] in [0,1]: 
                if mask_idx[i] == 0:
                    masked_mouse_feature,masked_mouse_fre_feature = np.zeros((1,9)),np.zeros((int(1000//40/2)*2,9))
                    masked_mouse_filtered_time_dis ,masked_mouse_token_sequence,masked_mouse_raw_token_sequence =                     np.asarray([0]),np.asarray([mouse_token2idx['pad']]),np.asarray([mouse_token2idx['pad']])
#                     masked_mouse_agg_feature = np.zeros(144)
                    masked_mouse_time_dis = np.asarray([0])
                    masked_mouse_xy = np.zeros([1,2])
                    masked_mouse_f_xy = np.zeros([1,2])
                else:
                    masked_mouse_feature,masked_mouse_fre_feature =                     day[i][1]['mouse_feature'],day[i][1]['mouse_fre_feature']
                    masked_mouse_filtered_time_dis ,masked_mouse_token_sequence,masked_mouse_raw_token_sequence =                     day[i][1]['mouse_f_dis'],day[i][1]['mouse_f_token'],day[i][1]['mouse_token']
#                     masked_mouse_agg_feature = day[i][1]['mouse_agg_feature']
                    masked_mouse_time_dis =  day[i][1]['mouse_dis']
                    masked_mouse_xy = day[i][1]['mouse_xy']
                    masked_mouse_f_xy = day[i][1]['mouse_f_xy']
                masked_location_feature,masked_location_fre_feature = day[i][1]['location_feature'],day[i][1]['location_fre_feature']
                day[i][1]['masked_mouse_feature'] = masked_mouse_feature
                day[i][1]['masked_mouse_fre_feature'] = masked_mouse_fre_feature
                day[i][1]['masked_location_feature'] = masked_location_feature
                day[i][1]['masked_location_fre_feature'] = masked_location_fre_feature
                
                day[i][1]['masked_mouse_f_token'] = masked_mouse_token_sequence
                day[i][1]['masked_mouse_token'] = masked_mouse_raw_token_sequence
                day[i][1]['masked_mouse_f_dis'] = masked_mouse_filtered_time_dis
                day[i][1]['masked_mouse_dis'] = masked_mouse_time_dis
                day[i][1]['masked_mouse_xy'] = masked_mouse_xy
                day[i][1]['masked_mouse_f_xy'] = masked_mouse_f_xy
                
                day[i][1]['masked_location_f_token'] = day[i][1]['location_f_token']
                day[i][1]['masked_location_token'] =  day[i][1]['location_token']
                day[i][1]['masked_location_f_dis'] = day[i][1]['location_f_dis']
                day[i][1]['masked_location_dis'] = day[i][1]['location_dis']
                day[i][1]['masked_location_xy'] = day[i][1]['location_xy']
                day[i][1]['masked_location_f_xy'] = day[i][1]['location_f_xy']
                
#                 day[i][1]['masked_mouse_agg_feature'] = masked_mouse_agg_feature
#                 day[i][1]['masked_location_agg_feature'] = day[i][1]['location_agg_feature']
            elif mask_idx[i] in [2,3]: 
                if mask_idx[i] == 2:
                    masked_location_feature,masked_location_fre_feature = np.zeros((1,9)),np.zeros((int(1000//40/2)*2,9))
                    masked_location_filtered_time_dis ,masked_location_token_sequence,masked_location_raw_token_sequence =                         np.asarray([0]),np.asarray([location_token2idx['pad']]),np.asarray([location_token2idx['pad']])
#                     masked_location_agg_feature = np.zeros(144)
                    masked_location_time_dis = np.asarray([0])
                    masked_location_xy = np.zeros([1,2])
                    masked_location_f_xy = np.zeros([1,2])
                    
                else:
                    masked_location_feature,masked_location_fre_feature = day[i][1]['location_feature'],day[i][1]['location_fre_feature']
                    masked_location_filtered_time_dis ,masked_location_token_sequence,masked_location_raw_token_sequence =                     day[i][1]['location_f_dis'],day[i][1]['location_f_token'],day[i][1]['location_token']
#                     masked_location_agg_feature = day[i][1]['location_agg_feature']
                    masked_location_time_dis = day[i][1]['location_dis']
                    masked_location_xy = day[i][1]['location_xy']
                    masked_location_f_xy = day[i][1]['location_f_xy']
                    
                masked_mouse_feature,masked_mouse_fre_feature = day[i][1]['mouse_feature'],day[i][1]['mouse_fre_feature']
                day[i][1]['masked_mouse_feature'] = masked_mouse_feature
                day[i][1]['masked_mouse_fre_feature'] = masked_mouse_fre_feature
                day[i][1]['masked_location_feature'] = masked_location_feature
                day[i][1]['masked_location_fre_feature'] = masked_location_fre_feature
                
                day[i][1]['masked_location_f_token'] = masked_location_token_sequence
                day[i][1]['masked_location_token'] = masked_location_raw_token_sequence
                day[i][1]['masked_location_f_dis'] = masked_location_filtered_time_dis
                day[i][1]['masked_location_dis'] = masked_location_time_dis
                day[i][1]['masked_location_xy'] = masked_location_xy
                day[i][1]['masked_location_f_xy'] = masked_location_f_xy
                
                day[i][1]['masked_mouse_f_token'] = day[i][1]['mouse_f_token']
                day[i][1]['masked_mouse_token'] =  day[i][1]['mouse_token']
                day[i][1]['masked_mouse_f_dis'] = day[i][1]['mouse_f_dis']
                day[i][1]['masked_mouse_dis'] = day[i][1]['mouse_dis']
                day[i][1]['masked_mouse_xy'] = day[i][1]['mouse_xy']
                day[i][1]['masked_mouse_f_xy'] = day[i][1]['mouse_f_xy']

#                 day[i][1]['masked_mouse_agg_feature'] = day[i][1]['mouse_agg_feature']
#                 day[i][1]['masked_location_agg_feature'] = masked_location_agg_feature
    print("dumping...")
    pickle.dump(day2action,open(os.path.join(data_dir,file_name+suffix),"wb"))
else:
    print("loading feature...")
    tmp = pickle.load(open(os.path.join(data_dir,file_name+suffix),"rb"))
    if len(tmp)==3:
        idx2user,user2idx,day2action = tmp
    else:
        day2action = tmp

