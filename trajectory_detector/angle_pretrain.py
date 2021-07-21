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


# In[2]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
parser.add_argument('-M',"--use_merge_split", default=False, action='store_true')
parser.add_argument('-T',"--use_time_dis", default=False, action='store_true')
parser.add_argument('-P',"--use_pretrain", default=False, action='store_true')
parser.add_argument("-DV","--device",type=str,default = '0,1,2,3')
args = parser.parse_args()
# args.output_data_dir = "./data/new_processed_data/"
# args.use_merge_split = True
# args.use_time_dis = False
# args.use_pretrain = True


# In[3]:


data_dir = args.output_data_dir

use_merge_split = args.use_merge_split
use_cluster = False
with_weight = False
suffix = ""
if use_merge_split:
    suffix = "_merge_split"
if use_cluster:
    if with_weight:
        suffix = "_cluster_w"
    else:
        suffix = "_cluster"
if use_merge_split:
    word2vec_config = {'location':[0,50000,0,50000,20,200,200],'mouse':[-5000,50000,-1000,35000,10,200,200]} #merge_and_split 专用
else:
    word2vec_config = {'location':[0,50000,0,50000,20,100,100],'mouse':[-5000,50000,-1000,35000,10,100,100]}


# In[4]:


if args.use_time_dis:
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
else:
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


# In[5]:


if args.use_time_dis:
    file_name = "masked_action_sequence_with_feature(time-dis).pickle"
else:
    file_name = "masked_action_sequence_with_feature.pickle"   
print("loading feature...")
tmp = pickle.load(open(os.path.join(data_dir,file_name+suffix),"rb"))
if len(tmp)==3:
    idx2user,user2idx,day2action = tmp
else:
    day2action = tmp


# In[6]:


from models import *
from trainer import *
from dataset import *
# data_dir = "pkdd_data/"
# file_name = "train_test_idx.pk"
# file_name = "train_test_idx_half.pk"
file_name = "train_test_idx_80.pk"
if not os.path.exists(os.path.join(data_dir,file_name)):
    train_test_idx=[(np.random.random(len(e))>=0.2).astype(np.int) for e in day2action]
    pickle.dump(train_test_idx,open(os.path.join(data_dir,file_name),"wb"))
else:
    print("loading...")
    train_test_idx = pickle.load(open(os.path.join(data_dir,file_name),"rb"))


# In[7]:


dataset=sampleDataset(day2action)


# In[9]:


if args.use_time_dis:
    use_idx_embedding = True
    use_geo_position = True
else:
    use_idx_embedding = False
    use_geo_position = False

location_config = {'input_size':100,'input_size_fre':9,'hidden_size':100,'dropout_rate':0.2,'bidirectional':True,\
                   'use_rnn':True,'use_self_attention':False,'num_hidden_layers':1,'num_attention_heads':4,'use_fre':False,'use_cnn':True,\
                   'use_embedding':True,'embedding_weight':None,'num_embeddings':wv_embeddings[0].shape[0],'num_cnn_layers':3,\
                   'use_time_position':False,'pre_cnn_time_position':False,'use_idx_embedding':use_idx_embedding,'idx_embedding_weight':idx_embeddings[0],\
                   'use_geo_position':use_geo_position,'mse_loss':False,'logloss':True,'sinusoidal':False}
mouse_config = {'input_size':100,'input_size_fre':9,'hidden_size':100,'dropout_rate':0.2,'bidirectional':True,\
                   'use_rnn':True,'use_self_attention':False,'num_hidden_layers':1,'num_attention_heads':4,'use_fre':False,'use_cnn':True,\
                   'use_embedding':True,'embedding_weight':None,'num_embeddings':wv_embeddings[1].shape[0],'num_cnn_layers':3,\
                   'use_time_position':False,'pre_cnn_time_position':False,'use_idx_embedding':use_idx_embedding,'idx_embedding_weight':idx_embeddings[1],\
                   'use_geo_position':use_geo_position,'mse_loss':False,'logloss':True,'sinusoidal':False}

if args.use_pretrain:
    location_config['embedding_weight'] = torch.tensor(wv_embeddings[0]).float()
    mouse_config['embedding_weight'] = torch.tensor(wv_embeddings[1]).float()
#mutual attentiond的时候只用一层rnn


# In[10]:



train_day2action = []
test_day2action = []
for day,idx_list in tqdm(zip(day2action,train_test_idx)):
    train_day2action.append([])
    test_day2action.append([])
    for sample,idx in zip(day,idx_list):
        if idx==1:
            #(sample[1]['label'] == 1 and random.random()<=0.1) or
#             if  sample[1]['label'] == 0:
            train_day2action[-1].append(sample)
#         elif random.random()<=0.1:
        else:
            test_day2action[-1].append(sample)


# In[11]:


batch_size = 32
import functools
func = functools.partial(collate_fn,use_token = True,fil =False)
train_dataset=sampleDataset(train_day2action)


# In[12]:


train_dataloader = Data.DataLoader(train_dataset,batch_size=batch_size,collate_fn=func,shuffle=True)
dataloader = Data.DataLoader(dataset,batch_size=batch_size,collate_fn=func)


# In[13]:




gpu_ids =args.device
# net = ConvGRU(**mouse_config)
# net = ConvGRU_AutoEncoder(**location_config)
# net = ConvGRU_VAE(**location_vae_config)
net = FusionModel(location_config,mouse_config,400,400,use_mutual_attention = True, use_residual=True,use_rnn_output=True,         pretrain=True,use_embed=False,model_type='ConvGRU', tri_loss=False,) #
lr = 0.005
wd = 0.05
optimizer = AdamW(net.parameters(),lr=lr,weight_decay=wd)
num_epoches = 10
t_total = num_epoches*len(dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(dataloader)*1, num_training_steps=t_total
)


# In[21]:


data_dir = "./models/"+data_dir.strip("/").split("/")[-1]+"_script_data_model/"
if args.use_merge_split:
    data_dir +="merge_split_"
if not args.use_pretrain:
    data_dir +="no_pretrain_"
if args.use_time_dis:
    data_dir +="idx_geo_dis_"
data_dir +="embedding_ConvGRU_mutual_attention_residual(rnn_input)_fusion_without_fre_100_trainset(ptretrain)"
os.makedirs(data_dir,exist_ok=True)


# In[22]:

print(net)
train(net,dataloader,train_dataloader,num_epoches,gpu_ids,optimizer,scheduler,data_transform=None,      fusion=True,mouse=False,output_dir=data_dir,max_grad_norm=5,eval_iter = 10)

