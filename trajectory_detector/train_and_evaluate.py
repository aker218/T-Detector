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


# In[15]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
parser.add_argument('-M',"--use_merge_split", default=False, action='store_true')
parser.add_argument('-T',"--use_time_dis", default=False, action='store_true')
parser.add_argument('-P',"--use_pretrain", default=False, action='store_true')
parser.add_argument('-A',"--use_angle_pretrain", default=False, action='store_true')
parser.add_argument("-DV","--device",type=str,default = '0,1,2,3')
parser.add_argument("-MT","--model_type",type=str,default = 'ConvGRU')
parser.add_argument("-MAT","--use_mutual_attention", default=False, action='store_true')
parser.add_argument("-AP","--use_attention_pooling", default=False, action='store_true')
parser.add_argument("-TL","--use_triloss", default=False, action='store_true')
args = parser.parse_args()
# args.output_data_dir = "./data/new_processed_data/"
# args.use_merge_split = True
# args.use_time_dis = False
# args.use_pretrain = True
# args.use_mutual_attention = True
# args.use_attention_pooling = True
# args.use_triloss = True


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
        os.makedirs(os.path.join(data_dir,"w2v"),exist_ok=True)
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


# In[43]:


from models import *
from trainer import *
from dataset import *
# data_dir = "pkdd_data/"
# file_name = "train_test_idx.pk"
# file_name = "train_test_idx_half.pk"
file_name = "train_test_idx_5.pk"
if not os.path.exists(os.path.join(data_dir,file_name)):
    train_test_idx=[(np.random.random(len(e))>=0.95).astype(np.int) for e in day2action]
    pickle.dump(train_test_idx,open(os.path.join(data_dir,file_name),"wb"))
else:
    print("loading...")
    train_test_idx = pickle.load(open(os.path.join(data_dir,file_name),"rb"))


# In[44]:


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


# In[45]:



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


# In[46]:


train_dataset=sampleDataset(train_day2action)
test_dataset=sampleDataset(test_day2action)


# In[47]:


len(train_dataset),len(test_dataset)


# In[48]:


batch_size = 32
import functools
func = functools.partial(collate_fn,use_token = True,fil =False)
train_dataloader = Data.DataLoader(train_dataset,batch_size=batch_size,collate_fn=func,shuffle=True)
test_dataloader = Data.DataLoader(test_dataset,batch_size=batch_size,collate_fn=func)




mtype = args.model_type
type_dim = {"ConvGRU":400,"MLP":100,"BiGRU":200,"CNN":100}
gpu_ids =args.device
# net = ConvGRU(**mouse_config)
# net = ConvGRU_AutoEncoder(**location_config)
# net = ConvGRU_VAE(**location_vae_config)
net = FusionModel(location_config,mouse_config,type_dim[mtype],type_dim[mtype],use_mutual_attention = args.use_mutual_attention,                  use_residual=args.use_attention_pooling ,use_rnn_output=True,         pretrain=False,use_embed=False,model_type=mtype, tri_loss=args.use_triloss,) #
lr = 0.005
wd = 0.05
optimizer = AdamW(net.parameters(),lr=lr,weight_decay=wd)
if args.use_angle_pretrain or args.use_time_dis:
    num_epoches = 8
else:
    num_epoches = 5
t_total = num_epoches*len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_dataloader)*1, num_training_steps=t_total
)


# In[55]:


if args.use_angle_pretrain:
    angle_pretrain_dir = "./models/"+data_dir.strip("/").split("/")[-1]+"_script_data_model/"
    if args.use_merge_split:
        angle_pretrain_dir +="merge_split_"
    if not args.use_pretrain:
        angle_pretrain_dir +="no_pretrain_"
    if args.use_time_dis:
        angle_pretrain_dir +="idx_geo_dis_"
    angle_pretrain_dir +="embedding_ConvGRU_mutual_attention_residual(rnn_input)_fusion_without_fre_100_trainset(ptretrain)"
    res = os.listdir(angle_pretrain_dir)
    sub_dir = "checkpoint-"+str(max([int(e.split("-")[-1]) for e in res if "checkpoint" in e]))
    angle_pretrain_dir = os.path.join(angle_pretrain_dir,sub_dir)
    print("angle_pretrain_dir: ",angle_pretrain_dir)
    net.load_state_dict(torch.load(os.path.join(angle_pretrain_dir,"model.pt")))

output_dir = "./models/"+data_dir.strip("/").split("/")[-1]+"_script_data_model/"
if args.use_merge_split:
    output_dir +="merge_split_"
if not args.use_pretrain:
    output_dir +="no_pretrain_"
if args.use_time_dis:
    output_dir +="idx_geo_dis_"
output_dir +="embedding_"+mtype
if args.use_mutual_attention:
    output_dir +="_mutual_attention"
if args.use_attention_pooling:
    output_dir +="_attention_pooling"
if args.use_triloss:
    output_dir  +="_triloss"
output_dir +="_5_trainset"
if args.use_angle_pretrain:
    output_dir +="(angle_pretrain)"
os.makedirs(output_dir,exist_ok=True)


# In[56]:

print("output_dir:",output_dir)
train(net,train_dataloader,test_dataloader,num_epoches,gpu_ids,optimizer,scheduler,data_transform=None,      fusion=True,mouse=False,output_dir=output_dir,max_grad_norm=5,eval_iter = num_epoches)

