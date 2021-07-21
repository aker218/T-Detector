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
parser = argparse.ArgumentParser()
parser.add_argument("-D","--output_data_dir",type=str,help="输出文件夹",default = './data/processed_data')
parser.add_argument("-T","--type", help="location/mouse",type=str,default = 'location')
parser.add_argument('-S',"--subsample", default=False, action='store_true')
parser.add_argument('-DV',"--device", default=0, type=int)
parser.add_argument('-L',"--lr", default=0.01, type=float)
parser.add_argument('-E',"--epoch", default=10, type=int)
parser.add_argument('-M',"--use_merge_split", default=False, action='store_true')


args = parser.parse_args()
device = torch.device("cuda:"+str(args.device))

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

ntype = 'cbow'
subsample = args.subsample
data_type = args.type
flag = "" if not subsample else "subsampled_"
print("loading...")
all_centers,all_idx = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+data_type+"_pair.pickle"+suffix),"rb"))
all_contexts = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+data_type+"_context.pickle"+suffix),"rb"))
all_dis = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+data_type+"_dis.pickle"+suffix),"rb"))
all_negatives = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+data_type+"_negative.pickle"+suffix),"rb"))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives,dis,idxes):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
        self.dis = dis
        self.idxes = idxes

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index],self.dis[index],self.idxes[index])

    def __len__(self):
        return len(self.centers)
def batchify(data,ntype = 'skip_gram'):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, 
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    if ntype == 'skip_gram':
        max_len = max(len(c) + len(n) for _, c, n,d,i in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative,dis,idx in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
                torch.tensor(masks), torch.tensor(labels))
    else:
        centers_negatives, contexts, masks, labels = [], [], [], []
        dis_list,idxes = [],[]
        max_len = max(len(c) for _, c, n,d,i in data)
        for center, context, negative,dis,idx in data:
            cur_len = len(context)
            centers_negatives.append([center]+negative)
            idxes.append(idx)
            dis_list.append(np.concatenate([dis,np.zeros(max_len - cur_len)]))
            contexts += [context + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] + [0] * len(negative)]
        return (torch.tensor(centers_negatives), torch.tensor(contexts),
                torch.tensor(masks), torch.tensor(labels),torch.tensor(dis_list),torch.tensor(idxes))
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
def cbow(center_negatives, contexts, masks,embed_v, embed_u,\
         dis = None,idx = None,embed_map = None,embed_weekday = None,embed_time = None):
    v = embed_v(center_negatives)
    u = embed_u(contexts)
    if idx is not None:
        a = embed_map(idx[:,0])
        b = embed_weekday(idx[:,1])
        c = embed_time(idx[:,2])
        idx_embed = (a+b+c).unsqueeze(1)
    v +=idx_embed
    u +=idx_embed
    masks = masks.float()
    if dis is None:
        u = (u * masks.unsqueeze(-1)).sum(dim = 1)/masks.sum(dim = 1,keepdim=True) 
    else:
        weights = torch.exp((torch.log(torch.tensor(0.8).to(dis.device))*dis)).to(dis.device)
        weights[masks==0] = -1e6
        weights = torch.softmax(weights,dim=-1)
        u = (u.float() * weights.unsqueeze(-1).float()).sum(dim=1)
    pred = torch.bmm(v, u.unsqueeze(dim = 1).permute(0, 2, 1))
    return pred
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=mask)
        return res.mean(dim=1)
def train(net, lr, num_epochs,ntype = 'skip_gram',idx_embedding = None,output_dir = "./models/time_dis_w2v/",device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    print("train on", device)
    net = net.to(device)
    if idx_embedding is not None:
        idx_embedding = idx_embedding.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in tqdm(data_iter,total =len(data_iter)):

            if ntype == 'skip_gram':
                center_negative, context, mask, label,dis_list,idxes = [d.to(device) for d in batch]
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.view(label.shape), label, mask) *
                     mask.shape[1] / mask.float().sum(dim=1)).mean() # 一个batch的平均loss
            else:
                center_negative, context, mask, label,dis_list,idxes = [d.to(device) for d in batch]
                pred = cbow(center_negative, context, mask, net[0], net[1],\
                            dis_list,idxes,idx_embedding[0],idx_embedding[1],idx_embedding[2])
                tmp_mask = torch.ones(label.shape).to(device)
                l = (loss(pred.view(label.shape), label, tmp_mask) *
                     tmp_mask.shape[1] / tmp_mask.float().sum(dim=1)).mean() # 一个batch的平均loss
            # 使用掩码变量mask来避免填充项对损失函数计算的影响

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok = True)
        torch.save(net.state_dict(),os.path.join(output_dir,"w2v_"+str(epoch)+".pt"))
        torch.save(idx_embedding.state_dict(),os.path.join(output_dir,"idx_embedding_"+str(epoch)+".pt"))
        
idx_to_token,token_to_idx,counter = pickle.load(open(os.path.join(data_dir,"time-dis-vec/"+flag+data_type+"_vocab.pickle"+suffix),"rb"))
        
dataset = MyDataset(all_centers,all_contexts,all_negatives,all_dis,all_idx)
import functools

func = functools.partial(batchify,ntype = ntype)
num_workers = 0 if sys.platform.startswith('win32') else 0
batch_size = 512
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=func, 
                            num_workers=num_workers)
for batch in data_iter:
    if ntype == 'skip_gram':
        for name, data in zip(['centers', 'contexts_negatives', 'masks',
                               'labels'], batch):
            print(name, 'shape:', data.shape)
        break
    else:
        for name, data in zip(['centers_negatives', 'contexts', 'masks',
                               'labels','dis','idx'], batch):
            print(name, 'shape:', data.shape)
        break
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)
idx_embedding = nn.Sequential(
    nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
    nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
    nn.Embedding(num_embeddings=500, embedding_dim=embed_size),
)

loss = SigmoidBinaryCrossEntropyLoss()
print("saved in:","./models/"+data_dir.strip("/").split("/")[-1]+"_time_dis_w2v_"+flag+data_type+suffix+"/")
train(net, args.lr, args.epoch, ntype = ntype,idx_embedding = idx_embedding,output_dir = "./models/"+data_dir.strip("/").split("/")[-1]+"_time_dis_w2v_"+flag+data_type+suffix+"/",device = device)
