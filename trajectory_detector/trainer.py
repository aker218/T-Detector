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

def rand_bbox(size, lam):
    L = size[1]
    cut_rat = 1. - lam
    cut_l = np.int(L * cut_rat)
    # uniform
    cx = np.random.randint(L)
    bbx1 = np.clip(cx - cut_l // 2, 0, L)
    bbx2 = np.clip(cx + cut_l // 2, 0, L)
    return bbx1, bbx2
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device
def save_model(output_dir,model, global_step):
    output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

from torch.optim import lr_scheduler
from torch import autograd


def evaluate_accuracy(train_iter, net, device=None,fusion = False,mouse=False,filter_lack = False):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    pred_score_list = []
    if fusion:
        loc_pred_score_list = []
        mouse_pred_score_list = []
    true_label_list = []
    train_acc_sum = 0.0
    train_auc_sum,train_p_acc_sum,train_n_acc_sum = 0,0,0
    for idx,sample in  tqdm(enumerate(train_iter),total = len(train_iter)):
        if 'mode' in net.__dict__:
            net.mode = 'test'
        net.eval()
#         sample = {key:sample[key].to(device) for key in sample}
        with torch.no_grad():
            y_hat = sample['labels'].to(device)
            if fusion:

                X_loc,X_len_loc,X_fre_loc,X_dis_loc = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                 sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                time_position_ids_loc = (torch.cumsum(X_dis_loc,dim=1)//400).long()
                idx_list_loc = None
                if 'map_idx' in sample:
                    idx_list_loc = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_loc = None
                if 'location_xy' in sample:
                    geo_position_ids_loc = sample['location_xy'].to(device)

                X_mo,X_len_mo,X_fre_mo,X_dis_mo = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                 sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                time_position_ids_mo = (torch.cumsum(X_dis_mo,dim=1)//400).long()
                idx_list_mo  = None
                if 'map_idx' in sample:
                    idx_list_mo  = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_mo  = None
                if 'mouse_xy' in sample:
                    geo_position_ids_mo  = sample['mouse_xy'].to(device)
                logits,location_logits,mouse_logits,loss,hidden = net(X_loc,X_fre_loc,X_len_loc,X_dis_loc,\
                X_mo,X_fre_mo,X_len_mo,X_dis_mo,y_hat,time_position_ids_loc,idx_list_loc,geo_position_ids_loc\
                ,time_position_ids_mo,idx_list_mo,geo_position_ids_mo)  
                length = sample['mouse_length'].detach().cpu().numpy()*sample['location_length'].detach().cpu().numpy()
            else:
                if not mouse:
                    X,X_len,X_fre,X_dis = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                     sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'location_xy' in sample:
                        geo_position_ids = sample['location_xy'].to(device)
                    logits,loss,hidden,_,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)
                    length = sample['location_length'].detach().cpu().numpy()
                else:
                    X,X_len,X_fre,X_dis = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                     sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'mouse_xy' in sample:
                        geo_position_ids = sample['mouse_xy'].to(device)
                    logits,loss,hidden,_,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)    
                    length = sample['mouse_length'].detach().cpu().numpy()

        pred_score = torch.softmax(logits,dim=-1)[:,1].detach().cpu().numpy()
        if fusion:
            loc_pred_score = torch.softmax(location_logits,dim=-1)[:,1].detach().cpu().numpy()
            mouse_pred_score = torch.softmax(mouse_logits,dim=-1)[:,1].detach().cpu().numpy()
        true_label = sample['labels'].detach().cpu().numpy()
        if filter_lack:
            pred_score = pred_score[length!=1]
            true_label = true_label[length!=1]
            if fusion:
                loc_pred_score = loc_pred_score[length!=1]
                mouse_pred_score = mouse_pred_score[length!=1]
        pred_score_list.append(pred_score)
        true_label_list.append(true_label)
        if fusion:
            loc_pred_score_list.append(loc_pred_score)            
            mouse_pred_score_list.append(mouse_pred_score)  
    pred_score = np.concatenate(pred_score_list,axis=0)
    true_label = np.concatenate(true_label_list,axis=0)
    if fusion:
        loc_pred_score = np.concatenate(loc_pred_score_list,axis=0)
        mouse_pred_score = np.concatenate(mouse_pred_score_list,axis=0)
    def get_metrics(pred_score,true_label):
        fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_score)
        train_AUC = metrics.auc(fpr, tpr)
        train_acc = metrics.accuracy_score(true_label,pred_score>0.5)
        temp = precision_recall_fscore_support(true_label,pred_score>0.5)
        train_precision = temp[0][1]
        train_recall = temp[1][1]
        train_f1 = temp[2][1]
        return train_acc,train_AUC,train_f1,train_recall,train_precision
    acc,AUC,f1,recall,precision = get_metrics(pred_score,true_label)
    if fusion:
        loc_acc,loc_AUC,loc_f1,loc_recall,loc_precision = get_metrics(loc_pred_score,true_label)
        mouse_acc,mouse_AUC,mouse_f1,mouse_recall,mouse_precision = get_metrics(mouse_pred_score,true_label)
        return acc,AUC,f1,recall,precision,loc_acc,loc_AUC,loc_f1,loc_recall,loc_precision,mouse_acc,mouse_AUC,mouse_f1,mouse_recall,mouse_precision
    return acc,AUC,f1,recall,precision
def train(net,train_iter,val_iter,num_epochs,gpu_ids,optimizer= None ,scheduler= None ,verbose_num= 1,
         mouse=False,fusion = False,max_grad_norm = 1,data_transform = None,output_dir = "./models/transformer_location/",eval_iter = 1):

    net, device = load_model_and_parallel(net, gpu_ids)

    if mouse:
        output_dir =output_dir.replace("location","mouse")
    use_n_gpus = False
    if hasattr(net, "module"):
        use_n_gpus = True
        
    # Train
    t_total = len(train_iter) * num_epochs
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {t_total}")
    net.zero_grad()

    global_step = 0

    save_steps = t_total // num_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 500

    avg_loss = 0.
    
    metric_str = ""
    eval_save_path = os.path.join(output_dir, 'eval_metric.txt')
    max_auc = 0
    max_epoch = -1
    loss_func = nn.CrossEntropyLoss()
    for num_epoch in tqdm(range(num_epochs)):
        pred_score_list=[]
        true_label_list=[]
        train_l_sum,train_acc_sum,n,batch_count,start = 0.0,0.0,0,0,time.time()
        train_auc_sum,train_p_acc_sum,train_n_acc_sum = 0,0,0
        for idx,sample in tqdm(enumerate(train_iter),total=len(train_iter)):
            if 'mode' in net.__dict__:
                net.mode = 'train'
            net.train()
#             sample={key:sample[key].to(device) for key in sample}
            if fusion:
                y_hat = sample['labels'].to(device)
            
                X_loc,X_len_loc,X_fre_loc,X_dis_loc = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                 sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                time_position_ids_loc = (torch.cumsum(X_dis_loc,dim=1)//400).long()
                idx_list_loc = None
                if 'map_idx' in sample:
                    idx_list_loc = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_loc = None
                if 'location_xy' in sample:
                    geo_position_ids_loc = sample['location_xy'].to(device)

                X_mo,X_len_mo,X_fre_mo,X_dis_mo = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                 sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                time_position_ids_mo = (torch.cumsum(X_dis_mo,dim=1)//400).long()
                idx_list_mo  = None
                if 'map_idx' in sample:
                    idx_list_mo  = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_mo  = None
                if 'mouse_xy' in sample:
                    geo_position_ids_mo  = sample['mouse_xy'].to(device)
                logits,location_logits,mouse_logits,loss,hidden = net(X_loc,X_fre_loc,X_len_loc,X_dis_loc,\
                X_mo,X_fre_mo,X_len_mo,X_dis_mo,y_hat,time_position_ids_loc,idx_list_loc,geo_position_ids_loc\
                ,time_position_ids_mo,idx_list_mo,geo_position_ids_mo)
            else:
                if not mouse:
                    X,X_len,X_fre,X_dis = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                     sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'location_xy' in sample:
                        geo_position_ids = sample['location_xy'].to(device)
                else:
                    X,X_len,X_fre,X_dis = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                     sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'mouse_xy' in sample:
                        geo_position_ids = sample['mouse_xy'].to(device)
                y_hat = sample['labels'].to(device)

                # data transform 只有在不用fre的时候以及是feature的时候用
                if data_transform == 'Cutmix':   
                    lam = np.random.beta(1, 1)
                    rand_index = torch.randperm(X.size()[0])
                    y_a = y_hat
                    y_b = y_hat[rand_index]
                    bbx1, bbx2 = rand_bbox(X.size(), lam)
                    X[:, bbx1:bbx2] = X[rand_index,bbx1:bbx2]

                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - (bbx2 - bbx1) / X.shape[1]
                    # compute output
                    logits,_,hidden,res,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)
                    loss = loss_func(logits, y_a) * lam + loss_func(logits, y_b) * (1. - lam)
                elif data_transform == 'mixup':
                    mixed_X, y_a, y_b, lam = mixup_data(X, y_hat,1)
                    logits,_,hidden,res,seq_embed = net(mixed_X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)

                    loss = mixup_criterion(loss_func, logits, y_a, y_b, lam)
                else:
                    logits,loss,hidden,res,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)
            
            if use_n_gpus:
                loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

            if optimizer is not None:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            net.zero_grad()
            
            global_step +=1
            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()


            if global_step % save_steps == 0:
                save_model(output_dir, net, global_step)
            
            pred_score=torch.softmax(logits,dim=-1)[:,1].detach().cpu().numpy()
            true_label=sample['labels'].detach().cpu().numpy()
            
            pred_score_list.append(pred_score)
            true_label_list.append(true_label)
            
            train_l_sum += loss.cpu().item()
            batch_count+=1
            n+=logits.shape[0]


        pred_score = np.concatenate(pred_score_list,axis=0)
        true_label = np.concatenate(true_label_list,axis=0)
        fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_score)
        train_AUC = metrics.auc(fpr, tpr)
        train_acc = metrics.accuracy_score(true_label,pred_score>0.5)
        temp = precision_recall_fscore_support(true_label,pred_score>0.5)
        train_precision = temp[0][1]
        train_recall = temp[1][1]
        train_f1 = temp[2][1]
        if (num_epoch+1)%eval_iter==0:
            if not fusion:
                val_acc,val_AUC,val_f1,val_recall,val_precision=\
                evaluate_accuracy(val_iter,net,device,fusion,mouse)
                clean_val_acc,clean_val_AUC,clean_val_f1,clean_val_recall,clean_val_precision=\
                val_acc,val_AUC,val_f1,val_recall,val_precision
#                 evaluate_accuracy(val_iter,net,device,fusion,mouse,True)

                tmp_metric_str = "after %d epochs:\n ===\nloss %g\ntrain acc %g\ntrain auc %g\ntrain f1 %g\ntrain recall %g\ntrain precision %g \
               \n===\nval acc %g\nval auc %g\nval f1 %g\nval recall %g\nval precision %g\
               \n===\nclean_val acc %g\nclean_val auc %g\nclean_val f1 %g\nclean_val recall %g\nclean_val precision %g\
               \ntime %.1f"%(num_epoch+1,train_l_sum/batch_count,train_acc,train_AUC,train_f1,train_recall,train_precision,\
                                       val_acc,val_AUC,val_f1,val_recall,val_precision,\
                                       clean_val_acc,clean_val_AUC,clean_val_f1,clean_val_recall,clean_val_precision,time.time()-start)
            else:
                val_acc,val_AUC,val_f1,val_recall,val_precision,\
                loc_val_acc,loc_val_AUC,loc_val_f1,loc_val_recall,loc_val_precision,\
                mo_val_acc,mo_val_AUC,mo_val_f1,mo_val_recall,mo_val_precision=\
                evaluate_accuracy(val_iter,net,device,fusion,mouse)
                clean_val_acc,clean_val_AUC,clean_val_f1,clean_val_recall,clean_val_precision,\
                clean_loc_val_acc,clean_loc_val_AUC,clean_loc_val_f1,clean_loc_val_recall,clean_loc_val_precision,\
                clean_mo_val_acc,clean_mo_val_AUC,clean_mo_val_f1,clean_mo_val_recall,clean_mo_val_precision=\
                val_acc,val_AUC,val_f1,val_recall,val_precision,\
                loc_val_acc,loc_val_AUC,loc_val_f1,loc_val_recall,loc_val_precision,\
                mo_val_acc,mo_val_AUC,mo_val_f1,mo_val_recall,mo_val_precision                
#                 evaluate_accuracy(val_iter,net,device,fusion,mouse,True)

                tmp_metric_str = "after %d epochs:\n ===\nloss %g\ntrain acc %g\ntrain auc %g\ntrain f1 %g\ntrain recall %g\ntrain precision %g \
               \n===\nval acc %g\tval auc %g\tval f1 %g\tval recall %g\tval precision %g\
               \nclean_val acc %g\tclean_val auc %g\tclean_val f1 %g\tclean_val recall %g\tclean_val precision %g\
               \n===\nloc_val acc %g\tloc_val auc %g\tloc_val f1 %g\tloc_val recall %g\tloc_val precision %g\
               \nclean_loc_val acc %g\tclean_loc_val auc %g\tclean_loc_val f1 %g\tclean_loc_val recall %g\tclean_loc_val precision %g\
               \n===\nmouse_val acc %g\tmouse_val auc %g\tmouse_val f1 %g\tmouse_val recall %g\tmouse_val precision %g\
               \nclean_mouse_val acc %g\tclean_mouse_val auc %g\tclean_mouse_val f1 %g\tclean_mouse_val recall %g\tclean_mouse_val precision %g\
               \ntime %.1f"%(num_epoch+1,train_l_sum/batch_count,train_acc,train_AUC,train_f1,train_recall,train_precision,\
                                       val_acc,val_AUC,val_f1,val_recall,val_precision,\
                                       clean_val_acc,clean_val_AUC,clean_val_f1,clean_val_recall,clean_val_precision,\
                                       loc_val_acc,loc_val_AUC,loc_val_f1,loc_val_recall,loc_val_precision,\
                                       clean_loc_val_acc,clean_loc_val_AUC,clean_loc_val_f1,clean_loc_val_recall,clean_loc_val_precision,\
                                       mo_val_acc,mo_val_AUC,mo_val_f1,mo_val_recall,mo_val_precision,\
                                       clean_mo_val_acc,clean_mo_val_AUC,clean_mo_val_f1,clean_mo_val_recall,clean_mo_val_precision,time.time()-start) 
            if (num_epoch+1) % verbose_num == 0:
                print(tmp_metric_str)
                if val_AUC>max_auc:
                    max_auc = val_AUC
                    max_epoch = num_epoch+1
                metric_str += tmp_metric_str + "\n"
                with open(eval_save_path, 'a', encoding='utf-8') as f1:
                    f1.write(tmp_metric_str + "\n")
    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()

    logger.info('Train done')
    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write("max val auc {} max epoch {}\n".format(max_auc,max_epoch))

def get_hidden(train_iter, net, device=None,fusion = False,mouse=False):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    pred_hidden_list = []
#     if fusion:
#         loc_pred_score_list = []
#         mouse_pred_score_list = []

    for idx,sample in  tqdm(enumerate(train_iter),total = len(train_iter)):
        if 'mode' in net.__dict__:
            net.mode = 'test'
        net.eval()
#         sample = {key:sample[key].to(device) for key in sample}
        with torch.no_grad():
            y_hat = sample['labels'].to(device)
            if fusion:

                X_loc,X_len_loc,X_fre_loc,X_dis_loc = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                 sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                time_position_ids_loc = (torch.cumsum(X_dis_loc,dim=1)//400).long()
                idx_list_loc = None
                if 'map_idx' in sample:
                    idx_list_loc = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_loc = None
                if 'location_xy' in sample:
                    geo_position_ids_loc = sample['location_xy'].to(device)

                X_mo,X_len_mo,X_fre_mo,X_dis_mo = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                 sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                time_position_ids_mo = (torch.cumsum(X_dis_mo,dim=1)//400).long()
                idx_list_mo  = None
                if 'map_idx' in sample:
                    idx_list_mo  = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                geo_position_ids_mo  = None
                if 'mouse_xy' in sample:
                    geo_position_ids_mo  = sample['mouse_xy'].to(device)
                logits,location_logits,mouse_logits,loss,hidden = net(X_loc,X_fre_loc,X_len_loc,X_dis_loc,\
                X_mo,X_fre_mo,X_len_mo,X_dis_mo,y_hat,time_position_ids_loc,idx_list_loc,geo_position_ids_loc\
                ,time_position_ids_mo,idx_list_mo,geo_position_ids_mo)  
                length = sample['mouse_length'].detach().cpu().numpy()*sample['location_length'].detach().cpu().numpy()
            else:
                if not mouse:
                    X,X_len,X_fre,X_dis = sample['location_feature'].to(device),sample['location_length'].to(device),\
                                     sample['location_fre_feature'].to(device),sample['location_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'location_xy' in sample:
                        geo_position_ids = sample['location_xy'].to(device)
                    logits,loss,hidden,_,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)
                    length = sample['location_length'].detach().cpu().numpy()
                else:
                    X,X_len,X_fre,X_dis = sample['mouse_feature'].to(device),sample['mouse_length'].to(device),\
                                     sample['mouse_fre_feature'].to(device),sample['mouse_dis'].to(device)
                    time_position_ids = (torch.cumsum(X_dis,dim=1)//400).long()
                    idx_list = None
                    if 'map_idx' in sample:
                        idx_list = [sample['map_idx'].to(device),sample['weekday_idx'].to(device),sample['time_idx'].to(device)]
                    geo_position_ids = None
                    if 'mouse_xy' in sample:
                        geo_position_ids = sample['mouse_xy'].to(device)
                    logits,loss,hidden,_,seq_embed = net(X,X_fre,X_len,X_dis,y_hat,time_position_ids,idx_list,geo_position_ids)    
                    length = sample['mouse_length'].detach().cpu().numpy()
            pred_hidden_list.append(hidden.detach().cpu().numpy())

    return np.concatenate(pred_hidden_list,axis=0)
