import warnings
warnings.filterwarnings('ignore')

import wandb 

import numpy as np
import pandas as pd

import random
import os, shutil, gc, yaml

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from attrdict import AttrDict
import time
import copy
import joblib
from joblib import Parallel, delayed

from IPython import display as ipd
from colorama import Fore, Back, Style
c_ = Fore.GREEN
sr_ = Style.RESET_ALL

import cv2
import albumentations as A
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import timm

import rasterio

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
## custom
from utils import *
from data import *
from model import *
from scheduler import *
from loss import *

#################  config  #################
config_name = 'unet-efficient-b3.yaml'
with open(f'./configs/{config_name}', 'r') as f:
    CFG = AttrDict(yaml.load(f, yaml.FullLoader))
print(CFG)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    train_scores = []
    class_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        # print(masks.shape, y_pred.shape)
        train_dice1 = dice_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        train_dice2 = dice_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        train_dice3 = dice_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        class_scores.append([train_dice1, train_dice2, train_dice3])
        
        train_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        train_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        train_scores.append([train_dice, train_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        
    train_scores = np.mean(train_scores, axis=0)
    class_scores = np.mean(class_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, train_scores, class_scores

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    class_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        
        train_dice1 = dice_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        train_dice2 = dice_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        train_dice3 = dice_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        class_scores.append([train_dice1, train_dice2, train_dice3])
        
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
    val_scores  = np.mean(val_scores, axis=0)
    class_scores = np.mean(class_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores, class_scores

def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss, train_scores, train_class_scores = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        valid_loss, valid_scores, valid_class_scores = valid_one_epoch(model, valid_loader, 
                                                 device=CFG.device, 
                                                 epoch=epoch)
        train_dice, train_jaccard = train_scores
        valid_dice, valid_jaccard = valid_scores
        
        print("train class score", train_class_scores, np.mean(train_class_scores, axis=0))
        print("valid class score", valid_class_scores, np.mean(valid_class_scores, axis=0))
    
        history['Train Loss'].append(train_loss)
        history['Train Dice'].append(train_dice)
        history['Train Jaccard'].append(train_jaccard)
        history['Valid Loss'].append(valid_loss)
        history['Valid Dice'].append(valid_dice)
        history['Valid Jaccard'].append(valid_jaccard)
        
        # Log the metrics
        wandb.log({"Train Loss": train_loss,
                   "Train Dice": train_dice,
                   "Train Jaccard": train_jaccard,
                   "Train class1": train_class_scores[0],
                   "Train class2": train_class_scores[1],
                   "Train class3": train_class_scores[2],
                   "Valid Loss": valid_loss,
                   "Valid Dice": valid_dice,
                   "Valid Jaccard": valid_jaccard,
                   "Valid class1": valid_class_scores[0],
                   "Valid class2": valid_class_scores[1],
                   "Valid class3": valid_class_scores[2],
                   "LR":scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {valid_dice:0.4f} | Valid Jaccard: {valid_jaccard:0.4f}')
        
        # deep copy the model
        if valid_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {valid_dice:0.4f})")
            best_dice    = valid_dice
            best_jaccard = valid_jaccard
            best_epoch   = epoch
            run.summary["Best Dice"]    = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"/jckim/seg/pths/{CFG.comment}/best_epoch-{fold:02d}-dice{best_dice:.4f}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            # wandb.save(PATH)
            print(f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"/jckim/seg/pths/{CFG.comment}/last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


os.makedirs(f"./pths/{CFG.comment}", exist_ok=True) 

set_seed(CFG.seed)

# data setting
path_df = pd.DataFrame(glob(CFG.data_root), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))

df = pd.read_csv('/jckim/seg/train_.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len)

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks

df = df.drop(columns=['image_path','mask_path'])
df = df.merge(path_df, on=['id'])

fault1 = 'case7_day0'
fault2 = 'case81_day30'
df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
    df.loc[val_idx, 'fold'] = fold
    
    
# env setting
model = build_model(CFG)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer, CFG)


for fold in CFG.folds:
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
    run = wandb.init(project='uw-segmentation', 
                     config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
                     anonymous='must',
                     name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
                     group=CFG.comment,
                    )
    train_loader, valid_loader = prepare_loaders(fold, df, get_train_transforms(CFG), get_valid_transforms(CFG), CFG,  debug=CFG.debug)
    model     = build_model(CFG)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer, CFG)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CFG.device,
                                  num_epochs=CFG.epochs)
    run.finish()
    # display(ipd.IFrame(run.url, width=1000, height=720))