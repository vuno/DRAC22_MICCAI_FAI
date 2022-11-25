import os
import cv2
import numpy as np
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

from src.lightning_modules.drac_lm import DRACLM


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_dir', default='./ckpt', type=str)
    parser.add_argument('--data_dir', default='./datasets', type=str)
    
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--reduce_fn', default='mean', type=str, choices=['mean', 'median'])
    parser.add_argument('--pix_threshold', default=5, type=int)
    parser.add_argument('--cutoff_v1', default=0.53, type=float)
    parser.add_argument('--cutoff_v2', default=1.5, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    # init
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    task_tag = 'B. Image Quality Assessment'
    os.makedirs(f'./submit', exist_ok=True)
    
    # models
    models = glob(os.path.join(args.model_dir, '*.ckpt'))
    assert len(models) > 0
    n_ensemble = len(models)
    
    # data
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
    ])
    list_img_path = sorted(glob(os.path.join(args.data_dir, task_tag, '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # inference
    preds = [[] for _ in range(n_ensemble)]
    low_pix = []
    for model_idx, ckpt_path in enumerate(models):
        lm = DRACLM.load_from_checkpoint(ckpt_path)
        lm.eval()
        lm.cuda()
        
        for img_path in tqdm(list_img_path):
            img = cv2.imread(img_path)
            
            # calc low pix
            low_pix.append(np.count_nonzero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < args.pix_threshold))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs = [transform(image=img)['image']]
            if args.tta:
                imgs.append(transform(image=cv2.flip(img, 1))['image'])
                imgs.append(transform(image=cv2.flip(img, 0))['image'])
                imgs.append(transform(image=cv2.flip(cv2.flip(img, 1), 0))['image'])
            
            imgs = torch.stack(imgs)
            imgs = imgs.cuda()
            
            with torch.no_grad():
                out = lm(imgs)
                pred = torch.clamp(torch.mean(out), min=0, max=2).item()
            preds[model_idx].append(pred)

    # generate submission file
    preds = np.array(preds)
    if args.reduce_fn == 'mean':
        preds_reduced = np.mean(preds, axis=0)
    else:
        preds_reduced = np.median(preds, axis=0)
    
    label_preds = preds_reduced.copy()
    label_preds[label_preds < args.cutoff_v1] = 0
    label_preds[(label_preds >= args.cutoff_v1) & (label_preds < args.cutoff_v2)] = 1
    label_preds[label_preds >= args.cutoff_v2] = 2
    
    df_dict = {'case': [], 'class': [], 'P0': [], 'P1': [], 'P2': []}
    
    for idx, (img_path, pred_r, lbl, lp) in enumerate(zip(list_img_path, preds_reduced, label_preds, low_pix)):
        df_dict['case'].append(os.path.basename(img_path))
        
        # heuristics post-processing
        gap = max(preds[:, idx]) - min(preds[:, idx])
        if lbl == 1 and (gap > 0.8 or (gap > 0.68 and lp >= 80000)):
            lbl = lbl - 1
        
        df_dict['class'].append(int(lbl))
        
        if int(lbl) == 0:
            df_dict['P0'].append(1.)
            df_dict['P1'].append(0.)
            df_dict['P2'].append(0.)
        if int(lbl) == 1:
            df_dict['P0'].append(0.)
            df_dict['P1'].append(1.)
            df_dict['P2'].append(0.)
        if int(lbl) == 2:
            df_dict['P0'].append(0.)
            df_dict['P1'].append(0.)
            df_dict['P2'].append(1.)
    
    df = pd.DataFrame(df_dict)
    df.to_csv(f'./submit/DRAC_task2.csv', index=False)