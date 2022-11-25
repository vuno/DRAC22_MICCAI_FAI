import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import torch
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_modules.drac_lm import DRACLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_dir", default="/data1/kimjaeyoung_folders/DRAC/", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    task = 'task2'
    input_size = args.image_size
    task_tag = 'B. Image Quality Assessment'
    os.makedirs(f'./submit', exist_ok=True)
    
    # task 2 best models
    models = [os.path.join(args.model_dir, 'outputs_drac/task2/debug/fold0_2-epoch=11-CohenKapp=0.9267-AUROC=0.9335.ckpt'),
              os.path.join(args.model_dir, 'outputs_drac/task2/debug/fold1_2-epoch=13-CohenKapp=0.9285-AUROC=0.9072.ckpt'),
              os.path.join(args.model_dir, 'outputs_drac/task2/debug/fold2_0-epoch=15-CohenKapp=0.9336-AUROC=0.9286.ckpt'),
              os.path.join(args.model_dir, 'outputs_drac/task2/debug/fold3_3-epoch=20-CohenKapp=0.9463-AUROC=0.9274.ckpt'),
              os.path.join(args.model_dir, 'outputs_drac/task2/debug/fold4_3-epoch=34-CohenKapp=0.9508-AUROC=0.9406.ckpt')
    ]
    assert len(models) > 0
    print(len(models))
    num_ensemble = len(models)
    
    data_dir = '/data1/external_data/DRAC'
    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2(),
    ])
    list_img_path = sorted(glob(os.path.join(data_dir, task_tag, '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    preds = [[] for _ in range(len(models))]
    for idx, ckpt_path in enumerate(models):
        lm = DRACLM.load_from_checkpoint(ckpt_path)
        lm.eval()
        lm = lm.cuda()
        for img_path in tqdm(list_img_path):
            img = cv2.imread(img_path)
            img_flip1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # tta
            img_flip2 = cv2.flip(np.copy(img_flip1), 1) # 1은 좌우 반전, 0은 상하 반전입니다.  #cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
            img_flip3 = cv2.flip(np.copy(img_flip1), 0) # 180도 회전
            img_flip4 = cv2.flip(cv2.flip(np.copy(img_flip1), 0), 1)

            img_flip1 = transform(image=img_flip1)['image']
            img_flip1 = img_flip1.unsqueeze(dim=0)

            img_flip2 = transform(image=img_flip2)['image']
            img_flip2 = img_flip2.unsqueeze(dim=0)

            img_flip3 = transform(image=img_flip3)['image']
            img_flip3 = img_flip3.unsqueeze(dim=0)

            img_flip4 = transform(image=img_flip4)['image']
            img_flip4 = img_flip4.unsqueeze(dim=0)
            
            with torch.no_grad():
                pred1 = lm(img_flip1.cuda())
                pred2 = lm(img_flip2.cuda())
                pred3 = lm(img_flip3.cuda())
                pred4 = lm(img_flip4.cuda())
                # # test = (torch.clamp(pred1, min=0, max=2).item() + torch.clamp(pred2, min=0, max=2).item() + torch.clamp(pred3, min=0, max=2).item() + torch.clamp(pred4, min=0, max=2).item()) / 4.

                pred = (pred1 + pred2 + pred3 + pred4) / 4.
                pred = torch.clamp(pred, min=0, max=2).item()
                
            preds[idx].append(pred)
            # break
    
    preds = np.array(preds)
    
    mean_preds = np.mean(preds, axis=0)
    median_preds = np.median(preds, axis=0)
    current_preds = np.round(median_preds)
    total_preds = np.array(preds)
    current_preds = np.round(mean_preds)
            
    df_dict = {'case': [], 'class': [], 'P0': [], 'P1': [], 'P2': [], 'M1': [], 'M2': [],
    'M3': [], 'M4': [], 'M5': [], 'Mean': [], 'Median': []}
    
    count = 0
    for img_path, median_pred, mean_pred, current_pred in zip(list_img_path, median_preds, mean_preds, current_preds):
        df_dict['case'].append(os.path.basename(img_path))
        df_dict['class'].append(int(current_pred))
        for m in range(num_ensemble):
            df_dict[f'M{m+1}'].append(total_preds[m][count])
        df_dict['Mean'].append(mean_pred)
        df_dict['Median'].append(median_pred)
        #print(df_dict)
        if int(current_pred) == 0:
            df_dict['P0'].append(1.)
            df_dict['P1'].append(0.)
            df_dict['P2'].append(0.)
        if int(current_pred) == 1:
            df_dict['P0'].append(0.)
            df_dict['P1'].append(1.)
            df_dict['P2'].append(0.)
        if int(current_pred) == 2:
            df_dict['P0'].append(0.)
            df_dict['P1'].append(0.)
            df_dict['P2'].append(1.)
        count += 1
    
    df = pd.DataFrame(df_dict)
    df.to_csv(f'./submit/predictions_task2.csv', index=False)