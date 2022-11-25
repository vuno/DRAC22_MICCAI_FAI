import os
import numpy as np
import argparse
import glob
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from src.data_modules.drac_dataloader import DRACDM
from src.lightning_modules.drac_lm import DRACLM
from src.utils import set_seed, construct_kfold_dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import pandas as pd
from tqdm import tqdm


def generate_ps_label_task3(conf, ratio, data_dir):
    ratio = ratio / 100.
    #data_dir = '/data1/external_data/DRAC'
    task_tag = 'C. Diabetic Retinopathy Grading'
    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2(),
    ])
    #save_dir = os.path.join(conf.save_dir, conf.pname)
    #print(save_dir)
    ckpt_path = glob.glob(os.path.join(conf.save_dir, f'fold{conf.fold_idx}_{conf.sl_idx}-*.ckpt'))[0]
    #print(ckpt_path)
    #exit()
    list_img_path = sorted(glob.glob(os.path.join(data_dir, task_tag, '1. Original Images', 'b. Testing Set', '*.png')))
    lm = DRACLM.load_from_checkpoint(ckpt_path)
    lm.eval().cuda()
    print("Start to generate ps labels")
    cls_0_files = [] 
    cls_1_files = []
    cls_2_files = []
    cls_0_probs = []
    cls_1_probs = []
    cls_2_probs = []
    for img_path in tqdm(list_img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)['image']
        img = img.unsqueeze(dim=0).cuda()
        
        with torch.no_grad():
            pred = lm(img)
            pred = torch.clamp(pred, min=0, max=2).item()

        pl = round(pred)
        confidence = abs(pl - pred)

        if pl == 0:
            cls_0_files.append(img_path)
            cls_0_probs.append(confidence)
        elif pl == 1:
            cls_1_files.append(img_path)
            cls_1_probs.append(confidence)
        else:
            cls_2_files.append(img_path)
            cls_2_probs.append(confidence)
    sorted_0_idx = np.argsort(cls_0_probs)
    sorted_1_idx = np.argsort(cls_1_probs)
    sorted_2_idx = np.argsort(cls_2_probs)
    cls_0_files = np.array(cls_0_files)[sorted_0_idx]
    cls_1_files = np.array(cls_1_files)[sorted_1_idx]
    cls_2_files = np.array(cls_2_files)[sorted_2_idx]

    num_0 = len(cls_0_files)
    num_1 = len(cls_1_files)
    num_2 = len(cls_2_files)
    target_0_num = int(num_0 * ratio)
    target_1_num = int(num_1 * ratio)
    target_2_num = int(num_2 * ratio)
    print("*******************************************")
    print(f"RATIO {ratio} | PS 0 : {target_0_num}, PS 1 : {target_1_num}, PS 2 : {target_2_num}")
    print("*******************************************")

    data = {"image name": [], "DR grade": [], "split": []}

    if target_0_num > 0:
        t_0_files = cls_0_files[:target_0_num].tolist()
        data["image name"] += t_0_files
        label = [0] * target_0_num
        split = ["train"] * target_0_num
        data["split"] += split
        data["DR grade"] += label
    if target_1_num > 0:
        t_1_files = cls_1_files[:target_1_num].tolist()
        data["image name"] += t_1_files
        label = [1] * target_1_num
        data["DR grade"] += label
        split = ["train"] * target_1_num
        data["split"] += split
    if target_2_num > 0:
        t_2_files = cls_2_files[:target_2_num].tolist()
        data["image name"] += t_2_files
        label = [2] * target_2_num
        data["DR grade"] += label
        split = ["train"] * target_2_num
        data["split"] += split

    u_df = pd.DataFrame(data=data)
    return u_df


def run(conf, u_df):
    num_classes = 1 if conf.loss in ['mse', 'smoothl1loss'] else 3
    dm = DRACDM('grading', conf.data_dir, conf.input_size, conf.batch_size, conf.num_workers, conf.balanced_sampling, conf.fold_idx, conf.auto_aug, u_df)
    lm = DRACLM('grading', conf.backbone, conf.lr, conf.optimizer, conf.loss, conf.beta1, conf.decay_step, conf.mix_up, num_classes)
    tb_logger = loggers.TensorBoardLogger(save_dir=conf.save_dir)
    ckpt_callback = ModelCheckpoint(
        dirpath=conf.save_dir,
        monitor='val/CohenKappa',
        filename=f'fold{conf.fold_idx}_{conf.sl_idx}'+'-epoch={epoch:02d}-CohenKapp={val/CohenKappa:.4f}-AUROC={val/AUROC:.4f}',
        auto_insert_metric_name=False,
        mode='max'
    )
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(logger=tb_logger, 
                      callbacks=[ckpt_callback, lr_callback],
                      default_root_dir=conf.save_dir,
                      devices=1, #[int(x) for x in conf.gpu_id.split(',')],
                      max_epochs=conf.max_epochs,
                      log_every_n_steps=1,
                      accelerator='gpu',
                      strategy=DDPPlugin(find_unused_parameters=True) if len(conf.gpu_id.split(',')) > 1 else None,
                      precision=16,
                      sync_batchnorm=True if len(conf.gpu_id.split(',')) > 1 else False,
                      num_sanity_val_steps=0,
                      )
    trainer.fit(lm, datamodule=dm)
    return ckpt_callback.best_model_score


def parse_argument():
    parser = argparse.ArgumentParser()
    boolean_flags = ['true', 'yes', '1', 't', 'y']
    is_boolean = lambda x: x.lower() in boolean_flags
    
    # Project Parameters
    parser.add_argument('--seed', type=int, default=42) # default 42
    parser.add_argument('--pname', type=str, default='debug')
    parser.add_argument('--save_dir', type=str, default='./outputs_drac/task3')
    #parser.add_argument('--task', type=str, default='grading', choices=['quality', 'grading', 'segment'])
    
    # Data Module Parameters
    parser.add_argument('--data_dir', type=str, default='/data1/external_data/DRAC')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--balanced_sampling', type=is_boolean, default=False)
    parser.add_argument('--auto_aug', type=str, default=None)
    
    # Lightning Module Parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b2')
    parser.add_argument('--loss', type=str, default='smoothl1loss', choices=['ce', 'focal', 'mse', 'smoothl1', 'ls', 'qwk', 'qwk2', 'smoothl1loss'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--decay_step', type=int, default=30)
    parser.add_argument('--mix_up', type=is_boolean, default=False)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd', 'radam'])
    parser.add_argument('--grad_clip', default=0.0, type=float)
    
    # Trainer Parameters
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--max_epochs', type=int, default=100)
    
    # K-fold validation
    parser.add_argument('--num_kfold', type=int, default=5)
    parser.add_argument('--target_fold', type=int, default=0)

    # algorithms
    #parser.add_argument('--algo', default='supervised', type=str, choices=['supervised', 'meta_pseudo'])
    
    return parser.parse_args()

if __name__ == '__main__':
    conf = parse_argument()
    set_seed(conf.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_id
    conf.save_dir = os.path.join(conf.save_dir, conf.pname)
    os.makedirs(conf.save_dir, exist_ok=True)
    conf.task = 'grading'

    # construct k-fold dataset
    construct_kfold_dataset(conf)
    
    total_scores = []
    ratio = 0
    u_df = None
    conf.fold_idx = conf.target_fold
    fold_best_score = 0
    for sl_idx in range(5):
        conf.sl_idx = sl_idx
        score = run(conf, u_df)
        if fold_best_score < score.item():
            fold_best_score = score.item()
            ratio += 20
            # generate ps label
            u_df = generate_ps_label_task3(conf, ratio, conf.data_dir)
    print('*********************************************')
    print(f'FOLD {conf.target_fold} | Quadratic Cohen Kappa : {fold_best_score}')
    print('*********************************************')