import argparse
import os
import cv2
import yaml
import numpy as np
import pandas as pd
import SimpleITK
from glob import glob
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
import torch

from src.lightning_modules.task1_lm import Task1LM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--model_dir", default="./ckpt/Task1", type=str)
    parser.add_argument("--data_dir", default="/data1/external_data/DRAC", type=str)
    parser.add_argument("--post_edit", default=1, type=int)
    return parser.parse_args()


def multi_scale_inference(model, image, scales=[1, 1.1, 1.2, 1.3, 1.4, 1.5], flip=False):
    batch, _, ori_height, ori_width = image.size()
    num_classes = 3
    crop_size = 512
    stride_h = 256
    stride_w = 256
    assert batch == 1, "only supporting batchsize 1."
    image = image.numpy()[0].transpose((1,2,0)).copy()
    final_pred = torch.zeros([1, num_classes, ori_height, ori_width]).cuda()
    for scale in scales:
        resize_w = int(scale * ori_width)
        resize_h = int(scale * ori_height)
        new_img = cv2.resize(np.uint8(image * 255), (resize_w, resize_h))
        new_img = new_img / 255.

        height, width = new_img.shape[:-1]
            
        if scale <= 1.0:
            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.FloatTensor(new_img)
            preds = model(new_img.cuda())
            preds = torch.sigmoid(preds[0])
        else:
            new_h, new_w = new_img.shape[:-1]
            rows = np.int(np.ceil(1.0 * (new_h - crop_size) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w - crop_size) / stride_w)) + 1
            preds = torch.zeros([1, num_classes, new_h,new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + crop_size, new_h)
                    w1 = min(w0 + crop_size, new_w)
                    h0 = max(int(h1 - crop_size), 0)
                    w0 = max(int(w1 - crop_size), 0)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.FloatTensor(crop_img)
                    pred = model(crop_img.cuda())
                    pred = torch.sigmoid(pred[0])
                    preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                    count[:,:,h0:h1,w0:w1] += 1
            preds = preds / count
            preds = preds[:,:,:height,:width]

        preds = F.interpolate(
            preds, (ori_height, ori_width), 
            mode='bilinear', align_corners=False
        )            
        final_pred += preds
    final_pred = final_pred / len(scales)
    return final_pred


def arr2nii(data, filename, reference_name=None):
    img = SimpleITK.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = SimpleITK.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    SimpleITK.WriteImage(img, filename)


def inference_class_1(args, pname):
    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    ensemble_preds = np.zeros([5, 65, 3, 1024, 1024])
    for seed_idx, seed in enumerate([42, 43, 44, 45, 46]):
        lm = Task1LM.load_from_checkpoint(glob(f'{args.model_dir}/class_1/{pname}/{seed}_class_{target_class}_*.ckpt')[0])
        lm.eval().cuda()

        for file_idx, img_path in enumerate(tqdm(list_img_path)):
            filename = os.path.basename(img_path).split('.')[0]
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img0 = np.copy(img)
            img90 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
            img180 = cv2.rotate(img0, cv2.ROTATE_180) # 180도 회전
            img270 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vis = Image.fromarray(img)
            vis.save(f'./submit/task1/class1/{pname}/fig/{filename}_origin.png')
            
            img0 = transform(image=img0)['image']
            img0 = img0.unsqueeze(dim=0)

            img90 = transform(image=img90)['image']
            img90 = img90.unsqueeze(dim=0)

            img180 = transform(image=img180)['image']
            img180 = img180.unsqueeze(dim=0)

            img270 = transform(image=img270)['image']
            img270 = img270.unsqueeze(dim=0)

            with torch.no_grad():
                e_pred = np.zeros([1024, 1024, 3])
                for z, img in enumerate([img0, img90, img180, img270]):
                    img = img.cuda()
                    pred = lm(img)
                    pred = np.transpose(torch.sigmoid(pred[0]).squeeze().detach().cpu().numpy(), [1, 2, 0])
                    if z == 1:
                        pred = cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif z == 2:
                        pred = cv2.rotate(pred, cv2.ROTATE_180)
                    elif z == 0:
                        pred = pred
                    else:
                        pred = cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE)
                    e_pred += pred
                pred = e_pred / 4.

                vis = Image.fromarray(pred[:,:,1]*255).convert('RGB')
                vis.save(f'./submit/task1/class2/{pname}/fig/{filename}_{args.target}_{seed}.png')

                pred = np.transpose(pred, [2, 0, 1])
                ensemble_preds[seed_idx, file_idx] = pred

    ensemble_preds = np.round(np.mean(ensemble_preds, axis=0))
    for z, img_path in enumerate(list_img_path):
        filename = os.path.basename(img_path).split('.')[0]
        vis = Image.fromarray(ensemble_preds[z, 0,:,:]*255).convert('RGB')
        vis.save(f'./submit/task1/class1/{pname}/fig/{filename}_{args.target}_ensemble.png')


def inference_class_2(args, pname):
    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    ensemble_preds = np.zeros([5, 65, 3, 1024, 1024])
    for seed_idx, seed in enumerate([42, 43, 44, 45, 46]):
        lm = Task1LM.load_from_checkpoint(glob(f'{args.model_dir}/class_2/{pname}/{seed}_class_avg_*.ckpt')[0])
        lm.eval().cuda()

        for file_idx, img_path in enumerate(tqdm(list_img_path)):
            filename = os.path.basename(img_path).split('.')[0]
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img0 = np.copy(img)
            img90 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
            img180 = cv2.rotate(img0, cv2.ROTATE_180) # 180도 회전
            img270 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vis = Image.fromarray(img)
            vis.save(f'./submit/task1/class1/{pname}/fig/{filename}_origin.png')
            
            img0 = transform(image=img0)['image']
            img0 = img0.unsqueeze(dim=0)

            img90 = transform(image=img90)['image']
            img90 = img90.unsqueeze(dim=0)

            img180 = transform(image=img180)['image']
            img180 = img180.unsqueeze(dim=0)

            img270 = transform(image=img270)['image']
            img270 = img270.unsqueeze(dim=0)

            with torch.no_grad():
                e_pred = np.zeros([1024, 1024, 3])
                for z, img in enumerate([img0, img90, img180, img270]):
                    img = img.cuda()
                    pred = lm(img)
                    pred = np.transpose(torch.sigmoid(pred[0]).squeeze().detach().cpu().numpy(), [1, 2, 0])
                    if z == 1:
                        pred = cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif z == 2:
                        pred = cv2.rotate(pred, cv2.ROTATE_180)
                    elif z == 0:
                        pred = pred
                    else:
                        pred = cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE)
                    e_pred += pred
                pred = e_pred / 4.

                vis = Image.fromarray(pred[:,:,1]*255).convert('RGB')
                vis.save(f'./submit/task1/class2/{pname}/fig/{filename}_{args.target}_{seed}.png')

                pred = np.transpose(pred, [2, 0, 1])
                ensemble_preds[seed_idx, file_idx] = pred

    ensemble_preds = np.round(np.mean(ensemble_preds, axis=0))
    for z, img_path in enumerate(list_img_path):
        filename = os.path.basename(img_path).split('.')[0]
        vis = Image.fromarray(ensemble_preds[z, 1,:,:]*255).convert('RGB')
        vis.save(f'./submit/task1/class2/{pname}/fig/{filename}_{args.target}_ensemble.png')


def inference_class_3(args, pname):
    transform = A.Compose([
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2(),
    ])

    lm = Task1LM.load_from_checkpoint(glob(f'{args.model_dir}/class_3/{pname}/*.ckpt')[0])
    lm.eval().cuda()

    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    for img_path in tqdm(list_img_path):
        filename = os.path.basename(img_path).split('.')[0]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img0 = np.copy(img)
        img90 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE) 
        img180 = cv2.rotate(img0, cv2.ROTATE_180) 
        img270 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

        vis = Image.fromarray(img)
        vis.save(f'./submit/task1/class3/{pname}/fig/{filename}_origin.png')
        
        img0 = transform(image=img0)['image']
        img0 = img0.unsqueeze(dim=0)

        img90 = transform(image=img90)['image']
        img90 = img90.unsqueeze(dim=0)

        img180 = transform(image=img180)['image']
        img180 = img180.unsqueeze(dim=0)

        img270 = transform(image=img270)['image']
        img270 = img270.unsqueeze(dim=0)

        with torch.no_grad():
            e_pred = np.zeros([1024, 1024, 3])
            for z, img in enumerate([img0, img90, img180, img270]):
                pred = multi_scale_inference(lm2, img)
                pred = np.transpose(pred.squeeze().detach().cpu().numpy(), [1, 2, 0])
                
                if z == 1:
                    pred = cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif z == 2:
                    pred = cv2.rotate(pred, cv2.ROTATE_180)
                elif z == 0:
                    pred = pred
                else:
                    pred = cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE)
                e_pred += pred
            pred = e_pred / 4.
            pred = np.transpose(pred, [2, 0, 1])

            vis = Image.fromarray(pred[2,:,:]*255).convert('RGB')
            vis.save(f'./submit/task1/class3/{pname}/fig/{filename}_{args.target}.png')


def rm_duplicate_region_class1(c1, c3):
    result = np.copy(c1)
    fret, img_binary = cv2.threshold(c3, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    origin_c1_mask = np.copy(c1)

    n_c1 = c1 / 255.
    n_c3 = c3 / 255.

    mul_mask = np.ones([1024, 1024]).astype(np.float32)
    mask = np.zeros([1024, 1024, 3]).astype(np.uint8)
    for cnt in contours:
        #print(np.unique(mask))
        rect = cv2.boundingRect(cnt)
        xmin, ymin, width, height = rect

        temp_mask = np.zeros([1024, 1024, 3])
        
        mask = cv2.drawContours(temp_mask, [cnt], 0, (1, 1, 1), -1)  

        patch_mask = mask[ymin:ymin+height, xmin:xmin+width]
        
        c1_con_area = n_c1[ymin:ymin+height, xmin:xmin+width] * patch_mask[..., 0] 
        c3_con_area = n_c3[ymin:ymin+height, xmin:xmin+width] * patch_mask[..., 0] 
        mean_c1_prob = np.mean(c1_con_area)
        mean_c3_prob = np.mean(c3_con_area)
        max_c1_prob = np.max(c1_con_area)
        max_c3_prob = np.max(c3_con_area)
        b1 = c1_con_area >= 0.5
        b3 = c1_con_area >= 0.5
        dice_score = get_dice_score(b1, b3)
        if np.sum(c1_con_area) > 0:
            if mean_c3_prob > mean_c1_prob and dice_score > 0.95:
                mul_mask[np.where(mask[..., 0] == 1)] = 0

    b_result = np.float32(result >= 255 * 0.2)
    mask = (b_result * mul_mask)

    if np.sum(mask) > 50:
        return np.uint8(mask * 255)
    else:
        return np.uint8(result >= 127) * 255


def rm_duplicate_region_class3(c1, c3):
    result = np.copy(c3)
    fret, img_binary = cv2.threshold(c3, 255 * 0.3, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    origin_c3_mask = np.copy(c3)

    n_c1 = c1 / 255.
    n_c3 = c3 / 255.

    mask = np.zeros([1024, 1024, 3]).astype(np.uint8)
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        xmin, ymin, width, height = rect
        
        cv2.drawContours(mask, [cnt], 0, (1, 1, 1), -1)
        
        c1_con_area = n_c1[ymin:ymin+height, xmin:xmin+width] 
        c3_con_area = n_c3[ymin:ymin+height, xmin:xmin+width] 
        mean_c1_prob = np.mean(c1_con_area)
        mean_c3_prob = np.mean(c3_con_area)
        max_c1_prob = np.max(c1_con_area)
        max_c3_prob = np.max(c3_con_area)
        if np.sum(c1_con_area) > 0:
            if max_c1_prob >= max_c3_prob:
                mask[ymin:ymin+height, xmin:xmin+width] = 0
    mask = mask[..., 0]

    if np.sum(mask) > 100:
        return np.uint8(mask * 255)
    else:
        return np.uint8(result >= 127) * 255


def post_edit_class2(args, pname):
    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    for img_path in tqdm(list_img_path):
        filename = img_path.split("/")[-1].replace(".png", "")
        mask_all_path = f"./submit/task1/class2/{pname}/fig/{filename}_2_ensemble.png"
        mask_all = np.array(Image.open(mask_all_path).convert("RGB"))[..., 1] / 255.
        origin_mask = np.copy(mask_all)
        post_mask = np.uint8(mask_all >= 0.5) * 255
        post_mask = cv2.dilate(post_mask, np.ones([5, 5])) 

        fret, img_binary = cv2.threshold(post_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros([1024, 1024, 3]).astype(np.uint8)
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            xmin, ymin, width, height = rect
            if width * height > 100:
                cv2.drawContours(mask, [cnt], 0, (1, 1, 1), -1)
        post_mask = np.uint8(mask[..., 0] * 255)
        Image.fromarray(diff_image).save(f"./submit/task1/class2/{pname}/fig/{filename}_{args.target}_post.png")


def post_edit_class1(args, pname, pnamec3):
    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    for img_path in tqdm(list_img_path):
        filename = img_path.split("/")[-1].replace(".png", "")
        c1_mask_path1 = f"./submit/task1/class1/{pname}/fig/{filename}_1_44.png"
        c1_mask_path2 = f"./submit/task1/class1/{pname}/fig/{filename}_1_45.png"
        c3_mask_path = f"./submit/task1/class3/{pnamec3}/fig/{filename}_3.png"

        c1_mask1 = np.array(Image.open(c1_mask_path1).convert("RGB"))[..., 0]
        c1_mask2 = np.array(Image.open(c1_mask_path2).convert("RGB"))[..., 0]
        c3_mask = np.array(Image.open(c3_mask_path).convert("RGB"))[..., 2]

        c1_mask = (c1_mask1 / 255.) + (c1_mask2 / 255.)
        c1_mask = np.uint8(c1_mask / 2. * 255.)

        post_mask = rm_duplicate_region_class1(c1_mask, c3_mask)
        Image.fromarray(post_mask).save(f"./submit/task1/class1/{pname}/fig/{filename}_{args.target}_post.png")


def post_edit_class3(args, pname, pnamec1):
    list_img_path = sorted(glob(os.path.join(args.data_dir, 'A. Segmentation', '1. Original Images', 'b. Testing Set', '*.png')), reverse=False, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(len(list_img_path))

    for img_path in tqdm(list_img_path):
        filename = img_path.split("/")[-1].replace(".png", "")

        c3_path = f"./submit/task1/class3/{pname}/fig/{filename}_3.png" 
        c1_path = f"./submit/task1/class1/{pnamec1}/fig/{filename}_1_45.png"
        c3_mask = np.array(Image.open(c3_path).convert("RGB"))[..., 2] #/ 255.
        c1_mask = np.array(Image.open(c1_path).convert("RGB"))[..., 0] #/ 255.
        post_mask = rm_duplicate_region_class3(c1_mask, c3_mask)
        Image.fromarray(post_mask).save(f"./submit/task1/class3/{pname}/fig/{filename}_{args.target}_post.png")


if __name__ == "__main__":

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    pnames = {2:'train_all_u2net_lite_cosine_focal', 1:'train_all_u2net_full_ensemble', 3:'train_all_u2net_full'}

    for c in range(1, 4):
        args.target = c
        if args.target == 2:
            pname = pnames[args.target]
            os.makedirs(f'./submit/task1/class2/{pname}/fig', exist_ok=True)
            inference_class_2(args, pname)
        elif args.target == 1:
            pname = pnames[args.target]
            os.makedirs(f'./submit/task1/class1/{pname}/fig', exist_ok=True)
            inference_class_1(args, pname)
        elif args.target == 3:
            pname = pnames[args.target]
            os.makedirs(f'./submit/task1/class3/{pname}/fig', exist_ok=True)
            inference_class_3(args, pname)
        else:
            raise ValueError

    if args.post_edit:
        for c in range(1, 4):
            args.target = c
            if args.target == 1:
                pname = pnames[args.target]
                post_edit_class1(args, pname, pnames[3])
            elif args.target == 2:
                pname = pnames[args.target]
                post_edit_class2(args, pname)
            else:
                pname = pnames[args.target]
                post_edit_class3(args, pname, pnames[1])


