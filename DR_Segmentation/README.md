# DR segmentation

## Dataset
* [Download Link](https://drac22.grand-challenge.org/)
* After downloading, move downloaded folder to ~/DRAC22_FAI/datasets
   
## Runs

### Arguments
* **target:** 1 (microvascular abnormality), 2 (nonperfusion area), and 3 (neovascularization)
* **save_dir:** checkpoint directory

### Train

```bash
# Train the segmentation model with class 1
python train.py --gpu_id {GPU_ID} --target 1
# Train the segmentation model with class 2
python train.py --gpu_id {GPU_ID} --target 2
# Train the segmentation model with class 3
python train.py --gpu_id {GPU_ID} --target 3
```

#### Evaluation with pre-trained models
1. Download pre-trained models : [LINK](https://vunocorp-my.sharepoint.com/:f:/g/personal/jaeyoung_kim_vuno_co/EqbdOLqSRWJOjGGUQm9HPUkBgwVFWg1mLyPHgS9wRWXMfQ?e=HuQsqv)
2. Move the download folder to ~/DRAC22_FAI/DR_Segmentation/ckpt
3. Run "evaluation_pretrained_model.py"

