# DR Grading

## Dataset
* [Download Link](https://drac22.grand-challenge.org/)
* After downloading, move downloaded folder to ~/DRAC22_FAI/datasets
   
## Runs

### Arguments
* **save_dir:** checkpoint directory

### Train

```bash
sh run_task3.sh
```

#### Evaluation with pre-trained models
1. Download pre-trained models : [LINK](https://vunocorp-my.sharepoint.com/:f:/g/personal/jaeyoung_kim_vuno_co/ElQhUkwNH4dPpkLGnAO0_sUBk6TDcH6RR4SehD_Nzpg96A?e=XwHdyQ)
2. Move the download folder to ~/DRAC22_FAI/DR_Grading/ckpt
3. Run "evaluation_pretrained_model.py"
