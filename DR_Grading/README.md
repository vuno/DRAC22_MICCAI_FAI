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
1. Download pre-trained models : [LINK](https://drive.google.com/drive/folders/1QbD3Kcp8EjJCvGM2j5H8z568gBoZzdY-?usp=sharing)
2. Move the download folder to ~/DRAC22_FAI/DR_Grading/ckpt
3. Run "evaluation_pretrained_model.py"
