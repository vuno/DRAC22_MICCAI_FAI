# Image Quality Assessment

## Dataset
* [Download Link](https://drac22.grand-challenge.org/)
<!-- * After downloading, move downloaded folder to ~/DRAC22_FAI/datasets -->
   
## Runs

### Arguments
* **save_dir:** checkpoint directory

### Train

```bash
sh run_task2.sh
```

#### Create a file for submission with our pre-trained models
1. Download pre-trained models : [LINK](https://drive.google.com/drive/folders/1mHUd-XVGoL9uQzZvJ2knfb4M0d1aSsWy?usp=drive_link)
2. Run 'create_submission_file.py'
```bash
python create_submission_file.py --model_dir <model_dir> --data_dir <data_dir> --tta
```
