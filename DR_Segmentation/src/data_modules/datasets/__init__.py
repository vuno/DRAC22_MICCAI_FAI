from .quality_dataset import QualityDataset
from .grading_dataset import GradingDataset


def get_drac_dataset(task, data_dir, split, transform, fold_idx):
    if task == 'quality':
        return QualityDataset(data_dir, split, transform, fold_idx)
    
    elif task == 'grading':
        return GradingDataset(data_dir, split, transform, fold_idx)
    
    elif task == 'segment':
        return NotImplementedError
    
    else:
        raise ValueError