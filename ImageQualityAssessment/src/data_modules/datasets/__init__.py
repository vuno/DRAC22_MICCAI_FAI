from .quality_dataset import QualityDataset


def get_drac_dataset(task, data_dir, split, transform, fold_idx, input_size, u_df=None):
    if task == 'quality':
        return QualityDataset(data_dir, split, transform, fold_idx, input_size, u_df)
    else:
        raise ValueError