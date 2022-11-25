from .grading_dataset import GradingDataset


def get_drac_dataset(task, data_dir, split, transform, fold_idx, input_size, u_df=None):
    return GradingDataset(data_dir, split, transform, fold_idx, input_size, u_df)