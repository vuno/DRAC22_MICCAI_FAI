import numpy as np

from torch.utils.data import Sampler


class CycleSampler(Sampler):
    def __init__(self, dataset):
        self.len = len(dataset)
        self.num_classes = len(np.unique(dataset.labels))

        self.sample_pool = {}
        for t in range(self.num_classes):
            self.sample_pool[t] = np.where(dataset.labels == t)[0]

    def __iter__(self):
        num_pool = len(self.sample_pool)

        for i in range(len(self)):
            t = i % num_pool
            idx = np.random.randint(len(self.sample_pool[t]))
            yield self.sample_pool[t][idx]

    def __len__(self):
        return self.len