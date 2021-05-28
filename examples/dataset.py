import torch
from torch.utils.data import Dataset
import pickle
import os

from .util import create_sample_data


class SimpleDataset(Dataset):
    def __init__(self, args, file_type):
        cache_file = f'{args.data_dir}/{file_type}.pkl'
        if not os.path.exists(cache_file):
            print(f'dataset not found in {cache_file}. creating it now.')
            create_sample_data(100000)

        data = pickle.load(open(cache_file, 'rb'))

        self.inputs = data['X']
        self.labels = data['y']
        assert len(self.inputs) == len(self.labels), "inputs and labels need to be the same length."

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return (torch.tensor(self.inputs[item]),), \
               torch.tensor(self.labels[item])
