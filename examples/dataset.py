import torch
from torch.utils.data import Dataset
import pickle


class SimpleDataset(Dataset):
    def __init__(self, data_dir, file_type):
        cache_file = f'{data_dir}/{file_type}.pkl'
        data = pickle.load(open(cache_file, 'rb'))

        self.inputs = data['X']
        self.labels = data['y']
        assert len(self.inputs) == len(self.labels), "inputs and labels need to be the same length."

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return (torch.tensor(self.inputs[item]),), \
               torch.tensor(self.labels[item])
