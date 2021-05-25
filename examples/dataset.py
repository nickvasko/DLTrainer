import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        assert len(inputs) == len(labels), "inputs and labels need to be the same length."

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs), torch.tensor(self.labels)
