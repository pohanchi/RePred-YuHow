import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from torch import nn 
import IPython
import pdb

# define dataset
class Resisitivity_Dataset(Dataset):
    def __init__(self, path, config, mode):
        f = open(path, "r")
        data_list = f.readlines()
        dataset = []
        for item in data_list:
            item_clear=item.strip().split()
            float_item = torch.FloatTensor((list(map(float, item_clear))))
            dataset.append(float_item)
        
        self.dataset = torch.stack(dataset)[:,1:]
        self.label = torch.stack(dataset)[:,1]

        num_sample = int(len(self.dataset) * 0.9)

        if mode =="train":
            self.dataset = self.dataset[:num_sample]
            self.label = self.label[:num_sample]
        else:
            self.dataset = self.dataset[num_sample:]
            self.label = self.label[num_sample:]

    def __getitem__(self, idx):
        data=self.dataset[idx]
        label= self.label[idx]
        
        return data,label
    
    def __len__(self):
        return len(self.dataset)