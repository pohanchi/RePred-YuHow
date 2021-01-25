import torch
from torch import nn

# define Model
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.linear = nn.Linear(config['input_size'], config['middle_size'])
        self.act_fn = nn.ReLU()
        self.linear1 = nn.Linear(config['middle_size'], config['middle_size'])
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(config['middle_size'], config['output_size'])
    def forward(self, data):
        hid = self.linear(data)
        hid = self.act_fn(hid)
        hid = self.linear1(hid)
        hid = self.act_fn(hid)
        hid = self.linear2(hid)

        return hid