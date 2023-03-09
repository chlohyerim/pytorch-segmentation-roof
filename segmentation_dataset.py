import torch
from torch.utils import data

import PIL

random_seed = 42

# Segmentation을 위한 dataset 클래스 정의
class SegmentationDataset(data.Dataset):
    def __init__(self, inputs:list, targets:list, transform_input=None, transform_target=None):
        self.inputs = inputs
        self.targets = targets
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index:int):
        X = PIL.Image.open('data/train/X/' + self.inputs[index])
        y = PIL.Image.open('data/train/y/' + self.targets[index])

        if self.transform_input is not None:
            torch.manual_seed(random_seed)
            X = self.transform_input(X)

        if self.transform_target is not None:
            torch.manual_seed(random_seed)
            y = self.transform_target(y)

        return X, y