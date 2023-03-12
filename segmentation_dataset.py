import torch
from torch.utils import data

import time
import PIL

random_seed = time.time()

# Segmentation을 위한 dataset 클래스 정의
class SegmentationDataset(data.Dataset):
    def __init__(self, images:list, targets:list, transform_image=None, transform_target=None):
        self.images = images
        self.targets = targets
        self.transform_image = transform_image
        self.transform_target = transform_target
        self.images_dtype = torch.float32
        self.targets_dtype = torch.long
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index:int):
        X = PIL.Image.open('data/train/X/' + self.images[index])
        y = PIL.Image.open('data/train/y/' + self.targets[index])

        if self.transform_image is not None:
            torch.manual_seed(random_seed)
            X = self.transform_image(X)

        if self.transform_target is not None:
            torch.manual_seed(random_seed)
            y = self.transform_target(y)

        return X, y