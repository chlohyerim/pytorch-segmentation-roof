import torch
from torch.utils import data

from pathlib import Path
from os import listdir
from os.path import splitext
import time
import PIL

# Segmentation을 위한 dataset 클래스 정의
class SegmentationDataset(data.Dataset):
    def __init__(self, X_dir, y_dir, transform_X, transform_y):
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.transform_X = transform_X
        self.transform_y = transform_y

        self.names = [splitext(file)[0] for file in listdir(X_dir)]
        self.random_seed = time.time()

    def __len__(self):
        return len(self.names)
    
    def openimage(self, directory, name):
        return PIL.Image.open(directory + '/' + name + '.png')
    
    def __getitem__(self, index):
        name = self.names[index]
        X = self.openimage(self.X_dir, name)
        y = self.openimage(self.y_dir, name)

        torch.manual_seed(self.random_seed)

        return {
            'X': torch.as_tensor(self.transform_X(X)).float(),
            'y': torch.as_tensor(self.transform_y(y)).long()
        }