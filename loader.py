import os

from torch.utils.data import DataLoader

from segmentation_dataset import SegmentationDataset
import aug

train_X_list = os.listdir('data/train/X')
train_y_list = os.listdir('data/train/y')

train_X_list.sort()
train_y_list.sort()

train_dataset = SegmentationDataset(
    inputs=train_X_list,
    targets=train_y_list,
    transform_input=aug.trans_input,
    transform_target=aug.trans_target
)

# hyper-param 설정
_batch_size = 4
learning_rate = 0.001
n_epoch = 30

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=_batch_size,
    shuffle=True
)