import os

from torch.utils.data import DataLoader

from segmentation_dataset import SegmentationDataset
import aug

train_X_list = os.listdir('data/train/X')
train_y_list = os.listdir('data/train/y')

train_X_list.sort()
train_y_list.sort()

train_dataset = SegmentationDataset(
    images=train_X_list,
    targets=train_y_list,
    transform_image=aug.transform_image,
    transform_target=aug.transform_target
)

# hyper-param 설정
train_batch_size = 4

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    shuffle=True
)

train_images, train_targets = next(iter(train_dataloader))