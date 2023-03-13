import torch
from torch import optim
from torch import nn

from models import unet

from torch.utils.data import DataLoader

from segmentation_dataset import SegmentationDataset
import aug

def train(
    model,
    device,
    n_epoch=10,
    batch_size=1,
    learning_rate=1e-5
):
    train_dataset = SegmentationDataset(
        X_dir='data/train/X',
        y_dir='data/train/y',
        transform_X=aug.transform_X,
        transform_y=aug.transform_y
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )

    criterion = nn.BCEWithLogitsLoss()

    global_step = 0

    model.train()
    
    for epoch in range(1, n_epoch + 1):
        for batch in train_dataloader:
            X = batch['X'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            y = batch['y'].to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            pred = model(X)
            loss = criterion(pred.squeeze(dim=1), y.squeeze(dim=1).float())

            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()

            global_step += 1
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = unet.Model().to(device)

train(model=model, device=device)