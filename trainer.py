import os

import torch
from torch import optim
from torch import nn

import loader
from models import unet

# hyper-param 설정
learning_rate = 0.001
n_epoch = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.Model().to(device)

state_dict_path = 'checkpoints/state_dict.pt'

if os.path.exists(state_dict_path):
    checkpoint = torch.load(state_dict_path)

    model.load_state_dict(checkpoint['state_dict'])
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    step = 0
    loss_min = None
    state_dict_tosave = None

    # train
    for epoch in range(n_epoch):
        for image, target in loader.train_dataloader:
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            model_output = model(image)

            print(model_output.size(), target.size())

            loss = loss_fn(model_output, target) # 오차를 뭘로 ?

            loss.backward()
            optimizer.step()

            is_tosave = False

            if loss_min == None or loss_min > loss.item():
                is_tosave = True
                loss_min = loss.item()
                state_dict_tosave = model.state_dict()

            if (step + 1) % 1000 == 0:
                print("Step: {}\tLoss: {}\tUpdating model to save: {}".format(step + 1, loss.item(), is_tosave))

            step += 1

    torch.save({'state_dict': state_dict_tosave}, state_dict_path)