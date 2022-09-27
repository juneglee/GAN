import argparse

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
opt = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = DataLoader(
    MNIST('../../../data', transform=transform),
    batch_size=opt.batch_size,
    shuffle=True
)

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(
    autoencoder.parameters(), lr=opt.lr, weight_decay=1e-5
)
criterion = nn.MSELoss()


for epoch in range(opt.num_epochs):
    for data in dataloader:
        img, _ = data
        noise_x = add_noise(img)
        noise_x = noise_x.view(-1, 28*28).to(device)
        img = img.view(-1, 28*28).to(device)

        encoded, decoded = autoencoder(noise_x)
        loss = criterion(decoded, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch[{}/{}], loss:{:.4f}'
          .format(epoch + 1, opt.num_epochs, loss.item()))
