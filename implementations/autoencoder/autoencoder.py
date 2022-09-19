import argparse

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(opt.img_size * opt.img_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, opt.img_size * opt.img_size),
            # nn.Sigmoid(),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(
    autoencoder.parameters(), lr=opt.lr, weight_decay=1e-5
)
criterion = nn.MSELoss()

for epoch in range(opt.num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).to(device)

        output = autoencoder(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch[{}/{}], loss:{:.4f}'
          .format(epoch + 1, opt.num_epochs, loss.item()))

    os.makedirs('../../images', exist_ok=True)
    if epoch % 10 == 0:
        pic = output.cpu().data
        save_image(pic.view(96, 1, 28, 28),
                   '../../images/autoencoder_image_{}.png'.format(epoch))
