import argparse

import os
import torch
from torch import nn
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

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            # convolutional
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # de convolutional
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent_variable = self.encoder(x)
        output = self.decoder(latent_variable)
        return output


convolutionalAE = CAE().to(device)
optimizer = torch.optim.Adam(
    convolutionalAE.parameters(), lr=opt.lr, weight_decay=1e-5
)
criterion = nn.MSELoss()

for epoch in range(opt.num_epochs):
    for data in dataloader:
        img, _ = data
        output = convolutionalAE(img)
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
                   '../../images/convolutionalAE_image_{}.png'.format(epoch))
