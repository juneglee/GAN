import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--feature", type=int, default=128, help="dimensionality of layer")
parser.add_argument("--channels_noise", type=int, default=100, help="dimensionality of noise")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataloader = DataLoader(
    MNIST("../../../data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(opt.channels_noise, opt.feature * 8, 3, stride=1, padding=0),
            nn.BatchNorm2d(opt.feature * 8),
            nn.ReLU(),
        )
        self.conv_blocks = nn.Sequential(
            *block(opt.feature * 8, opt.feature * 4, 3, 2, 0),
            *block(opt.feature * 4, opt.feature * 2, 4, 2, 1),
            nn.ConvTranspose2d(opt.feature * 2, opt.channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.l1(x)
        img = self.conv_blocks(x)
        return img

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    def block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    self.l1 = nn.Sequential(
        nn.Conv2d(opt.channels, opt.feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(opt.feature * 2),
        nn.ReLU(),
    )
    self.conv_blocks = nn.Sequential(
        *block(opt.feature * 2, opt.feature * 4, 4, 2, 1),
        *block(opt.feature * 4, opt.feature * 8, 3, 2, 0),
        nn.Conv2d(opt.feature * 8, opt.channels, kernel_size=3, stride=1, padding=0, bias=False),
        nn.Sigmoid()
    )
    self.apply(weights_init)

  def forward(self, x):
      x = self.l1(x)
      val = self.conv_blocks(x)
      return val.view(val.shape[0], -1)

# Initialize generator and discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss function
criterion = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------
print('Starting Training Loop .. ')
for epoch in range(opt.num_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False) # 1
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)  # 0

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        noise = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.channels_noise))))

        # Generate a batch of images
        gen_imgs = netG(noise)

        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(netD(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion(netD(real_imgs), valid)
        fake_loss = criterion(netD(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "../../images/dcgan_%d.png" % batches_done, nrow=5, normalize=True)