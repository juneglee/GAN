import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import logging
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", default=64)
parser.add_argument("--lr", default=0.0002)
parser.add_argument("--b1", default=0.5)
parser.add_argument("--b2", default=0.999)
parser.add_argument("--worker", default=8)
parser.add_argument("--img_size", default=32)
parser.add_argument("--channels", default=1)
parser.add_argument("--n_classes", default=10)
parser.add_argument("--latent_dim", default=100)
parser.add_argument("--sample_interval", default=400)
parser.add_argument("--version", default="v0")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

logger = logging.getLogger()  # 로그 생성
logging.basicConfig(filename=f'../../images/{opt.version}/acgan_{opt.version}.log', level=logging.INFO)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# device
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# dataloader
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

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    sample_path = f"../../images/{opt.version}"
    os.makedirs(sample_path, exist_ok=True)
    save_image(gen_imgs.data, f"../../images/{opt.version}/%d.png" % batches_done, nrow=n_row, normalize=True)
    # print(labels)


G_losses = []
D_losses = []

print("Starting Training Loop...")
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, imgs.shape[0])))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)

        g_loss = 0.5 * (adversarial_loss(validity, valid)
                        + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real image
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
        # Loss for fake image
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
        # total loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]  [D loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item(), d_loss.item())
        )
        logger.info("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]  [D loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item(), d_loss.item()))

        # os.makedirs('./images', exist_ok = True)
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # save_image(gen_imgs.data[:25], f"images/ac_%d.png" % batches_done, nrow=5, normalize=True)
            sample_image(n_row=10, batches_done=batches_done)

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

    model_dir = "../../data"
    os.makedirs(model_dir, exist_ok=True)

    if (epoch + 1) % 100 == 0:
        torch.save({
            'num_epoch': opt.n_epochs,
            'model_G_state_dict': generator.state_dict(),
            'model_D_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, os.path.join(model_dir, f'acgan_implementation_epoch_%d_{opt.version}.pt' % (epoch + 1)))
        print(f'GAN Model checkpoint epoch {epoch + 1} saved .. ')