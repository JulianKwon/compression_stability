import torch
import torch.nn as nn
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    def __init__(self, latent_dim, patch_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)  # in_channels=3
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256 * patch_dim * patch_dim, 2048)
        self.bn4 = nn.BatchNorm1d(2048, momentum=0.9)
        self.fc2 = nn.Linear(2048, latent_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = out.view(batch_size, -1)
        out = self.relu(self.bn4(self.fc1(out)))
        z = self.fc2(out)

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, patch_dim):
        super(Decoder, self).__init__()
        self.patch_dim = patch_dim
        self.fc1 = nn.Linear(latent_dim, patch_dim * patch_dim * 256)
        self.bn1 = nn.BatchNorm1d(patch_dim * patch_dim * 256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(128, 32, 6, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, self.patch_dim, self.patch_dim)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(32, 128, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = nn.Conv2d(256, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)
        self.fc1 = nn.Linear(8 * 8 * 256, 512)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x1 = x;
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x, x1


class AEGAN(nn.Module):
    def __init__(self, latent_dim=128, patch_dim=8):
        super(AEGAN, self).__init__()
        self.encoder = Encoder(latent_dim, patch_dim)
        self.decoder = Decoder(latent_dim, patch_dim)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, x):
        bs = x.size()[0]
        z = self.encoder(x)
        x_tilda = self.decoder(z)

        return z, x_tilda
