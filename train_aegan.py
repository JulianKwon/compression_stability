import torch
from torch.autograd import Variable

from models import model_loader
from utils import load_checkpoint


def get_classifier(base_path, ratio):
    net = model_loader('resnet34')
    state = load_checkpoint(base_path, 'cifar10', 'resnet34', True,
                            filename='benign_train_%d' % int(100 * ratio))
    net.load_state_dict(state['state_dict'])
    return net


def main(epoch, EPOCH, loader, gan, discrim, criterion, optimizer, device):
    gan.train()

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        batch_size = X.size(0)

        z_x, x_rec = gan(X)
        z_p = Variable(torch.randn(batch_size, 128)).to(device)
        x_p_tilda = gan.decoder(z_p)

        orig_l = discrim(X)
        recon_l = discrim(x_rec)
        p_l = discrim(x_p_tilda)
        err_real = criterion(orig_l, recon_l)
        err_noise = criterion(orig_l, p_l)

        gan_loss = err_real + err_noise

        rec_loss = ((orig_l - recon_l) ** 2).mean()
        mse_loss = ((X - x_rec) ** 2).mean()

        loss = rec_loss + mse_loss + gan_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_rec: %.4f\tLoss_mse: %.4f\t'
                  % (epoch, EPOCH, i, len(loader), gan_loss.item(), rec_loss.item(), mse_loss.item()))
