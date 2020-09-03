from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


img_size = 84
n_path = '../data/FetchPush/vae_model_pick'


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(img_size * img_size * 3, 400)
        # Try to reduce
        self.fc21 = nn.Linear(400, 3)
        self.fc22 = nn.Linear(400, 3)
        self.fc3 = nn.Linear(3, 400)
        self.fc4 = nn.Linear(400, img_size * img_size * 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std) * 0.126
        #return mu

    # maybe z * 11
    def decode(self, z):
        h3 = F.relu(self.fc3(z / 0.126))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, img_size * img_size * 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def format(self, rgb_array):
        data = torch.from_numpy(rgb_array).float().to(device='cuda')
        data /= 255
        data = data.permute([2, 0, 1])
        data = data.reshape([-1, 3, img_size, img_size])
        return data.reshape(-1, img_size * img_size * 3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.reshape(-1, img_size * img_size * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # Try to adjust
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# torch.Size([128, 1, img_size, img_size])
def train(epoch, model, optimizer, device, log_interval, batch_size):
    model.train()
    train_loss = 0
    data_set = np.load('../data/FetchPush/vae_train_data_pick.npy')
    data_size = len(data_set)
    data_set = np.split(data_set, data_size / batch_size)

    for batch_idx, data in enumerate(data_set):
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        data = data.reshape([-1, 3, img_size, img_size])
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            save_image(data.cpu().view(-1, 3, img_size, img_size),
                       'results/original.png')
            save_image(recon_batch.cpu().view(-1, 3, img_size, img_size),
                       'results/recon.png')
            #           'results/recon_' + str(epoch) + '.png')

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), data_size,
                100. * (batch_idx+1) / len(data_set),
                loss.item() / len(data)))
            print('Loss: ', loss.item() / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / data_size))


def train_Vae(batch_size=128, epochs=100, no_cuda=False, seed=1, log_interval=9, load=False):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint = torch.load(n_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, device, log_interval, batch_size)
        # test(epoch, model, test_loader, batch_size, device)
        # with torch.no_grad():
        #    sample = torch.randn(64, 5).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 3, img_size, img_size),
        #               'results/sample.png')
        if not (epoch % 100):
            print('Saving Progress!')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, n_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, n_path)


def load_Vae(path, no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model


if __name__ == '__main__':
    # Train VAE
    print('Train VAE...')
    train_Vae(batch_size=128, epochs=50, load=True)
    print('Successfully trained VAE')
