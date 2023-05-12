import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

scaler = StandardScaler()
train_data.data = scaler.fit_transform(train_data.data.reshape(-1, 784)).reshape(-1, 28, 28)

# Create the data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)


class Encoder(nn.Module):
    def init(self, latent_size=20):
        super(Encoder, self).init()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar


class Decoder(nn.Module):
    def init(self, latent_size=20):
        super(Decoder, self).init()
        self.fc1 = nn.Linear(latent_size, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = z.view(-1, 1, 28, 28)
        return z


class VAE(nn.Module):
    def init(self, latent_size=20):
        super(VAE, self).init()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD