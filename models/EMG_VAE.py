import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims, variational=True):
        super(VariationalEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dims = latent_dims

        self.variational = variational

        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, latent_dims),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(latent_dims),
            nn.Dropout(0.6),

            ###################################################
            # nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.5),
            # nn.MaxPool2d(kernel_size=3, stride=3),
            
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # # nn.Dropout(p=0.2),
            # nn.Flatten(),
        )

        self.fc1 = nn.Linear(latent_dims, latent_dims)
        self.fc2 = nn.Linear(latent_dims, latent_dims)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.fc1(h), self.fc2(h)
        else:
            return h
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, out_channels):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.out_channels = out_channels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 16384),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16384),
            nn.Dropout(0.6),
            nn.Unflatten(1, (16, 32, 32)),
            # nn.Unflatten(1, (latent_dims, 1, 1)),
            # nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.5),
            
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.5),
            
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.5),
            
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.5),
            # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(0.5)
        )

        
    def forward(self, z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, out_channels, variational=True):
        super(VariationalAutoencoder, self).__init__()
        self.variational = variational
        self.encoder = VariationalEncoder(in_channels, latent_dims, variational=self.variational)
        self.decoder = Decoder(latent_dims, out_channels)
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        if self.variational:
            mu, log_var = self.encoder(x)
            z = self.reparametrize(mu, log_var)
            res = self.decoder(z)
            return res, z, mu, log_var
        else:
            z = self.encoder(x)
            res = self.decoder(z)
            return res, z