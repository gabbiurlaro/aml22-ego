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
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            
            nn.Conv2d(256, latent_dims, kernel_size=4, stride=2, padding=1),
            
            nn.Dropout(p=0.8),
            nn.Flatten(0),
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
            nn.Unflatten(0, (latent_dims, 1, 1)),
            nn.ConvTranspose2d(latent_dims, 256, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            
            nn.Dropout(p=0.8)
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