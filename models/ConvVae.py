import torch

from torch import nn
from torch.autograd import Variable


class ImgEncoder(nn.Module):
    def __init__(self,  nc, ndf, latent_variable_size=512, batchnorm=False):
        super(ImgEncoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm

        self.encoder = nn.Sequential(
            # input is 16 x 32 x 32
            nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. 32 x 16 x 16
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 8 x 8
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 4 x 4
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (256) x 2 x 2
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (512) x 2 x 2
        )

        self.bn_mean = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)


    def forward(self, x, **kwargs):
        h = self.encoder(x).reshape(-1, 512)
        if self.batchnorm:
            return self.bn_mean(self.fc1(h)), self.fc2(h)
        else:
            return self.fc1(h), self.fc2(h)


class ImgDecoder(nn.Module):
    def __init__(self,  nc, ngf, latent_variable_size=512):
        super(ImgDecoder, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.latent_variable_size = latent_variable_size

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (512) x 1 x 1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (128) x 8 x 8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (64) x 16 x 16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (32) x 32 x 32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
            # # state size. (16) x 64 x 64
        )

        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, latent_variable_size),
            nn.ReLU(inplace=True),
            )

    def forward(self, z, **kwargs):
        #print(z.shape)
        z = z.reshape(-1, 512)
        h = self.d1(z)
        h = h.reshape(-1, 512, 1, 1)
        return self.decoder(h)


class ImgVAE(nn.Module):
    def __init__(self, nc=16, ngf=16, ndf=16, latent_variable_size=512, imsize=32, batchnorm=False, *args, **kwargs):
        super().__init__()
 
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.imsize = imsize
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm
        self.device = None

        # encoder
        self.encoder = ImgEncoder(nc, ndf, latent_variable_size, batchnorm)
        # decoder
        self.decoder = ImgDecoder(nc, ngf, latent_variable_size)
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mu)
 
    def encode_and_sample(self, x):
        mu, logvar = self.encoder(x.reshape(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decoder(z)
        return res
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.device = device

    def forward(self, x):
        mu, logvar = self.encoder(x.reshape(-1, self.nc, self.imsize, self.imsize))
        mu.to(self.device)
        z = self.reparametrize(mu, logvar)
        res = self.decoder(z)
        print(res.shape)
        return res, z, mu, logvar



class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(nn.Linear(self.in_channels, latent_dims),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(latent_dims),
                                     nn.Linear(latent_dims, latent_dims),
                                     nn.BatchNorm1d(latent_dims),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(latent_dims, latent_dims),
                                     nn.BatchNorm1d(latent_dims),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(latent_dims, latent_dims),
                                     nn.BatchNorm1d(self.latent_dims),
                                     nn.ReLU(inplace=True)
                                     )

        self.fc1 = nn.Linear(latent_dims, latent_dims)
        self.fc2 = nn.Linear(latent_dims, latent_dims)
    
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, out_channels):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.out_channels = out_channels
        self.decoder = nn.Sequential(nn.Linear(self.latent_dims, self.latent_dims),
                             nn.ReLU(inplace=True),
                             nn.BatchNorm1d(self.latent_dims),
                             nn.Linear(self.latent_dims, latent_dims),
                             nn.BatchNorm1d(latent_dims),
                             nn.ReLU(inplace=True),
                             nn.Linear(latent_dims, latent_dims),
                             nn.BatchNorm1d(latent_dims),
                             nn.ReLU(inplace=True),
                             nn.Linear(latent_dims, self.out_channels)
                             )

        
    def forward(self, z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, out_channels):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims)
        self.decoder = Decoder(latent_dims, out_channels)
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if std.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        res = self.decoder(z)
        return res, z, mu, log_var
    


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, out_channels):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims)
        self.decoder = ImgDecoder(latent_dims, out_channels)
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if std.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        res = self.decoder(z)
        return res, z, mu, log_var