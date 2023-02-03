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
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
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
    def __init__(self, nc=16, ngf=32, ndf=32, latent_variable_size=512, imsize=32, batchnorm=False, *args, **kwargs):
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
        return res, z, mu, logvar
