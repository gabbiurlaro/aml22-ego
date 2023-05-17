import torch

from torch import nn
from torch.autograd import Variable


class ImgEncoder(nn.Module):
    def __init__(self,  in_channels, out_channels, latent_variable_size=512, batchnorm=False):
        super(ImgEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm

        self.encoder = nn.Sequential(
            # input is 16 x 32 x 32
            nn.Conv2d(in_channels, out_channels*3/4, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. 32 x 16 x 16
            nn.Conv2d(out_channels*3/4, latent_variable_size, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_variable_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GlobalAvgPool2d(latent_variable_size),
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
    def __init__(self,  in_channels, out_channels, latent_variable_size=512):
        super(ImgDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_variable_size = latent_variable_size

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (512) x 1 x 1
            nn.ConvTranspose2d(latent_variable_size, out_channels*3/4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*3/4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 4 x 4
            nn.ConvTranspose2d(out_channels*3/4, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            

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
    def __init__(self, in_channels=1024, latent_variable_size=512,  out_channels=16, *args):
        super().__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_variable_size = latent_variable_size
        self.device = None

        # encoder
        self.encoder = ImgEncoder(self.in_channels, self.out_channels, latent_variable_size)
        # decoder
        self.decoder = ImgDecoder(self.in_channels, self.out_channels,latent_variable_size)
        
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
        mu, logvar = self.encoder(x)
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
        mu, logvar = self.encoder(x)
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
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.2)
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
    
