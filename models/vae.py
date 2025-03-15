import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable 


@dataclass
class VAEConfig:
    hid_channels: int = 16
    downsample_factors: Iterable[int] = (2, 2)
    latent_channels: int = 1

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out), 
            nn.ReLU(), 
            conv(n_out, n_out), 
            nn.ReLU(), 
            conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self,  config: VAEConfig):
        super().__init__()
        h = config.hid_channels
        latent_channels = config.latent_channels

        self.prenet = nn.Sequential(conv(1, h), ResBlock(h, h))
        self.down_blocks = nn.Sequential(
            *(
                nn.Sequential(conv(h, h, stride=factor, bias=False), ResBlock(h, h)) 
                for factor in config.downsample_factors
            )
        )
        self.conv_mu = conv(h, latent_channels)
        self.conv_logvar = conv(h, latent_channels)
    
    def forward(self, x):
        x = self.prenet(x)
        x = self.down_blocks(x)

        mu = self.conv_mu(x)
        logvar = self.conv_mu(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        h = config.hid_channels 
        latent_channels = config.latent_channels

        self.prenet = nn.Sequential(
            conv(latent_channels, h), nn.ReLU(),
        )

        upsample_factors = config.downsample_factors[::-1]
        self.upsample_blocks = nn.Sequential(
            *(
                nn.Sequential(ResBlock(h, h), nn.Upsample(scale_factor=factor), conv(h, h, bias=False))
                for factor in upsample_factors
            )
        )
        self.head = nn.Sequential(ResBlock(h, h), conv(h, 1))
    
    def forward(self, x):
        x = self.prenet(x)
        x = self.upsample_blocks(x)
        x = self.head(x)

        return x 

class VAE(nn.Module):
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        self.config = config if config is not None else VAEConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
    
    def encode(self, x):
        return self.encoder(x)
    
    def sample(self, mu, logvar):
        z = torch.rand_like(logvar)
        return mu + z * logvar
    
    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.sample(mu, logvar)
        x_rec = self.decode(latent)

        return x_rec, mu, logvar


if __name__ == "__main__":
    vae = VAE()

    from torchinfo import summary
    summary(vae, (1, 28, 28))
