import torch
import torch.nn as nn


def mu(x):
    return torch.mean(x, dim=[2, 3], keepdim=True)

def sigma(x):
    return torch.sqrt(torch.mean((x - mu(x)) ** 2, dim=[2, 3], keepdim=True) + 0.000000023)

def adain(lc, ls):
    # lc: content , ls: style
    return sigma(ls) * (lc-mu(lc)) / sigma(lc) + mu(ls)


class Encoder(nn.Module):
    def __init__(self, cin, image_size=224):
        super().__init__()
        conv = lambda _cin, _cout: nn.Sequential(
            nn.Conv2d(_cin, _cout, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        cs = [cin, 32, 64, 128, 256, 512]
        self.network = nn.Sequential(*[
            conv(cs[i], cs[i+1]) for i in range(len(cs)-1)
        ])

    def forward(self, x):
        return self.network(x)      # (n, 128, 7, 7)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        conv = lambda _cin, _cout: nn.Sequential(
            nn.ConvTranspose2d(_cin, _cout, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        cs = [512, 256, 128, 64, 32, 32]
        self.network = nn.Sequential(*[
            conv(cs[i], cs[i+1]) for i in range(len(cs)-1)
        ])

    def forward(self, x):
        return self.network(x)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # For encoder load pretrained VGG19 model and remove layers upto relu4_1
        # resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        # self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder = Encoder(3, 224)
        output_size = 512 * 7 * 7
        self.fc_mu = nn.Linear(output_size, latent_dim)
        self.fc_var = nn.Linear(output_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, output_size)
        self.decoder = Decoder()
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, input: torch.Tensor):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z).view(z.size(0), -1, 7, 7)
        x = self.decoder(x)
        x = self.final(x)
        return x

    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar, z
