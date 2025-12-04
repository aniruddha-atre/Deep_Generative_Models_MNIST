import torch
import torch.nn as nn


def weights_init(m):
    """Weight initialization."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)


class DCVAE(nn.Module):
    """
    Convolutional VAE for MNIST (1x28x28)
    Uses a stronger decoder + higher latent dimension.
    """

    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 1x28x28 -> 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32x14x14 -> 64x7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), # 64x7x7 -> 128x7x7
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

        # -------------------------
        # Decoder
        # -------------------------
        # Increase decoder width for sharper images
        self.decoder_fc = nn.Linear(latent_dim, 256 * 7 * 7)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # -> 128x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),     # -> 1x28x28
            nn.Sigmoid(),                           # [0,1]
        )

    # -------------------------
    # Forward Functions
    # -------------------------
    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 256, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
