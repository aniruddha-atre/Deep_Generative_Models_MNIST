import torch
import torch.nn as nn


def weights_init_normal(m):
    """
    DCGAN-style weight initialization.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    Latent vector (N, nz, 1, 1) -> output (N, 1, 28, 28).
    """

    def __init__(self, nz: int = 100, ngf: int = 64):
        super().__init__()
        self.nz = nz
        self.ngf = ngf

        self.main = nn.Sequential(
            # Z -> (ngf*4) x 7 x 7
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4,7,7) -> (ngf*2,14,14)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2,14,14) -> (ngf,28,28)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf,28,28) -> (1,28,28)
            nn.ConvTranspose2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh(),  # [-1,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    Input: (N, 1, 28, 28)
    Output: logits (N, 1) for BCEWithLogitsLoss.
    """

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.ndf = ndf

        self.main = nn.Sequential(
            # 1x28x28 -> ndf x 14 x 14
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 14 x 14 -> (ndf*2) x 7 x 7
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 7 x 7 -> 1 x 1 x 1
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        return out.view(-1, 1)
