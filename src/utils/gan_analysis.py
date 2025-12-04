import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def _ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory != "":
        os.makedirs(directory, exist_ok=True)


def plot_dcgan_latent_interpolations(
    netG,
    nz: int,
    device,
    rows: int = 5,
    steps: int = 10,
    out_path: str = "outputs/plots/dcgan_latent_interpolations.png",
):
    """
    Interpolates in DCGAN latent space.

    Each row: interpolation from z_start -> z_end.
    To guarantee strong variation, we choose z_end = -z_start
    (opposite direction in latent space).
    """

    netG.eval()
    all_imgs = []

    with torch.no_grad():
        for _ in range(rows):
            # one random starting point
            z_start = torch.randn(1, nz, 1, 1, device=device)
            # opposite point for strong visual change
            z_end = -z_start

            alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1, 1)
            z_row = (1 - alphas) * z_start + alphas * z_end  # (steps, nz, 1, 1)

            fake_row = netG(z_row)                  # (steps, 1, 28, 28)
            fake_row = (fake_row + 1.0) / 2.0       # [-1,1] -> [0,1]
            all_imgs.append(fake_row)

        all_imgs = torch.cat(all_imgs, dim=0)       # (rows*steps, 1, 28, 28)

        grid = make_grid(all_imgs, nrow=steps).permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.title("DCGAN Latent Interpolations")
        plt.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
        plt.tight_layout()

        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Plot saved] {out_path}")
