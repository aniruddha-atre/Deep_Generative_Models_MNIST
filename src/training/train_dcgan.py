import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision.utils import make_grid

from src.models.dcgan import DCGANGenerator, DCGANDiscriminator, weights_init_normal
from src.utils.gan_analysis import plot_dcgan_latent_interpolations
from src.data.mnist import get_dcgan_dataloader
from src.utils.plotting import plot_training_curve


def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/samples", exist_ok=True)

    seed = 12
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    nz = 100
    ngf = 64
    ndf = 64
    batch_size = 64
    lr = 2e-4
    beta1 = 0.5
    epochs = 25

    dataloader = get_dcgan_dataloader(batch_size=batch_size)

    netG = DCGANGenerator(nz=nz, ngf=ngf).to(device)
    netD = DCGANDiscriminator(ndf=ndf).to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    criterion = nn.BCEWithLogitsLoss()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # per-batch (still logged, optional)
    G_losses = []
    D_losses = []

    # per-epoch (for clean curves)
    G_epoch_losses = []
    D_epoch_losses = []

    for epoch in range(1, epochs + 1):
        epoch_G_sum = 0.0
        epoch_D_sum = 0.0
        num_batches = 0

        with tqdm(dataloader, unit="batch") as epoch_pbar:
            for real_img, _ in epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}")

                real_img = real_img.to(device)
                b_size = real_img.size(0)
                num_batches += 1

                # labels
                real_label = torch.full((b_size, 1), 0.9, device=device)
                fake_label = torch.zeros((b_size, 1), device=device)

                # 1) Update D
                netD.zero_grad()

                output_real = netD(real_img)
                errD_real = criterion(output_real, real_label)
                errD_real.backward()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_img = netG(noise)
                output_fake = netD(fake_img.detach())
                errD_fake = criterion(output_fake, fake_label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                optimizerD.step()

                # 2) Update G
                netG.zero_grad()
                label_for_G = torch.ones((b_size, 1), device=device)
                output_fake_for_G = netD(fake_img)
                errG = criterion(output_fake_for_G, label_for_G)
                errG.backward()
                optimizerG.step()

                epoch_pbar.set_postfix(
                    OrderedDict(Dis_loss=errD.item(), Gen_loss=errG.item())
                )

                # # per-batch logs
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())

                # accumulate for this epoch
                epoch_G_sum += errG.item()
                epoch_D_sum += errD.item()

        # epoch averages
        G_epoch_losses.append(epoch_G_sum / num_batches)
        D_epoch_losses.append(epoch_D_sum / num_batches)

        # Save sample grid for this epoch
        with torch.no_grad():
            fake = netG(fixed_noise).cpu()
            fake = (fake + 1.0) / 2.0  # [-1,1] -> [0,1]
            grid = make_grid(fake, nrow=8).permute(1, 2, 0).numpy()

            plt.figure(figsize=(6, 6))
            plt.axis("off")
            plt.title(f"DCGAN Epoch {epoch}")
            plt.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
            plt.tight_layout()
            plt.savefig(f"outputs/samples/dcgan_epoch_{epoch}.png", dpi=150)
            plt.close()

    # ---- save logs ----
    os.makedirs("outputs/logs", exist_ok=True)
    with open("outputs/logs/dcgan_losses.json", "w") as f:
        json.dump(
            {
                # "G_loss": G_losses,                 # per-batch
                # "D_loss": D_losses,
                "G_epoch_loss": G_epoch_losses,     # per-epoch
                "D_epoch_loss": D_epoch_losses,
            },
            f,
            indent=4,
        )

    # ---- plot epoch curve immediately ----
    plot_training_curve(
        {
            "Generator Loss": G_epoch_losses,
            "Discriminator Loss": D_epoch_losses,
        },
        title="DCGAN Training Curve",
        out_path="outputs/plots/dcgan_training_curve.png",
    )

    # Latent interpolations to visualize DCGAN latent space
    plot_dcgan_latent_interpolations(
        netG,
        nz=nz,
        device=device,
        rows=5,
        steps=10,
        out_path="outputs/plots/dcgan_latent_interpolations.png",
    )

    # ---- save models ----
    torch.save(netG.state_dict(), "outputs/checkpoints/dcgan_gen.pth")
    torch.save(netD.state_dict(), "outputs/checkpoints/dcgan_disc.pth")
    print("Training complete! DCGAN models saved.")


if __name__ == "__main__":
    main()
