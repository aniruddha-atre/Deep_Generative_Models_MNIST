import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision.utils import make_grid

from src.models.dcvae import DCVAE, weights_init
from src.data.mnist import get_dcvae_dataloaders
from src.utils.plotting import plot_training_curve
from src.utils.vae_analysis import (
    plot_vae_reconstructions,
    plot_vae_latent_traversal,
    plot_vae_latent_tsne,
    plot_vae_latent_histograms,
)


def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    """
    Returns (total_loss, bce, kld)
    BCE = mean over pixels
    KLD = mean over batch
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = bce + beta * kld
    return loss, bce, kld


def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/samples", exist_ok=True)

    seed = 12
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 64
    latent_dim = 10
    beta = 0.01
    epochs = 25

    train_loader, test_loader = get_dcvae_dataloaders(batch_size=batch_size)

    model = DCVAE(latent_dim=latent_dim).to(device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # per-epoch curves
    train_losses = []
    test_losses = []
    train_bces = []
    train_klds = []

    for epoch in range(1, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        train_loss_total = 0.0
        train_bce_total = 0.0
        train_kld_total = 0.0

        with tqdm(train_loader, unit="batch") as pbar:
            for x, _ in pbar:
                pbar.set_description(f"Epoch {epoch}")

                x = x.to(device)
                optimizer.zero_grad()

                recon, mu, logvar = model(x)
                loss, bce, kld = vae_loss(recon, x, mu, logvar, beta=beta)

                loss.backward()
                optimizer.step()

                train_loss_total += loss.item()
                train_bce_total += bce.item()
                train_kld_total += kld.item()

                pbar.set_postfix(loss=loss.item())

        avg_train = train_loss_total / len(train_loader)
        avg_bce = train_bce_total / len(train_loader)
        avg_kld = train_kld_total / len(train_loader)

        train_losses.append(avg_train)
        train_bces.append(avg_bce)
        train_klds.append(avg_kld)

        print(
            f"===== Epoch {epoch}  "
            f"Train Loss: {avg_train:.4f}  "
            f"BCE: {avg_bce:.4f}  KL: {avg_kld:.4f} ====="
        )

        # -------------------- TEST --------------------
        model.eval()
        test_loss_total = 0.0

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
                test_loss_total += loss.item()

        avg_test = test_loss_total / len(test_loader)
        test_losses.append(avg_test)
        print(f"===== Test Loss: {avg_test:.4f} =====")

        # -------------------- PRIOR SAMPLES --------------------
        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim, device=device)
            samples = model.decode(z).cpu()

            grid = make_grid(samples, nrow=8).permute(1, 2, 0).numpy()

            plt.figure(figsize=(6, 6))
            plt.axis("off")
            plt.title(f"DCVAE Epoch {epoch}")
            plt.imshow(grid, cmap="gray")
            plt.tight_layout()
            plt.savefig(f"outputs/samples/dcvae_epoch_{epoch}.png", dpi=150)
            plt.close()

    # -------------------- SAVE LOSS DATA + CURVES --------------------
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    with open("outputs/logs/dcvae_losses.json", "w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "test_loss": test_losses,
                "train_bce": train_bces,
                "train_kld": train_klds,
            },
            f,
            indent=4,
        )

    # total loss curve
    plot_training_curve(
        {"Train Loss": train_losses, "Test Loss": test_losses},
        title="DCVAE Training Curve",
        out_path="outputs/plots/dcvae_training_curve.png",
    )

    # BCE vs KL curve (train)
    plot_training_curve(
        {"BCE": train_bces, "KL": train_klds},
        title="DCVAE BCE vs KL (Train)",
        out_path="outputs/plots/dcvae_bce_kl_train.png",
    )

    # --------------------  ANALYSIS --------------------
    # 1) Reconstructions (input vs recon) from test set
    plot_vae_reconstructions(
        model,
        test_loader,
        device,
        out_path="outputs/plots/dcvae_reconstructions.png",
    )

    # 2) Latent traversal over z0/z1
    plot_vae_latent_traversal(
        model,
        device,
        latent_dim,
        out_path="outputs/plots/dcvae_latent_traversal.png",
    )

    # 3) t-SNE of latent z
    plot_vae_latent_tsne(
        model,
        test_loader,
        device,
        out_path="outputs/plots/dcvae_latent_tsne.png",
    )

    # 4) Histograms of latent dimensions
    plot_vae_latent_histograms(
        model,
        test_loader,
        device,
        latent_dim,
        out_path="outputs/plots/dcvae_latent_hist.png",
    )

    # Save model
    torch.save(model.state_dict(), "outputs/checkpoints/dcvae.pth")
    print("Training complete! Saved to outputs/checkpoints/dcvae.pth")


if __name__ == "__main__":
    main()