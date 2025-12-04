import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def _ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory != "":
        os.makedirs(directory, exist_ok=True)


def plot_vae_reconstructions(
    model,
    data_loader,
    device,
    out_path="outputs/plots/dcvae_reconstructions.png",
    n=8,
):
    """
    Original (top) vs reconstruction (bottom) for first batch of test set.
    """
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(data_loader))
        x = x.to(device)
        recon, _, _ = model(x)

    x = x[:n].cpu()
    recon = recon[:n].cpu()

    grid = make_grid(torch.cat([x, recon], dim=0), nrow=n).permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.title("DCVAE: Original (top) vs Reconstruction (bottom)")
    plt.imshow(grid, cmap="gray")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {out_path}")


def plot_vae_latent_traversal(
    model,
    device,
    latent_dim,
    out_path="outputs/plots/dcvae_latent_traversal.png",
    steps=11,
    dim_x: int = 0,
    dim_y: int = 1,
):
    """
    2D traversal over two chosen latent dimensions (dim_x, dim_y).
    All other dimensions are fixed to zero.

    This will show meaningful variation *if* those dimensions are used by the VAE.
    """
    model.eval()
    values = torch.linspace(-3.0, 3.0, steps)
    grid_imgs = []

    with torch.no_grad():
        for vy in values:
            z = torch.zeros(steps, latent_dim, device=device)
            z[:, dim_x] = values
            z[:, dim_y] = vy
            samples = model.decode(z).cpu()  # (steps, 1, 28, 28)
            grid_imgs.append(samples)

    grid_imgs = torch.cat(grid_imgs, dim=0)  # (steps*steps, 1, 28, 28)
    grid = make_grid(grid_imgs, nrow=steps).permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"DCVAE Latent Traversal (z{dim_x} vs z{dim_y})")
    plt.imshow(grid, cmap="gray")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {out_path}")


def plot_vae_latent_tsne(
    model,
    data_loader,
    device,
    out_path="outputs/plots/dcvae_latent_tsne.png",
    max_samples=2000,
):
    """
    t-SNE of latent means μ, colored by digit label.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed; skipping t-SNE plot.")
        return

    model.eval()
    mus = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu())
            labels.append(y.cpu())
            if sum(len(lb) for lb in labels) >= max_samples:
                break

    mus = torch.cat(mus, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto")
    z2d = tsne.fit_transform(mus.numpy())

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        z2d[:, 0], z2d[:, 1], c=labels.numpy(), s=5, cmap="tab10", alpha=0.8
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit label")
    plt.title("DCVAE Latent Space (t-SNE of μ)")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {out_path}")


def plot_vae_latent_histograms(
    model,
    data_loader,
    device,
    latent_dim,
    out_path="outputs/plots/dcvae_latent_hist.png",
    max_batches: int = 200,
):
    """
    Histogram of latent μ dimensions over the test set.
    """
    model.eval()
    mus = []

    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu())
            if i >= max_batches:
                break

    mus = torch.cat(mus, dim=0)  # [N, latent_dim]
    mus_np = mus.numpy()

    cols = min(5, latent_dim)
    rows = int((latent_dim + cols - 1) / cols)

    plt.figure(figsize=(3 * cols, 2.5 * rows))
    for i in range(latent_dim):
        plt.subplot(rows, cols, i + 1)
        plt.hist(mus_np[:, i], bins=30, alpha=0.8)
        plt.title(f"z[{i}]")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {out_path}")
