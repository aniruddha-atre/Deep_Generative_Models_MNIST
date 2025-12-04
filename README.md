📘 Deep Generative Models From Scratch — DCGAN + DCVAE on MNIST

A PyTorch implementation of DCGAN and Deep Convolutional VAE, explores generative modelling, training curves and stability, latent space analysis and visualization techniques.

🚀 Models

🔷 DCGAN (Deep Convolutional Generative Adversarial Network)

a) Generator (G)

Purpose: Transform random Gaussian noise into realistic MNIST digits. The generator learns “how to draw digits” by fooling the discriminator

Input: 100-dimensional noise vector

Upsampling using ConvTranspose2D

BatchNorm for training stability

Final output is 1×28×28 image (MNIST)

Activation: tanh

b) Discriminator (D)

Purpose: Classify images as real or fake. The discriminator learns “what digits look like” by detecting artifacts in generator outputs.

Input: MNIST or generated image

Downsampling CNN

LeakyReLU activation

No BatchNorm in the first layer

Output: Real/Fake score (BCEWithLogits)

c) Adversarial Training Loop

Generator tries to create images that look real.

Discriminator learns to distinguish real vs fake.

The models compete, eventually improving each other.

This adversarial setup produces high-quality sharp samples, but GANs do not learn an interpretable latent space.


🔶 DCVAE (Deep Convolutional Variational Autoencoder)

A Variational Autoencoder learns:

How to encode an image into a smooth latent space,

How to decode a latent vector back into an image.

It models the data distribution using probabilistic encoding.

a) Encoder

Three convolutional layers

Flatten → Linear layers output μ (mean) and logσ² (log variance)

Represents each image as a probability distribution over latent space

Latent dimension = 20

Reparameterization Trick: To enable backpropagation through random sampling:

z = μ + σ * ε,   ε ~ N(0, I)

b) Decoder

Linear layer expanding latent vector

Transposed-Convolution layers

Output: 1×28×28 MNIST-style reconstruction

Activation: Sigmoid (since MNIST is grayscale 0–1)

c) Loss Function = Reconstruction + KL Divergence

The VAE optimizes:

Loss = BCE(Image, Reconstruction) + β * KL(q(z|x) || N(0,I))

BCE encourages accurate reconstruction

KL forces the latent space to follow a Gaussian distribution

β controls disentanglement (we use β = 0.1)

d) Interpretability Advantage

Unlike GANs, VAEs produce:

Smooth latent spaces

Continuous interpolations

Meaningful structure in latent dimensions

Cluster separation (visible in t-SNE plots)


📊 Results & Visualizations


📈 Training Curves (DCGAN)

<img width="1050" height="750" alt="dcgan_training_curve" src="https://github.com/user-attachments/assets/32e323ea-e8a8-42d5-ac48-b076cbbf7ca3" />


Shows the Generator and Discriminator losses over epochs.

Discriminator loss decreasing → D learns to distinguish real/fake

Generator loss stabilizing → G learns to fool D consistently

Balanced curves indicate healthy adversarial training (no mode collapse)


📈 Training Curves (DCVAE)

<img width="1050" height="750" alt="dcvae_training_curve" src="https://github.com/user-attachments/assets/2eff9bc0-c94c-4876-96ee-ff75e461449b" />

<img width="1050" height="750" alt="dcvae_bce_kl_train" src="https://github.com/user-attachments/assets/fe921637-7cc8-4a9d-8a51-afebe33eabac" />


Plots the total VAE loss (BCE + KL) across epochs.

Overall decreasing trend → network properly converging

BCE reduces → reconstructions improve

KL term stabilizes → latent distribution approaches N(0, 1)

🔍 Latent Space (DCVAE)

<img width="1050" height="750" alt="dcvae_latent_tsne" src="https://github.com/user-attachments/assets/80015bd7-e7ac-4dbb-ad89-f694915f62dd" />

<img width="2250" height="750" alt="dcvae_latent_hist" src="https://github.com/user-attachments/assets/c75b706c-f43c-43f6-b6df-1d09728d357b" />


Latent traversals showing interpretable dimensions

Histograms of learned latents approximate a standard Gaussian, which confirms KL regularization is working (Skewed or collapsed distributions would indicate training issues)

t-SNE showing digit clustering indicates that the encoder organizes latent space semantically (Digits with similar structure overlap (e.g., 3 & 5))


📦 Installation

git clone https://github.com/aniruddha-atre/Deep_Generative_Models_MNIST.git

pip install -r requirements.txt

How to Run Training

Train DCGAN:

python -m src.training.train_dcgan

Train DCVAE:

python -m src.training.train_dcvae

Plots and samples will be auto-generated.
