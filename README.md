# 📘 Deep Generative Models From Scratch — DCGAN + DCVAE on MNIST


A PyTorch implementation of DCGAN and Deep Convolutional VAE, explores generative modelling, training curves and stability, latent space analysis and visualization techniques.


## 🚀 Models

## 🔷 DCGAN (Deep Convolutional Generative Adversarial Network)


<img width="750" height="450" alt="image" src="https://github.com/user-attachments/assets/9c3e0c8a-aa00-4227-88bb-18ccf1d33323" />


### a) Generator (G)

Purpose: Transform random Gaussian noise into realistic MNIST digits. The generator learns “how to draw digits” by fooling the discriminator

Input: 100-dimensional noise vector

Output: 1×28×28 image (MNIST digit)

We use ConvTranspose2D for Upsampling, BatchNorm for training stability and tanh activation function

### b) Discriminator (D)

Purpose: Classify images as real or fake. The discriminator learns “what digits look like” by detecting artifacts in generator outputs.

Input: Generated MNIST image

Output: Real/Fake score (BCEWithLogits)

We use CNN for Downsampling and LeakyReLU activation function


### c) Adversarial Training Loop

The Generator tries to create images that look real and the Discriminator learns to distinguish real vs fake images. The models compete like a min-max game, eventually improving each other.

This adversarial setup produces high-quality sharp samples, but GANs do not learn an interpretable latent space.


## 🔶 DCVAE (Deep Convolutional Variational Autoencoder)


<img width="750" height="750" alt="image" src="https://github.com/user-attachments/assets/07da5a92-667a-4ae7-95f3-83e7110fd18d" />


A Variational Autoencoder learns how to encode an image into a smooth latent space and how to decode a latent vector back into an image. It models the data distribution using probabilistic encoding.

### a) Encoder

Three convolutional layers

Flatten → Linear layers output μ (mean) and logσ² (log variance)

Represents each image as a probability distribution over latent space

Latent dimension = 20

Reparameterization Trick: To enable backpropagation through random sampling:

z = μ + σ * ε,   ε ~ N(0, I)

### b) Decoder

Linear layer expanding latent vector

Transposed-Convolution layers

Output: 1×28×28 MNIST-style reconstruction

Activation: Sigmoid (since MNIST is grayscale 0–1)

### c) Loss Function = Reconstruction + KL Divergence

The VAE optimizes:

Loss = BCE(Image, Reconstruction) + β * KL(q(z|x) || N(0,I))

BCE encourages accurate reconstruction

KL forces the latent space to follow a Gaussian distribution

β controls disentanglement (we use β = 0.1)

### d) Interpretability Advantage

Unlike GANs, VAEs produce:

Smooth latent spaces

Continuous interpolations

Meaningful structure in latent dimensions

Cluster separation (visible in t-SNE plots)


## DCGAN Sample vs DCVAE Sample

<img width="450" height="450" alt="dcgan_epoch_25" src="https://github.com/user-attachments/assets/71011901-1c7c-4037-9b74-58e6a14fa172" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="450" height="450" alt="dcvae_epoch_25" src="https://github.com/user-attachments/assets/42cf958a-4db4-4d8b-8791-28b0129639a2" />




## 📊 Results & Visualizations


## 📈 DCGAN Training Curves

<img width="750" height="750" alt="dcgan_training_curve" src="https://github.com/user-attachments/assets/32e323ea-e8a8-42d5-ac48-b076cbbf7ca3" />


Shows the Generator and Discriminator losses over epochs.

Discriminator loss decreasing → D learns to distinguish real/fake

Generator loss stabilizing → G learns to fool D consistently

Balanced curves indicate healthy adversarial training (no mode collapse)


## 📈 DCVAE Training Curves

<img width="750" height="750" alt="dcvae_training_curve" src="https://github.com/user-attachments/assets/2eff9bc0-c94c-4876-96ee-ff75e461449b" />
<img width="750" height="750" alt="dcvae_bce_kl_train" src="https://github.com/user-attachments/assets/fe921637-7cc8-4a9d-8a51-afebe33eabac" />


Plots the total VAE loss (BCE + KL) across epochs.

Overall decreasing trend → network properly converging

BCE reduces → reconstructions improve

KL term stabilizes → latent distribution approaches N(0, 1)

## 🔍 DCVAE Latent Space

<img width="750" height="750" alt="dcvae_latent_tsne" src="https://github.com/user-attachments/assets/80015bd7-e7ac-4dbb-ad89-f694915f62dd" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="1050" height="450" alt="dcvae_latent_hist" src="https://github.com/user-attachments/assets/c75b706c-f43c-43f6-b6df-1d09728d357b" />


  Latent traversals showing interpretable dimensions
  
  Histograms of learned latents approximate a standard Gaussian, which confirms KL regularization is working (Skewed or collapsed distributions would indicate training issues)
  
  t-SNE showing digit clustering indicates that the encoder organizes latent space semantically (Digits with similar structure overlap (e.g., 3 & 5))


## 📦 Installation

git clone https://github.com/aniruddha-atre/Deep_Generative_Models_MNIST.git

pip install -r requirements.txt

How to Run Training

Train DCGAN:

    python -m src.training.train_dcgan

Train DCVAE:

    python -m src.training.train_dcvae

Plots and samples will be auto-generated.
