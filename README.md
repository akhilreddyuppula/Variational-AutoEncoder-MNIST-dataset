# Variational-AutoEncoder-MNIST-dataset
PyTorch VAE trained on MNIST — encoder learns latent distributions (μ, σ), decoder reconstructs and generates new digit variations using BCE + KL divergence loss.
vae.py        ← VAE model (encoder + decoder)
vaetrain.py   ← training loop + inference

How it works:

Takes a handwritten digit image
Compresses it into a small latent representation (mean + variance)
Reconstructs the original image from that representation
Can generate new digit variations by sampling from the learned distribution

A regular autoencoder compresses to a fixed point. A VAE compresses to a distribution, which means you can sample from it and generate new images that look like the original digit but aren't identical.

Reparameterization trick — makes backpropagation work through random sampling:
z = mean + sigma × random_noise
Loss function — two parts working together:
Total loss = Reconstruction loss (BCE) + KL divergence
