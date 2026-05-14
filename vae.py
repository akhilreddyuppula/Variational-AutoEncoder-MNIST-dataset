import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim):
        super().__init__()
        self.input2_hid = nn.Linear(input_dim, hid_dim)
        self.hid2_mu = nn.Linear(hid_dim, z_dim)
        self.hid2_sigma = nn.Linear(hid_dim, z_dim)

        self.z2_hidden = nn.Linear(z_dim, hid_dim)
        self.hid2_input = nn.Linear(hid_dim, input_dim)
        self.relu = nn.ReLU()


    def encode(self,x):
        h = self.relu(self.input2_hid(x))
        mu, sigma = self.hid2_mu(h), self.hid2_sigma(h)
        return mu, sigma
    def decode(self, z):
        h = self.relu(self.z2_hidden(z))
        return torch.sigmoid(self.hid2_input(h))

        
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(mu)
        z_new = mu +sigma*epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma



if __name__ == "__main__":
    
    x = torch.randn(1, 784)
    vae = VAE(input_dim=784, hid_dim=200, z_dim=20)
    recons, mu, sigma = vae(x)

    print(recons.shape)
    print(mu.shape)
    print(sigma.shape)




    

