import torch
import torchvision.datasets as datasets
from torch import nn
from vae import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

dataset = datasets.MNIST(root="./vaedata", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fxn = nn.BCELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)

        x_recons, mu, sigma = model(x)

        recons_loss = loss_fxn(x_recons, x)
        kl_div = -torch.sum(1+torch.log(sigma.pow(2)) - mu.pow(2)-sigma.pow(2))


        loss = recons_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


def inference(digits, num_example):
    images = []
    index = 0
    for x, y in dataset:
        if y == index:
            images.append(x)
            index+=1
        if index == 10:
            break

    encoded = []

    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784).to(DEVICE))
            encoded.append((mu, sigma))

    mu, sigma = encoded[digits]

    for num in range(num_example):
        epsilon = torch.randn_like(sigma)
        z = mu+sigma*epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"gen_{digits}_ex{num}.png")

    
for index in range(10):
    inference(index, 5)

