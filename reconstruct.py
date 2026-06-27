import torch
import torch.nn as nn

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import argparse

import os

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)
        return out
        
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path file model (.pth)"
)

parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index gambar Fashion-MNIST"
)

args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Latent dimension yang digunakan saat training
latent_dim = 32

# Membuat model
model = Autoencoder(latent_dim).to(device)

# Load model
model.load_state_dict(
    torch.load(args.model, map_location=device)
)

model.eval()

transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, label = test_dataset[args.index]

input_image = image.unsqueeze(0).to(device)

# Rekonstruksi gambar
with torch.no_grad():
    reconstructed = model(input_image)

# Pindahkan ke CPU
original = input_image.squeeze().cpu().numpy()
reconstructed = reconstructed.squeeze().cpu().numpy()

# Membuat folder output jika belum ada
os.makedirs("outputs", exist_ok=True)

plt.imsave(
    "outputs/original.png",
    original,
    cmap="gray"
)

plt.imsave(
    "outputs/reconstructed.png",
    reconstructed,
    cmap="gray"
)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(original, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed, cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/comparison.png",
    dpi=300
)

plt.show()

print("===================================")
print("Rekonstruksi berhasil!")
print("Hasil disimpan di folder outputs/")
print("===================================")