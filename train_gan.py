
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS
# ==========================================

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
LATENT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0002

DEVICE = torch.device("cpu")

torch.manual_seed(42)

print("===================================")
print("       ASTROSURFACE AI")
print("===================================")
print("Device:", DEVICE)


# ==========================================
# 2. LOAD TRAINING IMAGES
# ==========================================

class TerrainDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(folder.glob("*.png"))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("L")
            return self.transform(image)


dataset = TerrainDataset(DATA_DIR)

print("Number of training images:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError(
        "No PNG images found in dataset/. Run prepare_data.py first."
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=False
)


# ==========================================
# 3. GENERATOR
# Noise -> Synthetic terrain image
# ==========================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(LATENT_SIZE, 128 * 4 * 4),
            nn.ReLU(True)
        )

        self.network = nn.Sequential(
            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 16 x 16 -> 32 x 32
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # 32 x 32 -> 64 x 64
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(-1, 128, 4, 4)
        return self.network(x)


# ==========================================
# 4. DISCRIMINATOR
# Image -> Real/fake score
# ==========================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # 64 x 64 -> 32 x 32
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 -> 16 x 16
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 x 16 -> 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 x 8 -> 4 x 4
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1)
        )

    def forward(self, image):
        return self.network(image)


# ==========================================
# 5. WEIGHT INITIALIZATION
# ==========================================

def weights_init(module):
    name = module.__class__.__name__

    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


# ==========================================
# 6. CREATE MODELS AND OPTIMIZERS
# ==========================================

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()

g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

fixed_noise = torch.randn(4, LATENT_SIZE)


# ==========================================
# 7. TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    batches = 0

    for real_images in loader:
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # ----------------------------------
        # Train discriminator
        # ----------------------------------

        d_optimizer.zero_grad()

        real_scores = discriminator(real_images)
        real_loss = criterion(real_scores, real_labels)

        noise = torch.randn(batch_size, LATENT_SIZE, device=DEVICE)
        fake_images = generator(noise)

        fake_scores = discriminator(fake_images.detach())
        fake_loss = criterion(fake_scores, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # ----------------------------------
        # Train generator
        # ----------------------------------

        g_optimizer.zero_grad()

        noise = torch.randn(batch_size, LATENT_SIZE, device=DEVICE)
        generated_images = generator(noise)

        generated_scores = discriminator(generated_images)
        g_loss = criterion(generated_scores, real_labels)

        g_loss.backward()
        g_optimizer.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        batches += 1

    avg_g_loss = total_g_loss / batches
    avg_d_loss = total_d_loss / batches

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Generator loss: {avg_g_loss:.4f} | "
        f"Discriminator loss: {avg_d_loss:.4f}"
    )

    # ----------------------------------
    # Save generated samples
    # ----------------------------------

    generator.eval()

    with torch.no_grad():
        samples = generator(fixed_noise.to(DEVICE)).cpu()

    fig, axes = plt.subplots(1, 4, figsize=(8, 2))

    for i, ax in enumerate(axes):
        ax.imshow(
            samples[i, 0].numpy(),
            cmap="gray",
            vmin=-1,
            vmax=1
        )
        ax.axis("off")

    plt.tight_layout()

    sample_path = OUTPUT_DIR / f"dcgan_epoch_{epoch + 1:02d}.png"
    plt.savefig(sample_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Save model checkpoints periodically
    if (epoch + 1) % 10 == 0:
        torch.save(
            generator.state_dict(),
            OUTPUT_DIR / f"generator_epoch_{epoch + 1:02d}.pth"
        )

# ==========================================
# 8. SAVE FINAL MODELS
# ==========================================

torch.save(
    generator.state_dict(),
    OUTPUT_DIR / "generator.pth"
)

torch.save(
    discriminator.state_dict(),
    OUTPUT_DIR / "discriminator.pth"
)

print("\n===================================")
print("DCGAN TRAINING COMPLETE!")
print("Generated samples:", OUTPUT_DIR.resolve())
print("Generator saved:", OUTPUT_DIR / "generator.pth")
print("Discriminator saved:", OUTPUT_DIR / "discriminator.pth")
print("===================================")
