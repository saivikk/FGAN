from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
LATENT_SIZE = 64
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0002

DEVICE = torch.device("cpu")

SCENARIOS = {"small": 0, "medium": 1, "large": 2}

torch.manual_seed(42)

print("AstroSurface AI - Conditional DCGAN (128 x 128, anti-collapse)")
print("Device:", DEVICE)


# ==========================================
# DATASET
# ==========================================

class TerrainDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        for scenario, label in SCENARIOS.items():
            folder = root / scenario

            if not folder.exists():
                raise RuntimeError(
                    f"Missing {folder}. Run prepare_data.py first."
                )

            for path in sorted(folder.glob("*.png")):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = self.transform(image.convert("L"))

        return image, label


dataset = TerrainDataset(DATA_DIR)

if len(dataset) == 0:
    raise RuntimeError("No labelled images found.")

print("Total images:", len(dataset))

for name, label in SCENARIOS.items():
    count = sum(1 for _, y in dataset.samples if y == label)
    print(f"{name.capitalize()}: {count}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True          # needed for the batch-variety feature
)


# ==========================================
# GENERATOR (must match app.py exactly)
# ==========================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_embedding = nn.Embedding(NUM_CLASSES, 32)

        self.project = nn.Sequential(
            nn.Linear(LATENT_SIZE + 32, 256 * 4 * 4),
            nn.ReLU(True)
        )

        self.network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_features = self.label_embedding(labels)
        combined = torch.cat([noise, label_features], dim=1)

        x = self.project(combined)
        x = x.view(-1, 256, 4, 4)

        return self.network(x)


# ==========================================
# DISCRIMINATOR
# Adds a "batch variety" channel so the
# generator is punished for repeating itself
# ==========================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_embedding = nn.Embedding(
            NUM_CLASSES, IMAGE_SIZE * IMAGE_SIZE
        )

        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1)
        )

    def forward(self, image, labels):
        label_map = self.label_embedding(labels)
        label_map = label_map.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

        # Batch variety: how different are the images in this batch?
        variety = image.std(dim=0, unbiased=False).mean()
        variety_map = variety.expand(image.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)

        combined = torch.cat([image, label_map, variety_map], dim=1)
        return self.network(combined)


# ==========================================
# WEIGHT INITIALIZATION
# ==========================================

def weights_init(module):
    name = module.__class__.__name__

    if "Conv" in name or "Linear" in name:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)

        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)

    elif "Embedding" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


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

# Fixed previews: 3 classes x 3 different noise vectors
fixed_noise = torch.randn(9, LATENT_SIZE)
fixed_labels = torch.tensor([0, 1, 2] * 3, dtype=torch.long)


# ==========================================
# TRAINING
# ==========================================

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    # Instance noise: strong at the start, fades to zero at 70% of training
    noise_level = 0.10 * max(0.0, 1.0 - epoch / (EPOCHS * 0.7))

    total_g = 0.0
    total_d = 0.0
    batches = 0

    for real_images, labels in loader:
        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = real_images.size(0)

        smooth_real = torch.full((batch_size, 1), 0.9, device=DEVICE)
        real_targets = torch.ones(batch_size, 1, device=DEVICE)
        fake_targets = torch.zeros(batch_size, 1, device=DEVICE)

        # ---------- Train discriminator ----------
        d_optimizer.zero_grad()

        real_in = real_images + noise_level * torch.randn_like(real_images)
        real_scores = discriminator(real_in, labels)
        real_loss = criterion(real_scores, smooth_real)

        noise = torch.randn(batch_size, LATENT_SIZE, device=DEVICE)
        fake_images = generator(noise, labels)

        fake_in = fake_images.detach()
        fake_in = fake_in + noise_level * torch.randn_like(fake_in)
        fake_scores = discriminator(fake_in, labels)
        fake_loss = criterion(fake_scores, fake_targets)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # ---------- Train generator ----------
        g_optimizer.zero_grad()

        noise = torch.randn(batch_size, LATENT_SIZE, device=DEVICE)
        generated = generator(noise, labels)

        gen_in = generated + noise_level * torch.randn_like(generated)
        generated_scores = discriminator(gen_in, labels)
        g_loss = criterion(generated_scores, real_targets)

        g_loss.backward()
        g_optimizer.step()

        total_g += g_loss.item()
        total_d += d_loss.item()
        batches += 1

    avg_g = total_g / batches
    avg_d = total_d / batches

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"G loss: {avg_g:.4f} | D loss: {avg_d:.4f}"
    )

    # Preview every 5 epochs: rows = 3 noise vectors, columns = small/medium/large
    if (epoch + 1) % 5 == 0 or epoch == 0:
        generator.eval()

        with torch.no_grad():
            samples = generator(fixed_noise, fixed_labels).cpu()

        fig, axes = plt.subplots(3, 3, figsize=(7, 7))

        for i in range(9):
            row, col = divmod(i, 3)
            axes[row][col].imshow(
                samples[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1
            )
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(list(SCENARIOS)[col].capitalize())

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"conditional_epoch_{epoch + 1:03d}.png",
            dpi=110,
            bbox_inches="tight"
        )
        plt.close(fig)

    # Checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(
            generator.state_dict(),
            OUTPUT_DIR / f"conditional_generator_epoch_{epoch + 1:03d}.pth"
        )


# ==========================================
# SAVE FINAL MODEL
# ==========================================

torch.save(
    generator.state_dict(),
    OUTPUT_DIR / "conditional_generator.pth"
)

torch.save(
    discriminator.state_dict(),
    OUTPUT_DIR / "conditional_discriminator.pth"
)

print("\nCONDITIONAL GAN TRAINING COMPLETE!")
print("Final generator:", OUTPUT_DIR / "conditional_generator.pth")