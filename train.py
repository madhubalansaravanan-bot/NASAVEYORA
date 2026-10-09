from pathlib import Path
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from unet_model import UNet


ROOT = Path("/content/drive/MyDrive/VEYORA_DATA")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
OUTPUT = "/content/best_model.pth"

PATCH = 256
EPOCHS = 10
BATCH = 4

from google.colab import drive
drive.mount("/content/drive")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def read_image(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)

    # Support channel-first or channel-last two-band rasters.
    if arr.shape[0] == 2:
        pass
    elif arr.shape[-1] == 2:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected VV/VH bands: {path}, {arr.shape}")

    # Dataset Part I is in dB. Match the same scaling at inference.
    arr = np.nan_to_num(arr, nan=-35, posinf=5, neginf=-35)
    return np.clip((arr + 35.0) / 40.0, 0, 1).astype(np.float32)


def read_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.float32)


pairs = []
for p in sorted(IMAGE_DIR.glob("*.tif")):
    m = MASK_DIR / p.name
    if m.exists():
        pairs.append((p, m))

if len(pairs) < 10:
    raise RuntimeError(
        f"Only {len(pairs)} matching TIFF pairs found. "
        "Check the folder names and filenames."
    )

train_pairs, val_pairs = train_test_split(
    pairs, test_size=0.2, random_state=42
)


class PatchDataset(Dataset):
    def __init__(self, pairs, patches_per_scene):
        self.pairs = pairs
        self.patches_per_scene = patches_per_scene

    def __len__(self):
        return len(self.pairs) * self.patches_per_scene

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx % len(self.pairs)]
        x = read_image(image_path)
        y = read_mask(mask_path)

        h, w = y.shape
        if h < PATCH or w < PATCH:
            raise ValueError(f"Scene is smaller than {PATCH}px: {image_path}")

        top = random.randint(0, h - PATCH)
        left = random.randint(0, w - PATCH)

        x = x[:, top:top+PATCH, left:left+PATCH]
        y = y[top:top+PATCH, left:left+PATCH]

        if random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()

        if random.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[::-1, :].copy()

        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(y[None].copy()).float()
        )


train_loader = DataLoader(
    PatchDataset(train_pairs, 8),
    batch_size=BATCH, shuffle=True, num_workers=2
)

val_loader = DataLoader(
    PatchDataset(val_pairs, 2),
    batch_size=BATCH, shuffle=False, num_workers=2
)

model = UNet(in_channels=2, base=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    denominator = prob.sum(dims) + target.sum(dims)
    return (1 - (2 * intersection + 1) /
            (denominator + 1)).mean()


best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_sum = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        loss.backward()
        optimizer.step()
        train_sum += loss.item()

    model.eval()
    val_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_sum += (bce(logits, y) + dice_loss(logits, y)).item()

    train_avg = train_sum / max(len(train_loader), 1)
    val_avg = val_sum / max(len(val_loader), 1)

    print(
        f"Epoch {epoch+1}/{EPOCHS}: "
        f"train={train_avg:.4f}, val={val_avg:.4f}"
    )

    if val_avg < best_loss:
        best_loss = val_avg
        torch.save(model.state_dict(), OUTPUT)

print("Saved model:", OUTPUT)
