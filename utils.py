"""
utils.py — Shared helpers for the Adversarial Attack & Robustness project.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── CIFAR-10 class labels ────────────────────────────────────────────────────
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ── Data loaders ─────────────────────────────────────────────────────────────
def get_loaders(batch_size=128, num_workers=2):
    """Return (train_loader, test_loader) for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


# ── CNN Architecture ──────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    A compact but capable CNN for CIFAR-10.
    Architecture: 3 conv blocks → 2 FC layers.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32→16
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16→8
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 8→4
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Accuracy helper ───────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """Return accuracy (float 0-100) over a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ── Clamp helper (keeps perturbed images in valid normalised range) ────────────
# CIFAR-10 pixel range after normalisation: roughly [-2, 2] per channel.
# We clamp to the true per-channel min/max so adversarial images stay realistic.
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616])

def clamp_to_valid(x):
    """Clamp tensor (B,C,H,W) to the valid normalised CIFAR-10 range."""
    device = x.device
    lo = ((0.0 - CIFAR_MEAN) / CIFAR_STD).to(device).view(1, 3, 1, 1)
    hi = ((1.0 - CIFAR_MEAN) / CIFAR_STD).to(device).view(1, 3, 1, 1)
    return torch.clamp(x, lo, hi)
