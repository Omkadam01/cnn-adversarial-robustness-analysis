"""
step1_train_cnn.py — Train a CNN on CIFAR-10 and save the weights.

Run:
    python step1_train_cnn.py

Output:
    ./checkpoints/cnn_clean.pth   (saved model weights)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import SimpleCNN, get_loaders, evaluate

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 0.1
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR   = './checkpoints'
CKPT_PATH  = os.path.join(CKPT_DIR, 'cnn_clean.pth')

os.makedirs(CKPT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"[Step 1] Using device: {DEVICE}")
train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

# ── Model ──────────────────────────────────────────────────────────────────────
model = SimpleCNN().to(DEVICE)
print(f"[Step 1] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Optimiser & scheduler ─────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += labels.size(0)

    scheduler.step()

    train_loss = running_loss / total
    train_acc  = 100.0 * correct / total
    test_acc   = evaluate(model, test_loader, DEVICE)

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Loss: {train_loss:.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  ✓ Saved best model  (test acc = {best_acc:.2f}%)")

print(f"\n[Step 1] Training complete. Best test accuracy: {best_acc:.2f}%")
print(f"[Step 1] Model saved to: {CKPT_PATH}")
