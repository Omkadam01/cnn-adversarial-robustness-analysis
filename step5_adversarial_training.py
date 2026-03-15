"""
step5_adversarial_training.py — Defense via Adversarial Training (Madry et al.)

Theory:
    Instead of minimising the expected loss on clean data, we minimise the
    worst-case (adversarial) loss:

        min_θ  E_{(x,y)} [ max_{δ: ‖δ‖∞≤ε}  L(θ, x+δ, y) ]

    In practice, for each training batch we:
      1. Generate PGD adversarial examples
      2. Train on the adversarial examples (not the clean ones)

    This makes the model robust but slightly reduces clean accuracy — the
    robustness–accuracy trade-off.

Run:
    python step5_adversarial_training.py

Output:
    ./checkpoints/cnn_robust.pth
    ./outputs/training_comparison.png   (clean vs robust model side-by-side)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import SimpleCNN, get_loaders, evaluate, clamp_to_valid

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR   = './outputs'
CKPT_DIR  = './checkpoints'
ROBUST_CKPT = os.path.join(CKPT_DIR, 'cnn_robust.pth')
CLEAN_CKPT  = os.path.join(CKPT_DIR, 'cnn_clean.pth')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
EPOCHS     = 25        # Fewer epochs — adversarial training is expensive
BATCH_SIZE = 128
LR         = 0.01
ADV_EPS    = 0.03      # ε for training
ADV_ALPHA  = 0.007     # step size
ADV_STEPS  = 7         # PGD steps during training (keep light for speed)


# ── Data ───────────────────────────────────────────────────────────────────
train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
criterion = nn.CrossEntropyLoss()

print(f"[Step 5] Adversarial Training | Device: {DEVICE}")
print(f"         ε={ADV_EPS}, α={ADV_ALPHA}, PGD steps={ADV_STEPS}, Epochs={EPOCHS}")


# ── PGD for training (inline, efficient version) ──────────────────────────
def pgd_train(model, images, labels, eps, alpha, steps):
    """Generate PGD adversarial examples (used during training)."""
    x = clamp_to_valid(images + torch.empty_like(images).uniform_(-eps, eps))
    for _ in range(steps):
        x.requires_grad_(True)
        criterion(model(x), labels).backward()
        with torch.no_grad():
            x = clamp_to_valid(images + torch.clamp(
                x + alpha * x.grad.sign() - images, -eps, eps))
    return x.detach()


# ── Adversarially-trained model ───────────────────────────────────────────
robust_model = SimpleCNN().to(DEVICE)
optimizer    = optim.SGD(robust_model.parameters(), lr=LR,
                         momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler    = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Tracking metrics
hist = {'epoch': [], 'train_loss': [], 'clean_acc': [], 'adv_acc': []}
best_robust_acc = 0.0

# ── PGD for evaluation ─────────────────────────────────────────────────────
def pgd_eval(model, images, labels, eps=ADV_EPS):
    return pgd_train(model, images, labels, eps, ADV_ALPHA, steps=20)

def adv_accuracy(model):
    correct = total = 0
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        adv   = pgd_eval(model, imgs, lbls)
        preds = model(adv).argmax(1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
    return 100.0 * correct / total


print("\nEpoch  Train Loss  Clean Acc  Adv Acc")
print("-" * 42)

for epoch in range(1, EPOCHS + 1):
    robust_model.train()
    total_loss = n = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        # ── Generate adversarial examples ──────────────────────────────
        robust_model.eval()                        # eval mode for attack gen
        adv_imgs = pgd_train(robust_model, imgs, lbls, ADV_EPS, ADV_ALPHA, ADV_STEPS)

        # ── Train on adversarial examples ──────────────────────────────
        robust_model.train()
        optimizer.zero_grad()
        loss = criterion(robust_model(adv_imgs), lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * lbls.size(0)
        n          += lbls.size(0)

    scheduler.step()

    train_loss = total_loss / n
    clean_acc  = evaluate(robust_model, test_loader, DEVICE)
    adv_acc    = adv_accuracy(robust_model)

    hist['epoch'].append(epoch)
    hist['train_loss'].append(train_loss)
    hist['clean_acc'].append(clean_acc)
    hist['adv_acc'].append(adv_acc)

    print(f"  {epoch:02d}     {train_loss:.4f}    {clean_acc:.2f}%   {adv_acc:.2f}%")

    if adv_acc > best_robust_acc:
        best_robust_acc = adv_acc
        torch.save(robust_model.state_dict(), ROBUST_CKPT)
        print(f"       ✓ Saved robust model (adv acc = {best_robust_acc:.2f}%)")

print(f"\n[Step 5] Best robust model adv acc: {best_robust_acc:.2f}%")
print(f"[Step 5] Saved to: {ROBUST_CKPT}")


# ─────────────────────────────────────────────────────────────────────────────
# Compare clean model vs robust model
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final Comparison ──")
clean_model = SimpleCNN().to(DEVICE)
clean_model.load_state_dict(torch.load(CLEAN_CKPT, map_location=DEVICE))
clean_model.eval()

robust_model.load_state_dict(torch.load(ROBUST_CKPT, map_location=DEVICE))
robust_model.eval()

epsilons   = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
clean_accs = []
robust_accs = []

for eps in epsilons:
    def eval_pgd(mdl, e):
        correct = total = 0
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            adv   = pgd_train(mdl, imgs, lbls, e, e/4, 20) if e > 0 else imgs
            preds = mdl(adv).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
        return 100.0 * correct / total

    ca = eval_pgd(clean_model,  eps)
    ra = eval_pgd(robust_model, eps)
    clean_accs.append(ca)
    robust_accs.append(ra)
    print(f"  ε={eps:.3f}  clean={ca:.1f}%  robust={ra:.1f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: accuracy vs epsilon
axes[0].plot(epsilons, clean_accs,  'o-', color='steelblue', linewidth=2, label='Standard Model')
axes[0].plot(epsilons, robust_accs, 's-', color='green',     linewidth=2, label='Adversarially Trained')
axes[0].axhline(10, color='grey', linestyle=':', alpha=0.5, label='Random (10%)')
axes[0].set_xlabel('Perturbation Budget (ε)', fontsize=12)
axes[0].set_ylabel('Accuracy under PGD-20 (%)', fontsize=12)
axes[0].set_title('Robustness: Standard vs Adversarially Trained', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-2, 102)

# Right: training history
epochs = hist['epoch']
axes[1].plot(epochs, hist['clean_acc'], '-', color='steelblue', linewidth=2, label='Clean Acc (adv. model)')
axes[1].plot(epochs, hist['adv_acc'],   '-', color='green',     linewidth=2, label='Adv Acc (adv. model)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Adversarial Training History', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'training_comparison.png'), dpi=150)
print(f"\n[Step 5] Comparison plot saved → {OUT_DIR}/training_comparison.png")
