"""
step4_robustness_eval.py — Comprehensive robustness evaluation.

Evaluates the clean-trained model under:
  1. FGSM at many ε values
  2. PGD at many ε values
  3. Per-class robustness breakdown
  4. Confidence analysis: clean vs adversarial

Run:
    python step4_robustness_eval.py

Output:
    ./outputs/robustness_comparison.png
    ./outputs/per_class_robustness.png
    ./outputs/confidence_histogram.png
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import SimpleCNN, get_loaders, clamp_to_valid, CLASSES, \
                  CIFAR_MEAN, CIFAR_STD

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = './checkpoints/cnn_clean.pth'
OUT_DIR   = './outputs'
os.makedirs(OUT_DIR, exist_ok=True)

model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
criterion = nn.CrossEntropyLoss()

_, test_loader = get_loaders(batch_size=128)
print(f"[Step 4] Model loaded. Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Attack helpers
# ─────────────────────────────────────────────────────────────────────────────
def fgsm(images, labels, eps):
    x = images.clone().detach().requires_grad_(True)
    criterion(model(x), labels).backward()
    return clamp_to_valid(images + eps * x.grad.sign()).detach()

def pgd(images, labels, eps, alpha=None, steps=20):
    if alpha is None:
        alpha = eps / 4
    x = clamp_to_valid(images + torch.empty_like(images).uniform_(-eps, eps))
    for _ in range(steps):
        x.requires_grad_(True)
        criterion(model(x), labels).backward()
        with torch.no_grad():
            x = clamp_to_valid(images + torch.clamp(x + alpha * x.grad.sign() - images, -eps, eps))
    return x.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Accuracy vs epsilon curves
# ─────────────────────────────────────────────────────────────────────────────
epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]

def sweep_accuracy(attack_fn):
    accs = []
    for eps in epsilons:
        correct = total = 0
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            adv   = attack_fn(imgs, lbls, eps) if eps > 0 else imgs
            preds = model(adv).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
        accs.append(100.0 * correct / total)
        print(f"  ε={eps:.3f}  acc={accs[-1]:.2f}%")
    return accs

print("\n── FGSM sweep ──")
fgsm_accs = sweep_accuracy(fgsm)
print("\n── PGD-20 sweep ──")
pgd_accs  = sweep_accuracy(pgd)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epsilons, fgsm_accs, 'o-', color='steelblue', linewidth=2, label='FGSM')
ax.plot(epsilons, pgd_accs,  's-', color='crimson',   linewidth=2, label='PGD-20')
ax.axhline(10, color='grey', linestyle=':', alpha=0.5, label='Random chance (10%)')
ax.fill_between(epsilons, fgsm_accs, pgd_accs, alpha=0.1, color='purple',
                label='PGD advantage over FGSM')
ax.set_xlabel('Perturbation Budget (ε)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Robustness: FGSM vs PGD Attack', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_ylim(-2, 102)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'robustness_comparison.png'), dpi=150)
print(f"\n[Step 4] Robustness comparison saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-class robustness at ε = 0.03
# ─────────────────────────────────────────────────────────────────────────────
EPS = 0.03
clean_per_class = [0] * 10
pgd_per_class   = [0] * 10
total_per_class = [0] * 10

for imgs, lbls in test_loader:
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    adv = pgd(imgs, lbls, EPS)

    clean_preds = model(imgs).argmax(1)
    adv_preds   = model(adv).argmax(1)

    for c in range(10):
        mask = (lbls == c)
        total_per_class[c]  += mask.sum().item()
        clean_per_class[c]  += (clean_preds[mask] == c).sum().item()
        pgd_per_class[c]    += (adv_preds[mask]   == c).sum().item()

clean_acc_cls = [100.0 * clean_per_class[c] / total_per_class[c] for c in range(10)]
pgd_acc_cls   = [100.0 * pgd_per_class[c]   / total_per_class[c] for c in range(10)]

x = np.arange(10)
fig2, ax2 = plt.subplots(figsize=(12, 5))
bars1 = ax2.bar(x - 0.2, clean_acc_cls, 0.4, label='Clean',         color='steelblue', alpha=0.85)
bars2 = ax2.bar(x + 0.2, pgd_acc_cls,   0.4, label=f'PGD (ε={EPS})', color='crimson',   alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=15)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title(f'Per-Class Robustness: Clean vs PGD-20 (ε={EPS})', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11); ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim(0, 105)

# Annotate drop
for b1, b2 in zip(bars1, bars2):
    drop = b1.get_height() - b2.get_height()
    ax2.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 1,
             f'−{drop:.0f}', ha='center', va='bottom', fontsize=7, color='darkred')

fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'per_class_robustness.png'), dpi=150)
print(f"[Step 4] Per-class robustness saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence (softmax max) distribution: clean vs adversarial
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F

clean_confs, adv_confs = [], []
for imgs, lbls in test_loader:
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    adv = pgd(imgs, lbls, EPS)

    with torch.no_grad():
        clean_confs.extend(F.softmax(model(imgs), dim=1).max(1).values.cpu().numpy())
        adv_confs.extend(  F.softmax(model(adv),  dim=1).max(1).values.cpu().numpy())

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(clean_confs, bins=50, alpha=0.6, color='steelblue', label='Clean', density=True)
ax3.hist(adv_confs,   bins=50, alpha=0.6, color='crimson',   label=f'PGD (ε={EPS})', density=True)
ax3.set_xlabel('Max Softmax Confidence', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Model Confidence: Clean vs Adversarial Inputs', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11); ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'confidence_histogram.png'), dpi=150)
print(f"[Step 4] Confidence histogram saved.")

print(f"\n[Step 4] All evaluation plots saved to {OUT_DIR}/")
print(f"  Mean clean confidence : {np.mean(clean_confs):.4f}")
print(f"  Mean adv   confidence : {np.mean(adv_confs):.4f}")
