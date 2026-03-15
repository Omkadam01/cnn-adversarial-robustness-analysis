"""
step2_fgsm_attack.py — Fast Gradient Sign Method (FGSM) attack on CIFAR-10.

Theory:
    x_adv = x + ε · sign(∇_x L(θ, x, y))

    We nudge every pixel one step in the direction that INCREASES the loss,
    fooling the network with an imperceptible perturbation.

Run:
    python step2_fgsm_attack.py

Output:
    Console: clean accuracy vs FGSM accuracy at multiple ε values.
    Saves visualisation: ./outputs/fgsm_examples.png
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import SimpleCNN, get_loaders, evaluate, clamp_to_valid, CLASSES, \
                  CIFAR_MEAN, CIFAR_STD

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = './checkpoints/cnn_clean.pth'
OUT_DIR   = './outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print(f"[Step 2] Model loaded from {CKPT_PATH}")

_, test_loader = get_loaders(batch_size=128)
criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────────────────────
# FGSM function
# ─────────────────────────────────────────────────────────────────────────────
def fgsm_attack(model, images, labels, epsilon, criterion):
    """
    Generate FGSM adversarial examples.

    Args:
        images  : clean batch (B,C,H,W), requires_grad will be set internally
        labels  : ground-truth labels (B,)
        epsilon : perturbation magnitude (in normalised pixel space)

    Returns:
        adv_images : adversarial batch, clamped to valid range
    """
    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)

    images.requires_grad = True                    # enable gradient w.r.t. input

    outputs = model(images)
    loss    = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()                                # ∂L/∂x

    # FGSM step: x + ε · sign(∂L/∂x)
    sign_grad  = images.grad.data.sign()
    adv_images = images + epsilon * sign_grad
    adv_images = clamp_to_valid(adv_images)        # stay in valid pixel range

    return adv_images.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate across multiple ε values
# ─────────────────────────────────────────────────────────────────────────────
def fgsm_accuracy(epsilon):
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        adv = fgsm_attack(model, images, labels, epsilon, criterion)
        preds = model(adv).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total

clean_acc = evaluate(model, test_loader, DEVICE)
print(f"\n{'Epsilon':>10}  {'Accuracy':>10}")
print("-" * 25)
print(f"{'Clean':>10}  {clean_acc:>9.2f}%")

epsilons = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
accs     = []
for eps in epsilons:
    acc = fgsm_accuracy(eps)
    accs.append(acc)
    print(f"{eps:>10.3f}  {acc:>9.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Visualise examples: original vs adversarial
# ─────────────────────────────────────────────────────────────────────────────
def denorm(tensor):
    """Convert normalised tensor back to [0,1] for display."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)

# Grab one batch for visualisation
images_vis, labels_vis = next(iter(test_loader))
images_vis = images_vis[:8].to(DEVICE)
labels_vis = labels_vis[:8].to(DEVICE)

eps_vis   = 0.05
adv_vis   = fgsm_attack(model, images_vis, labels_vis, eps_vis, criterion)

clean_preds = model(images_vis).argmax(1).cpu()
adv_preds   = model(adv_vis).argmax(1).cpu()
labels_cpu  = labels_vis.cpu()

fig, axes = plt.subplots(3, 8, figsize=(18, 7))
fig.suptitle(f'FGSM Attack  (ε = {eps_vis})  —  Top: Original | Middle: Adversarial | Bottom: Perturbation ×10',
             fontsize=12, fontweight='bold')

for i in range(8):
    orig = denorm(images_vis[i]).permute(1, 2, 0).numpy()
    adv  = denorm(adv_vis[i]).permute(1, 2, 0).numpy()
    diff = np.clip((adv - orig) * 10 + 0.5, 0, 1)   # amplified difference

    axes[0, i].imshow(orig)
    axes[0, i].set_title(f'True: {CLASSES[labels_cpu[i]]}\n'
                          f'Pred: {CLASSES[clean_preds[i]]}', fontsize=7)
    axes[0, i].axis('off')

    axes[1, i].imshow(adv)
    axes[1, i].set_title(f'Adv pred:\n{CLASSES[adv_preds[i]]}', fontsize=7,
                          color='red' if adv_preds[i] != labels_cpu[i] else 'green')
    axes[1, i].axis('off')

    axes[2, i].imshow(diff)
    axes[2, i].set_title('Perturbation×10', fontsize=7)
    axes[2, i].axis('off')

plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'fgsm_examples.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n[Step 2] Visualisation saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot accuracy vs epsilon
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot([0] + epsilons, [clean_acc] + accs, 'o-', color='steelblue', linewidth=2)
ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='Random (10%)')
ax.set_xlabel('Epsilon (ε)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('FGSM Attack: Accuracy vs Perturbation Strength', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fgsm_accuracy_curve.png'), dpi=150)
print(f"[Step 2] Accuracy curve saved → {OUT_DIR}/fgsm_accuracy_curve.png")
