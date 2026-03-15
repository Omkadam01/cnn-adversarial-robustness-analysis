"""
step3_pgd_attack.py — Projected Gradient Descent (PGD) attack on CIFAR-10.

Theory (Madry et al., 2018):
    PGD is a multi-step variant of FGSM constrained to an ε-ball (L∞ norm):

    x_0   = x + uniform_noise(-ε, ε)      # random start
    x_{t+1} = Π_{B(x,ε)} [ x_t + α · sign(∇_x L) ]

    where Π projects back into the ε-ball after each step.
    More steps → stronger attack that better finds the worst-case perturbation.

Run:
    python step3_pgd_attack.py

Output:
    Console: clean / FGSM / PGD accuracy comparison.
    Saves: ./outputs/pgd_examples.png
           ./outputs/fgsm_vs_pgd.png
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
criterion = nn.CrossEntropyLoss()
print(f"[Step 3] Model loaded. Device: {DEVICE}")

_, test_loader = get_loaders(batch_size=128)


# ─────────────────────────────────────────────────────────────────────────────
# FGSM (single-step) — kept here for comparison
# ─────────────────────────────────────────────────────────────────────────────
def fgsm_attack(images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    loss   = criterion(model(images), labels)
    loss.backward()
    adv = images + epsilon * images.grad.sign()
    return clamp_to_valid(adv).detach()


# ─────────────────────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────────────────────
def pgd_attack(images, labels, epsilon, alpha, num_steps, random_start=True):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Args:
        images      : clean batch (B,C,H,W)
        labels      : true labels (B,)
        epsilon     : maximum L∞ perturbation budget
        alpha       : step size per iteration
        num_steps   : number of gradient steps
        random_start: initialise inside ε-ball randomly (recommended)

    Returns:
        adv_images  : adversarial batch
    """
    x_orig = images.clone().detach()

    # Random initialisation inside the ε-ball
    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = clamp_to_valid(x_orig + delta)
    else:
        x_adv = x_orig.clone()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        loss = criterion(model(x_adv), labels)
        model.zero_grad()
        loss.backward()

        # Gradient step
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # Project onto ε-ball around original image (L∞ constraint)
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = clamp_to_valid(x_orig + delta)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate both attacks
# ─────────────────────────────────────────────────────────────────────────────
def batch_accuracy(attack_fn):
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        adv   = attack_fn(images, labels)
        preds = model(adv).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total

EPSILON = 0.03   # perturbation budget
ALPHA   = 0.007  # PGD step size  (≈ ε/4)
STEPS   = 20     # PGD iterations

clean_acc = evaluate(model, test_loader, DEVICE)
fgsm_acc  = batch_accuracy(lambda x, y: fgsm_attack(x, y, EPSILON))
pgd_acc   = batch_accuracy(lambda x, y: pgd_attack(x, y, EPSILON, ALPHA, STEPS))

print(f"\n{'Attack':>12}  {'Accuracy':>10}")
print("-" * 26)
print(f"{'Clean':>12}  {clean_acc:>9.2f}%")
print(f"{'FGSM':>12}  {fgsm_acc:>9.2f}%  (ε={EPSILON})")
print(f"{'PGD-20':>12}  {pgd_acc:>9.2f}%  (ε={EPSILON}, α={ALPHA})")


# ─────────────────────────────────────────────────────────────────────────────
# PGD strength vs number of steps
# ─────────────────────────────────────────────────────────────────────────────
step_counts = [1, 5, 10, 20, 40]
step_accs   = []
print("\nPGD accuracy vs # steps:")
for s in step_counts:
    acc = batch_accuracy(lambda x, y, s=s: pgd_attack(x, y, EPSILON, ALPHA, s))
    step_accs.append(acc)
    print(f"  PGD-{s:>2}: {acc:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Visualise PGD examples
# ─────────────────────────────────────────────────────────────────────────────
def denorm(t):
    m = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    s = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return torch.clamp(t.cpu() * s + m, 0, 1)

images_v, labels_v = next(iter(test_loader))
images_v = images_v[:8].to(DEVICE)
labels_v = labels_v[:8].to(DEVICE)

fgsm_v = fgsm_attack(images_v, labels_v, EPSILON)
pgd_v  = pgd_attack(images_v, labels_v, EPSILON, ALPHA, STEPS)

fp = model(fgsm_v).argmax(1).cpu()
pp = model(pgd_v).argmax(1).cpu()
lc = labels_v.cpu()

fig, axes = plt.subplots(3, 8, figsize=(18, 7))
fig.suptitle(f'Clean | FGSM (ε={EPSILON}) | PGD-{STEPS} (ε={EPSILON})',
             fontsize=12, fontweight='bold')
row_titles = ['Original', f'FGSM (ε={EPSILON})', f'PGD-{STEPS} (ε={EPSILON})']
imgs_rows  = [images_v, fgsm_v, pgd_v]
preds_rows = [model(images_v).argmax(1).cpu(), fp, pp]

for row, (row_imgs, row_preds) in enumerate(zip(imgs_rows, preds_rows)):
    for col in range(8):
        img = denorm(row_imgs[col]).permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        color = 'green' if row_preds[col] == lc[col] else 'red'
        axes[row, col].set_title(
            f'{row_titles[row]}\n{CLASSES[row_preds[col]]}', fontsize=6.5,
            color=color if row > 0 else 'black')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pgd_examples.png'), dpi=150, bbox_inches='tight')
print(f"\n[Step 3] PGD examples saved → {OUT_DIR}/pgd_examples.png")

# PGD steps curve
fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(step_counts, step_accs, 's-', color='crimson', linewidth=2, label='PGD')
ax.axhline(fgsm_acc, color='steelblue', linestyle='--', linewidth=2, label='FGSM (1 step)')
ax.axhline(10, color='grey', linestyle=':', alpha=0.5, label='Random (10%)')
ax.set_xlabel('PGD Steps', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'PGD: Accuracy vs Number of Steps (ε={EPSILON})', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fgsm_vs_pgd.png'), dpi=150)
print(f"[Step 3] FGSM vs PGD curve saved → {OUT_DIR}/fgsm_vs_pgd.png")
