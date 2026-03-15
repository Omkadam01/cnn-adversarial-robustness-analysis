# Adversarial Attack & Robustness in CNNs

> Studying how tiny, invisible perturbations can fool neural networks and how to defend against them.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What This Project Does

This project explores **adversarial machine learning** — a key area of AI safety — using the CIFAR-10 image dataset. We:

1. Train a CNN to classify images (~85% accuracy)
2. Attack it with **FGSM** (Fast Gradient Sign Method) — one-step noise
3. Attack it with **PGD** (Projected Gradient Descent) — stronger, iterative noise
4. Evaluate robustness across different attack strengths
5. Defend it using **Adversarial Training** — retrain on attacked images

The key finding: a model that looks 85% accurate is easily broken by invisible noise. Adversarial training recovers much of that robustness at a small accuracy cost.

---

## Results

| Model | Clean Accuracy | Under FGSM (ε=0.03) | Under PGD-20 (ε=0.03) |
|---|---|---|---|
| Standard training | ~85% | ~45% | ~20% |
| Adversarial training | ~75% | ~65% | ~55% |

---

## Project Structure

```
adversarial-cnn/
├── utils.py                      # CNN architecture, data loaders, shared helpers
├── step1_train_cnn.py            # Train baseline CNN on CIFAR-10
├── step2_fgsm_attack.py          # Fast Gradient Sign Method attack
├── step3_pgd_attack.py           # Projected Gradient Descent attack
├── step4_robustness_eval.py      # Full robustness evaluation suite
├── step5_adversarial_training.py # Adversarial training defense
├── Adversarial_Attack_CIFAR10.ipynb  # All-in-one Google Colab notebook
├── requirements.txt
└── README.md
```

## Theory

### FGSM — Goodfellow et al. (2015)
One step in the direction that maximises loss:
```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```

### PGD — Madry et al. (2018)
Multi-step attack constrained to an ε-ball (stronger):
```
x_{t+1} = Π_{B(x,ε)} [ x_t + α · sign(∇_x L) ]
```

### Adversarial Training (Defense)
Train on worst-case inputs each batch:
```
min_θ  E[(x,y)] [ max_{‖δ‖≤ε} L(θ, x+δ, y) ]
```

---

## Generated Plots

After running all steps, the `outputs/` folder contains:

| File | Description |
|---|---|
| `fgsm_examples.png` | Original vs adversarial images side by side |
| `fgsm_accuracy_curve.png` | Accuracy vs epsilon for FGSM |
| `pgd_examples.png` | Original / FGSM / PGD comparison |
| `fgsm_vs_pgd.png` | FGSM vs PGD across step counts |
| `robustness_comparison.png` | Both attacks across all epsilon values |
| `per_class_robustness.png` | Per-class clean vs adversarial accuracy |
| `confidence_histogram.png` | Model confidence shift under attack |
| `training_comparison.png` | Standard vs adversarially trained model |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- GPU recommended (runs on CPU but slow)
  
---

## References

- Goodfellow et al. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
- Szegedy et al. (2014). *Intriguing Properties of Neural Networks*. ICLR 2014.

---

## License

MIT License — free to use and modify.
