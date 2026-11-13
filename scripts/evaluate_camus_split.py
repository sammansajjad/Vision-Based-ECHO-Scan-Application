"""Evaluate the trained U-Net on a whole CAMUS split.

Usage:
    python scripts\\evaluate_camus_split.py [split] [max_samples]

Examples:
    python scripts\\evaluate_camus_split.py validation
    python scripts\\evaluate_camus_split.py train 40
"""
import sys
from pathlib import Path

import numpy as np
import torch

# NOTE: adjust this import if your dataset class lives in a different module.
from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.models.unet import UNet
from cv_pipeline.postprocess import keep_largest_component


split = sys.argv[1] if len(sys.argv) > 1 else "validation"
max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None

checkpoint_path = Path("outputs/checkpoints/unet_4ch_best_full_validation.pth")
device = torch.device("cpu")


def dice_score(gt, pr):
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    return float(2 * (gt & pr).sum() / (gt.sum() + pr.sum() + 1e-8))


checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(state_dict)
model.eval()

dataset = CAMUSDataset(split=split, view="4CH", phases=("ED", "ES"))
n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

results = []  # (dice_post, dice_raw, patient_id, phase)
for i in range(n):
    sample = dataset[i]
    x = torch.from_numpy(sample["image"]).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()

    raw_pred = (prob >= 0.5).astype(np.uint8)
    post_pred = keep_largest_component(raw_pred)

    d_raw = dice_score(sample["mask"], raw_pred)
    d_post = dice_score(sample["mask"], post_pred)
    results.append((d_post, d_raw, sample["patient_id"], sample["phase"]))

    if (i + 1) % 20 == 0 or i == n - 1:
        print(f"  evaluated {i + 1}/{n}")

post = np.array([r[0] for r in results])
raw = np.array([r[1] for r in results])

print(f"\nSplit: {split} | samples: {n}")
print(f"Mean Dice (raw):            {raw.mean():.3f}")
print(f"Mean Dice (post-processed): {post.mean():.3f}")
print(f"Median Dice (post):         {np.median(post):.3f}")
print(f"Std (post):                 {post.std():.3f}")
print(f"Min / Max (post):           {post.min():.3f} / {post.max():.3f}")

for ph in ("ED", "ES"):
    vals = [r[0] for r in results if r[3] == ph]
    if vals:
        print(f"Mean Dice {ph} (post): {np.mean(vals):.3f}")

print("\nWorst 5 cases (post-processed):")
for d, _, pid, ph in sorted(results)[:5]:
    print(f"  {pid} {ph}: Dice {d:.3f}")

print("\nBest 5 cases (post-processed):")
for d, _, pid, ph in sorted(results, reverse=True)[:5]:
    print(f"  {pid} {ph}: Dice {d:.3f}")