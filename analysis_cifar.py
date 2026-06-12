"""
analysis_cifar.py

Self-supervised SimCLR training on CIFAR-10 + kNN evaluation.
Compares SGD baseline vs ElasticOptimSimCLR across (alpha, beta) grid.
Includes cosine LR schedule + checkpointing.
"""

import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import argparse


from utils.simCLR_helpers import SimCLRTransform, collate_fn, knn, extract_features
from src.model import SimCLR, nt_xent
from src.ElasticOptim import ElasticOptimSimCLR

# ── Results dir ────────────────────────────────────────────────────────────

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Training hyperparameter configuration")

    p.add_argument("--sgd_epochs", type=int,   default=2)
    p.add_argument("--n_epochs",   type=int,   default=2)
    p.add_argument("--n_workers",  type=int,   default=4)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--tau",        type=int,   default=20)

    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--alphas_pull", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--betas",  type=float, nargs="+", default=[0.0, 1.0, 5.0])

    return p.parse_args()


args = parse_args()

SGD_EPOCHS = args.sgd_epochs
N_EPOCHS   = args.n_epochs
N_WORKERS  = args.n_workers
LR         = args.lr
TAU        = args.tau
ALPHAS     = args.alphas
ALPHAS_PULL= args.alphas_pull
BETAS      = args.betas

eval_frac = 0.05 # fraction of data to use for kNN eval (to speed up)
DEBUG = False # if True, runs only 3 mini-batches per epoch for quick testing

# ── Results container ──────────────────────────────────────────────────────

columns = [
    "optimizer", "alpha", "alpha_pull", "beta",
    "train_loss", "test_accuracy",
    "train_loss_curve",
]
results_df = pd.DataFrame(columns=columns)

# ── Device ────────────────────────────────────────────────────────────────

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available()
#     else "cpu"
# )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ── Data ──────────────────────────────────────────────────────────────────

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=SimCLRTransform())
test_ds  = CIFAR10(root="./data", train=False, download=True, transform=SimCLRTransform())

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

# ── Cosine LR schedule ────────────────────────────────────────────────────

warmup_epochs = int(0.05 * SGD_EPOCHS)

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / (SGD_EPOCHS - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * t))

# ══════════════════════════════════════════════════════════════════════════
# 1.  SGD BASELINE
# ══════════════════════════════════════════════════════════════════════════

print("\n══ Training SGD baseline ══")

sgd_model = SimCLR().to(device)
sgd_opt   = torch.optim.SGD(sgd_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(sgd_opt, lr_lambda)

sgd_loss_curve = []

for epoch in tqdm(range(SGD_EPOCHS), desc="Training SGD"):
    sgd_model.train()
    total_loss = 0.0
    idx = 0
    for (x1, x2), _ in tqdm(train_loader, desc="Training batches"):
        if DEBUG:
            idx+=1
            if idx >= 3:
                break
        x1, x2 = x1.to(device), x2.to(device)

        _, z1 = sgd_model(x1)
        _, z2 = sgd_model(x2)

        loss = nt_xent(z1, z2)
        sgd_opt.zero_grad()
        loss.backward()
        sgd_opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    sgd_loss_curve.append(avg_loss)
    scheduler.step()

    print(f"[SGD Epoch {epoch+1}/{SGD_EPOCHS}] loss={avg_loss:.4f}  lr={sgd_opt.param_groups[0]['lr']:.5f}")

    if (epoch + 1) % 20 == 0 or epoch == SGD_EPOCHS - 1:
        path = os.path.join(RESULTS_DIR, f"simclr_sgd_epoch{epoch}.pth")
        torch.save(sgd_model.state_dict(), path)
        print("  Saved:", path)

print("eval kNN on SGD model...")
# kNN eval for SGD
feat_tr, y_tr = extract_features(sgd_model, train_loader, device, frac=eval_frac)
feat_te, y_te = extract_features(sgd_model, test_loader,  device, frac=eval_frac)
sgd_acc = knn(feat_tr, y_tr, feat_te, y_te, k=20)
print(f"SGD kNN accuracy: {sgd_acc * 100:.2f}%")

results_df = pd.concat([results_df, pd.DataFrame([{
    "optimizer":        "SGD",
    "alpha":            None,
    "alpha_pull":       None,
    "beta":             None,
    "train_loss":       sgd_loss_curve[-1],
    "test_accuracy":    sgd_acc,
    "train_loss_curve": sgd_loss_curve,
}])], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════
# 2.  ELASTIC SGD SWEEP
# ══════════════════════════════════════════════════════════════════════════

print("\n══ Starting ElasticSimCLR sweep ══")
for alpha in ALPHAS:
    for alpha_pull in ALPHAS_PULL:
        for beta in BETAS:
            print(f"\n══ ElasticSimCLR  alpha={alpha} alpha_pull={alpha_pull}  beta={beta} ══")

            workers = [SimCLR().to(device) for _ in range(N_WORKERS)]
            master  = SimCLR().to(device)

            worker_opts = [
                torch.optim.SGD(w.parameters(), lr=LR)
                for w in workers
            ]

            elastic_opt = ElasticOptimSimCLR(
                workers=workers,
                master=master,
                optimizers=worker_opts,
                alpha=alpha,
                alpha_pull=alpha_pull,
                beta=beta,
                tau=TAU,
                device=device,
            )

            elastic_loss_curve = []

            for epoch in tqdm(range(N_EPOCHS), desc="Training ElasticSGD"):
                batches = list(train_loader)
                epoch_loss = 0.0
                n_steps    = 0
                
                idx = 0
                for i in range(0, len(batches) - N_WORKERS, N_WORKERS):
                    if DEBUG:
                        idx+=1
                        if idx >= 3:
                            break
                    elastic_opt.step(batches[i : i + N_WORKERS])

                    # track average worker loss for this mini-step
                    with torch.no_grad():
                        step_loss = 0.0
                        for w_idx, worker in enumerate(workers):
                            (x1, x2), _ = batches[i + w_idx]
                            x1, x2 = x1.to(device), x2.to(device)
                            _, z1 = worker(x1)
                            _, z2 = worker(x2)
                            step_loss += nt_xent(z1, z2).item()
                        epoch_loss += step_loss / N_WORKERS
                        n_steps    += 1

                avg_loss = epoch_loss / max(n_steps, 1)
                elastic_loss_curve.append(avg_loss)
                print(f"  [Epoch {epoch+1}/{N_EPOCHS}] avg_worker_loss={avg_loss:.4f}")

            # kNN eval on master
            feat_tr, y_tr = extract_features(master, train_loader, device, frac=eval_frac)
            feat_te, y_te = extract_features(master, test_loader,  device, frac=eval_frac)
            acc = knn(feat_tr, y_tr, feat_te, y_te, k=20)
            print(f"  kNN accuracy: {acc * 100:.2f}%")

            results_df = pd.concat([results_df, pd.DataFrame([{
                "optimizer":        "ElasticSGD",
                "alpha":            alpha,
                "alpha_pull":       alpha_pull,
                "beta":             beta,
                "train_loss":       elastic_loss_curve[-1],
                "test_accuracy":    acc,
                "train_loss_curve": elastic_loss_curve,
            }])], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════
# 3.  SAVE
# ══════════════════════════════════════════════════════════════════════════

SAVE_DIR = "./results/cifar_simCLR"
os.makedirs(SAVE_DIR, exist_ok=True)

pkl_path = os.path.join(SAVE_DIR, f"elastic_sweep_results_lr{LR}_modPull.pkl")
csv_path = os.path.join(SAVE_DIR, f"elastic_sweep_results_lr{LR}_modPull.csv")

results_df.to_pickle(pkl_path)
results_df.drop(columns=["train_loss_curve"]).to_csv(csv_path, index=False)

print(f"\nSaved to {pkl_path} and {csv_path}")
print("\n══ Summary ══")
print(results_df[["optimizer", "alpha", "alpha_pull", "beta", "train_loss", "test_accuracy"]].to_string(index=False))