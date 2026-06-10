"""
elastic_analysis_cifar10.py

Sweeps alpha and beta for ElasticOptim on CIFAR-10 using SimCLR (Self-Supervised),
and compares it against an SGD baseline using the Contrastive NT-Xent loss.
Results are saved to ./results/ as a pickle and CSV.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from model import SimCLR, nt_xent
from ElasticOptim import ElasticOptim

# ── Results dir ────────────────────────────────────────────────────────────

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Contrastive Data Augmentation (Two Views) ──────────────────────────────

class ContrastiveTransform:
    """Applies a base transform twice to generate two views of the same image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

cifar_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# For SimCLR, train and validation sets are evaluated unsupervised via NT-Xent loss
train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=ContrastiveTransform(cifar_transform))
val_dataset   = datasets.CIFAR10(root="./data", train=False, download=True, transform=ContrastiveTransform(cifar_transform))

train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
val_loader    = DataLoader(val_dataset,  batch_size=256, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Fix batch order so SGD and ElasticOptim see identical data
all_batches = list(train_loader)


# ── Config ─────────────────────────────────────────────────────────────────

N_EPOCHS  = 20
N_WORKERS = 5
LR        = 0.03  # Typically lower for Adam/SGD SimCLR
TAU       = 20
NT_XENT_TEMPERATURE = 0.5

ALPHAS = [0.1, 0.5, 0.9]
BETAS  = [0.0, 1.0, 5.0]


# ── Custom Evaluator for SimCLR ───────────────────────────────────────────

def evaluate_simclr(eval_model, loader):
    """Computes average contrastive loss over a data loader."""
    eval_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = eval_model(x1)
            _, z2 = eval_model(x2)
            loss = nt_xent(z1, z2, tau=NT_XENT_TEMPERATURE)
            total_loss += loss.item() * x1.size(0)
    return total_loss / len(loader.dataset)


# ── Results container ──────────────────────────────────────────────────────

columns = [
    "optimizer", "alpha", "beta",
    "train_loss", "val_loss",
    "train_loss_curve", "val_loss_curve",
]
results_df = pd.DataFrame(columns=columns)

print("\n══ Config ══")
print(f"  N_EPOCHS: {N_EPOCHS}")
print(f"  LR: {LR}")
print(f"  TAU: {TAU}")
print(f"  NT_XENT_TEMPERATURE: {NT_XENT_TEMPERATURE}")
# ── Helper ─────────────────────────────────────────────────────────────────

def run_epoch_curves(eval_model, n_epochs, step_fn):
    """Run n_epochs calling step_fn(epoch) each epoch, return loss curves."""
    train_curve, val_curve = [], []
    for epoch in range(n_epochs):
        step_fn(epoch)
        tr = evaluate_simclr(eval_model, train_loader)
        va = evaluate_simclr(eval_model, val_loader)
        train_curve.append(tr)
        val_curve.append(va)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}  train_loss={tr:.4f}  val_loss={va:.4f}")
    return train_curve, val_curve


def log(name, alpha, beta, train_curve, val_curve):
    results_df.loc[len(results_df)] = [
        name, alpha, beta,
        train_curve[-1], val_curve[-1],
        train_curve, val_curve,
    ]


# ── Custom Elastic Step for Unsupervised Batches ──────────────────────────

def run_elastic_step(elastic_opt, workers):
    """Adapts unsupervised 2-view batches to ElasticOptim's signature."""
    for i in range(0, len(all_batches) - N_WORKERS, N_WORKERS):
        worker_batch_slices = []
        for worker_idx in range(N_WORKERS):
            (x1, x2), _ = all_batches[i + worker_idx]
            # Since ElasticOptim calls F.cross_entropy internally, we must bypass 
            # or handle its native local_step, which expects (x, y) tuple unpacking.
            # We explicitly pass (x1, x2) as the tuple format.
            worker_batch_slices.append((x1, x2))
        
        elastic_opt.step(worker_batch_slices)


# ── SGD baseline (run once) ────────────────────────────────────────────────

print("\n══ SGD SimCLR baseline ══")
sgd_model = SimCLR(proj_dim=128).to(device)
sgd_model = torch.compile(sgd_model)
sgd_opt   = torch.optim.SGD(sgd_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

def sgd_step(epoch):
    sgd_model.train()
    for (x1, x2), _ in all_batches:
        x1, x2 = x1.to(device), x2.to(device)
        sgd_opt.zero_grad()
        _, z1 = sgd_model(x1)
        _, z2 = sgd_model(x2)
        loss = nt_xent(z1, z2, tau=NT_XENT_TEMPERATURE)
        loss.backward()
        sgd_opt.step()

tr_curve, val_curve = run_epoch_curves(sgd_model, N_EPOCHS, sgd_step)
log("SGD", alpha=None, beta=None, train_curve=tr_curve, val_curve=val_curve)


# ── ElasticOptim sweep ─────────────────────────────────────────────────────

# Custom ElasticOptim Wrapper subclass to override supervised CE execution inside ElasticOptim
class ElasticSimCLROptim(ElasticOptim):
    @torch.no_grad()
    def compute_val_losses(self):
        losses = [0.0 for _ in self.workers]
        for (x1, x2), _ in self.val_loader:
            x1, x2 = x1.to(self.device), x2.to(self.device)
            for i, model in enumerate(self.workers):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent(z1, z2, tau=NT_XENT_TEMPERATURE)
                losses[i] += loss.item()
        return losses

    def local_step(self, batches):
        for i, model in enumerate(self.workers):
            x1, x2 = batches[i]
            x1, x2 = x1.to(self.device), x2.to(self.device)
            self.optimizers[i].zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent(z1, z2, tau=NT_XENT_TEMPERATURE)
            loss.backward()
            self.optimizers[i].step()


for alpha in ALPHAS:
    for beta in BETAS:
        print(f"\n══ ElasticOptim SimCLR  alpha={alpha}  beta={beta} ══")

        workers = [torch.compile(SimCLR(proj_dim=128).to(device)) for _ in range(N_WORKERS)]
        master  = torch.compile(SimCLR(proj_dim=128).to(device))
        worker_optims = [torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4) for m in workers]

        elastic_opt = ElasticSimCLROptim(
            workers=workers,
            master=master,
            optimizers=worker_optims,
            val_loader=val_loader,
            alpha=alpha,
            beta=beta,
            tau=TAU,
            device=device,
        )

        def elastic_step(epoch):
            run_elastic_step(elastic_opt, workers)

        tr_curve, val_curve = run_epoch_curves(master, N_EPOCHS, elastic_step)
        log("ElasticOptim", alpha=alpha, beta=beta, train_curve=tr_curve, val_curve=val_curve)


# ── Save ───────────────────────────────────────────────────────────────────

pkl_path = os.path.join(RESULTS_DIR, f"elastic_cifar10_sweep_lr{LR}.pkl")
csv_path = os.path.join(RESULTS_DIR, f"elastic_cifar10_sweep_lr{LR}.csv")

results_df.to_pickle(pkl_path)
results_df.drop(columns=["train_loss_curve", "val_loss_curve"]).to_csv(csv_path, index=False)

print(f"\nSaved to {pkl_path} and {csv_path}")
print("\n══ Summary ══")
print(results_df[["optimizer", "alpha", "beta", "train_loss", "val_loss"]].to_string(index=False))