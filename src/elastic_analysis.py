"""
elastic_analysis.py

Sweeps alpha and beta for ElasticOptim on MNIST and compares against
an SGD baseline. Results are saved to ./results/ as a pickle and CSV.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from model import SimpleMLP, evaluate, test
from ElasticOptim import ElasticOptim


# ── Results dir ────────────────────────────────────────────────────────────

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data ───────────────────────────────────────────────────────────────────

transform     = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

print(f"Device: {device}")

# Fix batch order so SGD and ElasticOptim see identical data
all_batches = list(train_loader)


# ── Config ─────────────────────────────────────────────────────────────────

N_EPOCHS  = 10
N_WORKERS = 5
LR        = 0.8 #0.1
TAU       = 20

ALPHAS = [0.1, 0.5, 0.9]
BETAS  = [0.0, 1.0, 5.0]


# ── Results container ──────────────────────────────────────────────────────

columns = [
    "optimizer", "alpha", "beta",
    "train_loss", "test_loss", "test_accuracy",
    "train_loss_curve", "test_loss_curve",
]
results_df = pd.DataFrame(columns=columns)


# ── Helper ─────────────────────────────────────────────────────────────────

def run_epoch_curves(eval_model, n_epochs, step_fn):
    """Run n_epochs calling step_fn(epoch) each epoch, return loss curves."""
    train_curve, test_curve = [], []
    for epoch in range(n_epochs):
        step_fn(epoch)
        tr = evaluate(eval_model, train_loader, criterion, device)
        te = evaluate(eval_model, test_loader,  criterion, device)
        train_curve.append(tr)
        test_curve.append(te)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}  train={tr:.4f}  test={te:.4f}")
    return train_curve, test_curve


def log(name, alpha, beta, train_curve, test_curve, accuracy):
    results_df.loc[len(results_df)] = [
        name, alpha, beta,
        train_curve[-1], test_curve[-1], accuracy,
        train_curve, test_curve,
    ]


# ── SGD baseline (run once) ────────────────────────────────────────────────

print("\n══ SGD baseline ══")
sgd_model = SimpleMLP().to(device)
sgd_opt   = torch.optim.SGD(sgd_model.parameters(), lr=LR)

def sgd_step(epoch):
    sgd_model.train()
    for x, y in all_batches:
        x, y = x.to(device), y.to(device)
        sgd_opt.zero_grad()
        criterion(sgd_model(x), y).backward()
        sgd_opt.step()

tr_curve, te_curve = run_epoch_curves(sgd_model, N_EPOCHS, sgd_step)
acc = test(sgd_model, test_loader, device)
print(f"  SGD accuracy: {acc:.4f}")
log("SGD", alpha=None, beta=None, train_curve=tr_curve, test_curve=te_curve, accuracy=acc)


# ── ElasticOptim sweep ─────────────────────────────────────────────────────

for alpha in ALPHAS:
    for beta in BETAS:
        print(f"\n══ ElasticOptim  alpha={alpha}  beta={beta} ══")

        workers       = [SimpleMLP().to(device) for _ in range(N_WORKERS)]
        master        = SimpleMLP().to(device)
        worker_optims = [torch.optim.SGD(m.parameters(), lr=LR) for m in workers]

        elastic_opt = ElasticOptim(
            workers=workers,
            master=master,
            optimizers=worker_optims,
            val_loader=test_loader,
            alpha=alpha,
            beta=beta,
            tau=TAU,
            device=device,
        )

        def elastic_step(epoch):
            for i in range(0, len(all_batches) - N_WORKERS, N_WORKERS):
                elastic_opt.step(all_batches[i : i + N_WORKERS])

        tr_curve, te_curve = run_epoch_curves(master, N_EPOCHS, elastic_step)
        acc = test(master, test_loader, device)
        print(f"  accuracy: {acc:.4f}")
        log("ElasticOptim", alpha=alpha, beta=beta,
            train_curve=tr_curve, test_curve=te_curve, accuracy=acc)


# ── Save ───────────────────────────────────────────────────────────────────

pkl_path = os.path.join(RESULTS_DIR, f"elastic_sweep_results_lr{LR}.pkl")
csv_path = os.path.join(RESULTS_DIR, f"elastic_sweep_results_lr{LR}.csv")

results_df.to_pickle(pkl_path)
results_df.drop(columns=["train_loss_curve", "test_loss_curve"]).to_csv(csv_path, index=False)

print(f"\nSaved to {pkl_path} and {csv_path}")
print("\n══ Summary ══")
print(results_df[["optimizer", "alpha", "beta", "train_loss", "test_loss", "test_accuracy"]].to_string(index=False))
