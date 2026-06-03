"""
run_datasets.py
───────────────
Train a small neural network on MNIST or CIFAR-10 using EASGD, EASGD2, or SGD.
Loss curves are saved to ./figures/  and per-run results to ./results/

Usage examples
--------------
# Train on MNIST with EASGD (default)
python run_datasets.py --dataset mnist

# Train on CIFAR-10 with EASGD2
python run_datasets.py --dataset cifar10 --optimizer easgd2

# Run all three optimizers on MNIST for comparison
python run_datasets.py --dataset mnist --compare_all

# Full hyperparameter control
python run_datasets.py --dataset cifar10 --optimizer sgd \
    --lr 0.01 --num_epochs 20 --batch_size 128 --num_workers 4

Available optimizers
--------------------
  easgd  easgd2  sgd

Available datasets
------------------
  mnist  cifar10

Data paths expected (relative to working directory)
----------------------------------------------------
  data/MNIST/raw/train-images-idx3-ubyte  (and matching labels / test files)
  data/cifar-10-batches-py/data_batch_1 … data_batch_5 + test_batch
"""

import argparse
import os
import struct
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

def load_mnist(data_dir="data/MNIST/raw"):
    X_train = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
    y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    X_test  = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    y_test  = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test),  torch.tensor(y_test))

def unpickle(file):
    import pickle
    with open(file, "rb") as f:
        return pickle.load(f, encoding="bytes")

def load_cifar10(data_dir="data/cifar-10-batches-py"):
    batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
    X_train = np.concatenate([b[b"data"] for b in batches], axis=0).astype(np.float32)
    y_train = np.concatenate([b[b"labels"] for b in batches]).astype(np.int64)
    test    = unpickle(os.path.join(data_dir, "test_batch"))
    X_test  = test[b"data"].astype(np.float32)
    y_test  = np.array(test[b"labels"], dtype=np.int64)

    # Normalize to [0,1] and reshape to (N, 3, 32, 32)
    X_train = (X_train / 255.0).reshape(-1, 3, 32, 32)
    X_test  = (X_test  / 255.0).reshape(-1, 3, 32, 32)

    # Per-channel mean/std normalisation using training set
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std  = X_train.std( axis=(0, 2, 3), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test),  torch.tensor(y_test))


# ── Models ────────────────────────────────────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)


# ── EASGD / EASGD2 trainer wrappers ──────────────────────────────────────────

def make_model(dataset_name, device):
    if dataset_name == "mnist":
        return MnistCNN().to(device)
    return CifarCNN().to(device)


def train_sgd(model, train_loader, test_loader, lr, num_epochs, device):
    """Standard SGD training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        tloss, tacc = evaluate(model, test_loader, criterion, device)
        test_losses.append(tloss); test_accs.append(tacc)
        print(f"  Epoch {epoch+1:3d}/{num_epochs} — "
              f"train {train_losses[-1]:.4f}  test {tloss:.4f}  acc {tacc:.2%}")

    return train_losses, test_losses, test_accs


def train_easgd(model, train_loader, test_loader, lr, rho, num_workers,
                num_epochs, device, comm_period=5):
    """
    Synchronous EASGD: maintain `num_workers` copies of the model plus one
    central variable.  Every `comm_period` batches workers are pulled toward
    the center and the center is updated.
    """
    criterion = nn.CrossEntropyLoss()
    alpha = lr * rho

    # One optimizer per worker copy
    worker_models = [make_model_copy(model) for _ in range(num_workers)]
    worker_opts   = [torch.optim.SGD(m.parameters(), lr=lr) for m in worker_models]
    center_params = [p.data.clone() for p in model.parameters()]

    train_losses, test_losses, test_accs = [], [], []
    data_iter = iter(train_loader)
    total_batches = num_epochs * len(train_loader)
    batch_count = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            batch_count += 1

            for wi, (wm, wo) in enumerate(zip(worker_models, worker_opts)):
                wm.train()
                wo.zero_grad()
                loss = criterion(wm(xb), yb)
                loss.backward()
                # Add elastic coupling gradient
                with torch.no_grad():
                    for wp, cp in zip(wm.parameters(), center_params):
                        wp.grad.add_(alpha / lr * (wp.data - cp))
                wo.step()

            epoch_loss += loss.item() * len(xb)

            # Communication step
            if batch_count % comm_period == 0:
                with torch.no_grad():
                    for wi, wm in enumerate(worker_models):
                        for wp, cp in zip(wm.parameters(), center_params):
                            cp.add_(alpha * (wp.data - cp))

        # Set eval model to center params
        with torch.no_grad():
            for p, cp in zip(model.parameters(), center_params):
                p.data.copy_(cp)

        train_losses.append(epoch_loss / (len(train_loader.dataset) * num_workers))
        tloss, tacc = evaluate(model, test_loader, criterion, device)
        test_losses.append(tloss); test_accs.append(tacc)
        print(f"  Epoch {epoch+1:3d}/{num_epochs} — "
              f"train {train_losses[-1]:.4f}  test {tloss:.4f}  acc {tacc:.2%}")

    return train_losses, test_losses, test_accs


def train_easgd2(model, train_loader, test_loader, lr, alpha, alpha_pull,
                 beta, num_workers, num_epochs, device, comm_period=5):
    """
    EASGD2: weighted-average center update (Boltzmann weights from worker losses).
    """
    criterion = nn.CrossEntropyLoss()

    worker_models = [make_model_copy(model) for _ in range(num_workers)]
    worker_opts   = [torch.optim.SGD(m.parameters(), lr=lr) for m in worker_models]
    center_params = [p.data.clone() for p in model.parameters()]

    train_losses, test_losses, test_accs = [], [], []
    batch_count = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            batch_count += 1

            # Gradient steps for all workers
            worker_losses_batch = []
            for wm, wo in zip(worker_models, worker_opts):
                wm.train()
                wo.zero_grad()
                loss = criterion(wm(xb), yb)
                loss.backward()
                wo.step()
                worker_losses_batch.append(loss.item())

            epoch_loss += np.mean(worker_losses_batch) * len(xb)

            # Communication step with Boltzmann-weighted center update
            if batch_count % comm_period == 0:
                losses_arr = np.array(worker_losses_batch)
                # Numerically stable softmax
                losses_arr -= losses_arr.min()
                w = np.exp(-beta * losses_arr)
                w /= w.sum()

                with torch.no_grad():
                    for pi, cp in enumerate(center_params):
                        # Weighted average of worker params
                        weighted_sum = sum(
                            w[wi] * list(wm.parameters())[pi].data
                            for wi, wm in enumerate(worker_models)
                        )
                        new_cp = (1 - alpha)*cp + alpha*weighted_sum
                        # Pull workers toward center
                        for wi, wm in enumerate(worker_models):
                            wp = list(wm.parameters())[pi].data
                            wp.add_(-alpha_pull * w[wi] * (wp - cp))
                        cp.copy_(new_cp)

        # Set eval model to center params
        with torch.no_grad():
            for p, cp in zip(model.parameters(), center_params):
                p.data.copy_(cp)

        train_losses.append(epoch_loss / (len(train_loader.dataset) * num_workers))
        tloss, tacc = evaluate(model, test_loader, criterion, device)
        test_losses.append(tloss); test_accs.append(tacc)
        print(f"  Epoch {epoch+1:3d}/{num_epochs} — "
              f"train {train_losses[-1]:.4f}  test {tloss:.4f}  acc {tacc:.2%}")

    return train_losses, test_losses, test_accs


def make_model_copy(model):
    """Deep copy a model (same architecture, independent parameters)."""
    import copy
    return copy.deepcopy(model)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(xb)
    return total_loss / n, correct / n


# ── Plotting ──────────────────────────────────────────────────────────────────

DARK_BG = "#1e1e2e"
COLORS   = {"easgd": "#60a5fa", "easgd2": "#a78bfa", "sgd": "#f87171"}

def _apply_dark_style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4b5563")
        ax.tick_params(colors="#9ca3af")
        ax.xaxis.label.set_color("#9ca3af")
        ax.yaxis.label.set_color("#9ca3af")
        ax.title.set_color("white")
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor("#2d2d3f")
            for text in ax.get_legend().get_texts():
                text.set_color("white")


def plot_loss_curve(results_dict, dataset_name, figures_dir):
    """Plot train and test loss curves for all optimizers in results_dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for opt_name, res in results_dict.items():
        c = COLORS.get(opt_name, "white")
        epochs = range(1, len(res["train_losses"]) + 1)
        axes[0].plot(epochs, res["train_losses"], color=c, label=opt_name.upper(), lw=1.8)
        axes[1].plot(epochs, res["test_losses"],  color=c, label=opt_name.upper(), lw=1.8)

    axes[0].set_title(f"{dataset_name.upper()} — Train Loss")
    axes[1].set_title(f"{dataset_name.upper()} — Test Loss")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.2)

    _apply_dark_style(fig, axes)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_accuracy_curve(results_dict, dataset_name, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    for opt_name, res in results_dict.items():
        c = COLORS.get(opt_name, "white")
        epochs = range(1, len(res["test_accs"]) + 1)
        ax.plot(epochs, [a*100 for a in res["test_accs"]],
                color=c, label=opt_name.upper(), lw=1.8)
    ax.set_title(f"{dataset_name.upper()} — Test Accuracy (%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(True, alpha=0.2)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    path = os.path.join(figures_dir, f"{dataset_name}_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_results_csv(results_dict, dataset_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # Per-epoch CSV
    epoch_path = os.path.join(results_dir, f"{dataset_name}_epoch_results.csv")
    with open(epoch_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "epoch", "train_loss", "test_loss", "test_acc"])
        for opt_name, res in results_dict.items():
            for e, (tl, vl, acc) in enumerate(
                    zip(res["train_losses"], res["test_losses"], res["test_accs"]), 1):
                writer.writerow([opt_name, e, f"{tl:.6f}", f"{vl:.6f}", f"{acc:.6f}"])
    print(f"  Saved → {epoch_path}")

    # Summary CSV
    summary_path = os.path.join(results_dir, f"{dataset_name}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "final_train_loss", "final_test_loss",
                         "best_test_acc", "final_test_acc", "time_s"])
        for opt_name, res in results_dict.items():
            writer.writerow([
                opt_name,
                f"{res['train_losses'][-1]:.6f}",
                f"{res['test_losses'][-1]:.6f}",
                f"{max(res['test_accs']):.6f}",
                f"{res['test_accs'][-1]:.6f}",
                f"{res.get('time_s', 0):.1f}",
            ])
    print(f"  Saved → {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train on MNIST / CIFAR-10 with EASGD, EASGD2, or SGD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",     default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--optimizer",   default="easgd",
                   choices=["easgd", "easgd2", "sgd"])
    p.add_argument("--compare_all", action="store_true",
                   help="Run all three optimizers and overlay their curves")

    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--num_epochs",  type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--num_workers", type=int,   default=4,
                   help="Number of EASGD worker copies")
    p.add_argument("--comm_period", type=int,   default=5,
                   help="Batches between communication steps")

    # EASGD-specific
    p.add_argument("--rho",         type=float, default=0.1,
                   help="EASGD coupling strength")

    # EASGD2-specific
    p.add_argument("--beta",        type=float, default=1.0,
                   help="EASGD2 Boltzmann temperature beta")
    p.add_argument("--alpha",       type=float, default=0.3,
                   help="EASGD2 center update rate")
    p.add_argument("--alpha_pull",  type=float, default=0.3,
                   help="EASGD2 worker pull strength")

    p.add_argument("--data_dir",    default="data",
                   help="Root data directory (default: ./data)")
    p.add_argument("--figures_dir", default="./figures")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default=None,
                   help="'cpu', 'cuda', or 'mps' — auto-detected if omitted")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    print(f"\nLoading {args.dataset} …")
    if args.dataset == "mnist":
        X_tr, y_tr, X_te, y_te = load_mnist(
            os.path.join(args.data_dir, "MNIST", "raw"))
    else:
        X_tr, y_tr, X_te, y_te = load_cifar10(
            os.path.join(args.data_dir, "cifar-10-batches-py"))
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

    optimizers_to_run = (["easgd", "easgd2", "sgd"]
                         if args.compare_all else [args.optimizer])

    all_results = {}

    for opt_name in optimizers_to_run:
        print(f"\n{'='*55}")
        print(f" {opt_name.upper()} on {args.dataset.upper()}")
        print(f"{'='*55}")
        model = make_model(args.dataset, device)
        t0 = time.time()

        if opt_name == "sgd":
            train_losses, test_losses, test_accs = train_sgd(
                model, train_loader, test_loader,
                lr=args.lr, num_epochs=args.num_epochs, device=device)

        elif opt_name == "easgd":
            train_losses, test_losses, test_accs = train_easgd(
                model, train_loader, test_loader,
                lr=args.lr, rho=args.rho, num_workers=args.num_workers,
                num_epochs=args.num_epochs, device=device,
                comm_period=args.comm_period)

        elif opt_name == "easgd2":
            train_losses, test_losses, test_accs = train_easgd2(
                model, train_loader, test_loader,
                lr=args.lr, alpha=args.alpha, alpha_pull=args.alpha_pull,
                beta=args.beta, num_workers=args.num_workers,
                num_epochs=args.num_epochs, device=device,
                comm_period=args.comm_period)

        elapsed = time.time() - t0
        print(f"  → Finished in {elapsed:.1f}s  |  "
              f"Best test acc: {max(test_accs):.2%}")
        all_results[opt_name] = {
            "train_losses": train_losses,
            "test_losses":  test_losses,
            "test_accs":    test_accs,
            "time_s":       elapsed,
        }

    # Save plots and CSV
    print("\nSaving outputs …")
    plot_loss_curve(    all_results, args.dataset, args.figures_dir)
    plot_accuracy_curve(all_results, args.dataset, args.figures_dir)
    save_results_csv(   all_results, args.dataset, args.results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
