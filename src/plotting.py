"""
plot_elastic_sweep.py

Loads results/elastic_sweep_results.pkl and produces:
  1. Heatmaps  — test accuracy / test loss for each (alpha, beta)
  2. Loss curves — train & test per (alpha, beta) vs SGD baseline
  3. Best config summary bar chart
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = "./results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────

df = pd.read_pickle(os.path.join(RESULTS_DIR, "elastic_sweep_results_lr0.8.pkl"))

sgd_row    = df[df["optimizer"] == "SGD"].iloc[0]
elastic_df = df[df["optimizer"] == "ElasticOptim"].copy()

sgd_acc       = sgd_row["test_accuracy"]
sgd_te_loss   = sgd_row["test_loss"]
sgd_te_curve  = sgd_row["test_loss_curve"]
sgd_tr_curve  = sgd_row["train_loss_curve"]

alphas = sorted(elastic_df["alpha"].unique())
betas  = sorted(elastic_df["beta"].unique())


# ── 1. Heatmaps ────────────────────────────────────────────────────────────

def make_grid(metric):
    grid = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            val = elastic_df.loc[
                (elastic_df["alpha"] == a) & (elastic_df["beta"] == b), metric
            ].values[0]
            grid[i, j] = val
    return grid

acc_grid  = make_grid("test_accuracy")
loss_grid = make_grid("test_loss")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ElasticOptim Parameter Sweep — MNIST", fontsize=14, fontweight="bold")

for ax, grid, title, cmap, fmt in zip(
    axes,
    [acc_grid,    loss_grid],
    ["Test Accuracy", "Test Loss"],
    ["YlGn",      "YlOrRd"],
    [".4f",       ".4f"],
):
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(betas)));  ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"α={a}" for a in alphas])
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax)

    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, format(grid[i, j], fmt),
                    ha="center", va="center", fontsize=9,
                    color="black" if grid[i, j] < grid.max() * 0.9 else "white")

# SGD reference lines in colorbar area (annotated as text)
for ax, val, label in zip(axes, [sgd_acc, sgd_te_loss], ["SGD acc", "SGD loss"]):
    ax.set_xlabel(f"(SGD baseline: {label} = {val:.4f})", fontsize=9, color="gray")

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "heatmaps.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ── 2. Loss curves grid ────────────────────────────────────────────────────

n_epochs = len(sgd_te_curve)
epochs   = range(1, n_epochs + 1)

fig, axes = plt.subplots(
    len(alphas), len(betas),
    figsize=(4 * len(betas), 3.5 * len(alphas)),
    sharex=True, sharey=True,
)
fig.suptitle("Train / Test Loss Curves  (— ElasticOptim   ·· SGD baseline)",
             fontsize=13, fontweight="bold")

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        ax  = axes[i][j]
        row = elastic_df.loc[
            (elastic_df["alpha"] == alpha) & (elastic_df["beta"] == beta)
        ].iloc[0]

        ax.plot(epochs, row["train_loss_curve"], color="#2196F3", lw=1.8, label="train")
        ax.plot(epochs, row["test_loss_curve"],  color="#F44336", lw=1.8, label="test")
        ax.plot(epochs, sgd_tr_curve, color="#2196F3", lw=1, ls=":", alpha=0.5)
        ax.plot(epochs, sgd_te_curve, color="#F44336", lw=1, ls=":", alpha=0.5)

        ax.set_title(f"α={alpha}  β={beta}\nacc={row['test_accuracy']:.4f}", fontsize=8)
        ax.tick_params(labelsize=7)

        if i == 0 and j == len(betas) - 1:
            ax.legend(fontsize=7, loc="upper right")

for i, alpha in enumerate(alphas):
    axes[i][0].set_ylabel("Loss", fontsize=8)
for j, beta in enumerate(betas):
    axes[-1][j].set_xlabel("Epoch", fontsize=8)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "loss_curves_grid.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")


# ── 3. Best configs bar chart ──────────────────────────────────────────────

top_n = 5
top   = elastic_df.nlargest(top_n, "test_accuracy").copy()
top["label"] = top.apply(lambda r: f"α={r['alpha']}\nβ={r['beta']}", axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#4CAF50" if v > sgd_acc else "#FF7043" for v in top["test_accuracy"]]
bars   = ax.bar(range(len(top)), top["test_accuracy"], color=colors, width=0.5)
ax.axhline(sgd_acc, color="black", lw=1.5, ls="--", label=f"SGD baseline ({sgd_acc:.4f})")
ax.set_xticks(range(len(top)))
ax.set_xticklabels(top["label"], fontsize=9)
ax.set_ylabel("Test Accuracy")
ax.set_title(f"Top {top_n} ElasticOptim Configs vs SGD Baseline", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(min(top["test_accuracy"].min(), sgd_acc) - 0.005, 1.0)

for bar, val in zip(bars, top["test_accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "top_configs.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

print("\nDone. All plots saved to", PLOTS_DIR)
