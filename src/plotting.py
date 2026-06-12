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

from utils.plot_utils import plot_heatmaps, plot_accuracy_heatmap, plot_loss_curves_grid, plot_top_configs


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


plot_heatmaps(
    elastic_df,
    alphas,
    betas,
    sgd_acc,
    sgd_te_loss,
    save_path=os.path.join(PLOTS_DIR, "heatmaps.png")
)

plot_accuracy_heatmap(
    elastic_df,
    alphas,
    betas,
    sgd_acc,
    save_path=os.path.join(PLOTS_DIR, "accuracy_heatmap.png"),
    fontsize=12,
    cmap="YlOrRd"
)

plot_loss_curves_grid(
    elastic_df,
    alphas,
    betas,
    sgd_tr_curve,
    sgd_te_curve,
    save_path=os.path.join(PLOTS_DIR, "loss_curves_grid.png")
)

plot_top_configs(
    elastic_df,
    sgd_acc,
    save_path=os.path.join(PLOTS_DIR, "top_configs.png")
)
