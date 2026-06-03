"""
analyze_results.py
──────────────────
Analyzes optimization_results_2.csv to identify best hyperparameter
configurations for EASGD2.

Produces:
  1. Console summary: top configs per function + global best
  2. Correlation analysis: which parameters matter most
  3. Heatmaps: eta vs alpha, eta vs alpha_pull, alpha vs alpha_pull
     (one figure per function + one aggregated figure)
  4. Marginal effect plots: mean loss vs each parameter
  5. Best config recommendation with confidence bands

Usage
-----
python analyze_results.py                          # uses optimization_results_2.csv
python analyze_results.py --csv my_results.csv     # custom file
python analyze_results.py --top_n 5                # show top 5 per function
python analyze_results.py --no_plots               # console only
python analyze_results.py --out_dir ./my_figures   # custom output directory
"""

import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from collections import defaultdict

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "border":   "#2e3347",
    "text":     "#e8eaf0",
    "muted":    "#6b7280",
    "accent1":  "#6c8ebf",
    "accent2":  "#82c88a",
    "accent3":  "#f0a070",
    "bad":      "#e05c5c",
    "good":     "#5cc8a0",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linewidth":    0.6,
    "legend.facecolor":  PALETTE["surface"],
    "legend.edgecolor":  PALETTE["border"],
    "font.family":       "monospace",
    "font.size":         9,
})

FN_ORDER = ["sphere", "rosenbrock", "rastrigin", "himmelblau", "styblinski_tang", "ackley"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_file):
    if not os.path.isfile(csv_file):
        sys.exit(f"[ERROR] File not found: {csv_file}")

    rows = []
    with open(csv_file, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "function":    r["function"],
                "eta":         float(r["eta"]),
                "alpha":       float(r["alpha"]),
                "alpha_pull":  float(r["alpha_pull"]),
                "num_workers": int(r["num_workers"]),
                "beta":        float(r["beta"]),
                "num_epochs":  int(r["num_epochs"]),
                "final_loss":  float(r["final_loss"]),
                "best_loss":   float(r["best_loss"]),
                "converged":   int(r["converged"]),
                "elapsed_s":   float(r.get("elapsed_s", 0)),
            })
    print(f"[INFO] Loaded {len(rows)} rows from {csv_file}")
    return rows


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(rows, top_n=3):
    by_fn = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    print("\n" + "═" * 72)
    print(f"  TOP-{top_n} CONFIGS PER FUNCTION  (sorted by final_loss)")
    print("═" * 72)

    for fn_name in sorted(by_fn):
        fn_rows = sorted(by_fn[fn_name], key=lambda r: r["final_loss"])
        print(f"\n  {fn_name.upper()}")
        print(f"  {'eta':<10} {'alpha':<8} {'alpha_pull':<12} {'final_loss':<14} {'best_loss':<14} conv")
        print(f"  {'-'*8:<10} {'-'*6:<8} {'-'*10:<12} {'-'*12:<14} {'-'*12:<14} ----")
        for r in fn_rows[:top_n]:
            print(f"  {r['eta']:<10.5f} {r['alpha']:<8.4f} {r['alpha_pull']:<12.4f} "
                  f"{r['final_loss']:<14.4e} {r['best_loss']:<14.4e} "
                  f"{'YES' if r['converged'] else ' no'}")

    all_sorted = sorted(rows, key=lambda r: r["final_loss"])
    print("\n" + "═" * 72)
    print("  GLOBAL BEST CONFIGS  (across all functions)")
    print("═" * 72)
    print(f"  {'function':<18} {'eta':<10} {'alpha':<8} {'alpha_pull':<12} {'final_loss'}")
    print(f"  {'-'*16:<18} {'-'*8:<10} {'-'*6:<8} {'-'*10:<12} -----------")
    for r in all_sorted[:top_n * 2]:
        print(f"  {r['function']:<18} {r['eta']:<10.5f} {r['alpha']:<8.4f} "
              f"{r['alpha_pull']:<12.4f} {r['final_loss']:.4e}")
    print()


def print_correlation(rows):
    """Spearman rank correlation between each hyperparameter and final_loss."""
    params = ["eta", "alpha", "alpha_pull"]
    losses = np.array([r["final_loss"] for r in rows])

    # Use log-loss for correlation (more meaningful for losses spanning decades)
    log_losses = np.log1p(np.clip(losses, 0, None))
    rank_loss  = np.argsort(np.argsort(log_losses)).astype(float)

    print("  PARAMETER CORRELATION WITH FINAL LOSS  (Spearman, all functions)")
    print(f"  {'parameter':<14} {'rho':>8}  interpretation")
    print(f"  {'-'*12:<14} {'-'*6:>8}  ---------------")
    for p in params:
        vals     = np.array([r[p] for r in rows])
        rank_val = np.argsort(np.argsort(vals)).astype(float)
        n        = len(rank_val)
        rho      = 1 - 6 * np.sum((rank_val - rank_loss)**2) / (n * (n**2 - 1))
        direction = "↑ higher → worse" if rho > 0.1 else ("↓ higher → better" if rho < -0.1 else "  no clear trend")
        print(f"  {p:<14} {rho:>8.3f}  {direction}")
    print()


# ── Marginal effect plots ─────────────────────────────────────────────────────

def plot_marginals(rows, out_dir):
    """Mean (and std) of final_loss vs each parameter, per function."""
    params     = ["eta", "alpha", "alpha_pull"]
    param_labels = {"eta": "η (eta)", "alpha": "α (alpha)", "alpha_pull": "α_pull"}
    by_fn      = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    fn_names = [f for f in FN_ORDER if f in by_fn]
    n_fn     = len(fn_names)

    fig, axes = plt.subplots(n_fn, 3, figsize=(14, 2.5 * n_fn), squeeze=False)
    fig.suptitle("Marginal Effect: Mean Final Loss vs Each Parameter",
                 color=PALETTE["text"], fontsize=11, y=1.01)

    for row_i, fn_name in enumerate(fn_names):
        fn_rows = by_fn[fn_name]
        for col_i, param in enumerate(params):
            ax = axes[row_i][col_i]
            vals    = sorted(set(r[param] for r in fn_rows))
            means, stds = [], []
            for v in vals:
                sub = [np.log1p(r["final_loss"]) for r in fn_rows if r[param] == v]
                means.append(np.mean(sub))
                stds.append(np.std(sub))

            means = np.array(means)
            stds  = np.array(stds)

            ax.plot(vals, means, color=PALETTE["accent1"], linewidth=1.8, marker="o",
                    markersize=4, zorder=3)
            ax.fill_between(vals, means - stds, means + stds,
                            color=PALETTE["accent1"], alpha=0.18)

            best_idx = int(np.argmin(means))
            ax.axvline(vals[best_idx], color=PALETTE["good"], linewidth=1,
                       linestyle="--", alpha=0.7)
            ax.text(vals[best_idx], ax.get_ylim()[0],
                    f" ★{vals[best_idx]:.4g}", color=PALETTE["good"],
                    fontsize=7, va="bottom")

            ax.set_xlabel(param_labels[param], fontsize=8)
            ax.set_ylabel("log(1+loss)" if col_i == 0 else "", fontsize=8)
            ax.set_title(fn_name if col_i == 1 else "", fontsize=8,
                         color=PALETTE["muted"])
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(out_dir, "marginal_effects.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Heatmaps ──────────────────────────────────────────────────────────────────

def _pivot(rows, xparam, yparam, agg="median"):
    xs = sorted(set(r[xparam] for r in rows))
    ys = sorted(set(r[yparam] for r in rows))
    grid = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vals = [r["final_loss"] for r in rows
                    if r[xparam] == x and r[yparam] == y]
            if vals:
                grid[i, j] = np.median(vals) if agg == "median" else np.mean(vals)
    return xs, ys, grid


def plot_heatmaps(rows, out_dir):
    by_fn   = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    param_pairs = [("eta", "alpha"), ("eta", "alpha_pull"), ("alpha", "alpha_pull")]
    fn_names    = [f for f in FN_ORDER if f in by_fn]

    for pair in param_pairs:
        xp, yp = pair
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        fig.suptitle(f"Median Final Loss — {xp} × {yp}  (per function)",
                     color=PALETTE["text"], fontsize=11, y=1.01)

        for idx, fn_name in enumerate(fn_names):
            ax  = axes[idx // 3][idx % 3]
            fn_rows = by_fn[fn_name]
            xs, ys, grid = _pivot(fn_rows, xp, yp)

            vmin = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1e-10
            vmax = np.nanmax(grid)
            if vmin <= 0 or vmax <= 0 or vmin >= vmax:
                norm = None
            else:
                norm = LogNorm(vmin=vmin, vmax=vmax)

            im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r",
                           norm=norm, origin="lower",
                           extent=[0, len(xs), 0, len(ys)])

            ax.set_xticks(np.arange(len(xs)) + 0.5)
            ax.set_yticks(np.arange(len(ys)) + 0.5)
            ax.set_xticklabels([f"{v:.4g}" for v in xs], rotation=45, fontsize=7)
            ax.set_yticklabels([f"{v:.4g}" for v in ys], fontsize=7)
            ax.set_xlabel(xp, fontsize=8)
            ax.set_ylabel(yp, fontsize=8)
            ax.set_title(fn_name, fontsize=9, color=PALETTE["muted"])

            # Star the best cell
            best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
            ax.text(best_idx[1] + 0.5, best_idx[0] + 0.5, "★",
                    ha="center", va="center", fontsize=12,
                    color=PALETTE["accent2"], fontweight="bold")

            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.tick_params(labelsize=6, colors=PALETTE["muted"])

        fig.tight_layout()
        path = os.path.join(out_dir, f"heatmap_{xp}_vs_{yp}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        print(f"  [saved] {path}")


# ── Aggregated rank plot ──────────────────────────────────────────────────────

def plot_ranked_configs(rows, out_dir, top_n=15):
    """
    For each unique (eta, alpha, alpha_pull) combination, compute the mean
    rank across functions (lower rank = better). Plot top_n configs.
    """
    by_fn   = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    # Rank within each function
    config_ranks = defaultdict(list)
    for fn_name, fn_rows in by_fn.items():
        sorted_rows = sorted(fn_rows, key=lambda r: r["final_loss"])
        for rank, r in enumerate(sorted_rows, 1):
            key = (r["eta"], r["alpha"], r["alpha_pull"])
            config_ranks[key].append(rank)

    # Mean rank across functions
    mean_ranks = {k: np.mean(v) for k, v in config_ranks.items()
                  if len(v) == len(by_fn)}  # only configs tested on ALL functions
    if not mean_ranks:
        print("  [WARN] No config was tested on every function — skipping rank plot.")
        return

    sorted_configs = sorted(mean_ranks.items(), key=lambda x: x[1])[:top_n]

    labels     = [f"η={c[0]:.4g}\nα={c[1]:.4g}\nαp={c[2]:.4g}" for c, _ in sorted_configs]
    mean_r     = [v for _, v in sorted_configs]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 5))
    colors  = [PALETTE["good"] if i == 0 else
               PALETTE["accent1"] if i < 3 else
               PALETTE["muted"] for i in range(len(mean_r))]
    bars = ax.bar(range(len(mean_r)), mean_r, color=colors, width=0.65, zorder=3)
    ax.set_xticks(range(len(mean_r)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean Rank Across Functions\n(lower = better)", fontsize=9)
    ax.set_title(f"Top-{top_n} Configs by Mean Cross-Function Rank", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)

    for bar, val in zip(bars, mean_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                color=PALETTE["text"])

    fig.tight_layout()
    path = os.path.join(out_dir, "ranked_configs.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [saved] {path}")

    # Print the best config
    best_cfg, best_rank = sorted_configs[0]
    print(f"\n  ★  OVERALL BEST CONFIG (mean rank {best_rank:.2f} across {len(by_fn)} functions)")
    print(f"     eta        = {best_cfg[0]}")
    print(f"     alpha      = {best_cfg[1]}")
    print(f"     alpha_pull = {best_cfg[2]}")
    print()


# ── Convergence rate plot ─────────────────────────────────────────────────────

def plot_convergence_rate(rows, out_dir):
    """Bar chart: fraction of configs that converged, per function."""
    by_fn = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    fn_names = [f for f in FN_ORDER if f in by_fn]
    rates    = [np.mean([r["converged"] for r in by_fn[f]]) * 100 for f in fn_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE["good"] if r >= 50 else
              PALETTE["accent3"] if r >= 20 else
              PALETTE["bad"] for r in rates]
    bars = ax.barh(fn_names, rates, color=colors, height=0.55, zorder=3)
    ax.set_xlabel("Convergence Rate (%)", fontsize=9)
    ax.set_title("Fraction of Configurations that Converged (final_loss < 1e-3)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", alpha=0.4)

    for bar, val in zip(bars, rates):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color=PALETTE["text"])

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_rate.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Per-function best config summary plot ────────────────────────────────────

def plot_best_per_function(rows, out_dir, top_n=5):
    """Dot plot: top-N configs per function, colored by loss magnitude."""
    by_fn = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)

    fn_names = [f for f in FN_ORDER if f in by_fn]
    fig, axes = plt.subplots(1, len(fn_names), figsize=(3.5 * len(fn_names), 5),
                             squeeze=False)
    fig.suptitle(f"Top-{top_n} Configs per Function", fontsize=11, y=1.01)

    for col, fn_name in enumerate(fn_names):
        ax     = axes[0][col]
        fn_rows = sorted(by_fn[fn_name], key=lambda r: r["final_loss"])[:top_n]
        losses  = [r["final_loss"] for r in fn_rows]
        labels  = [f"η={r['eta']:.4g}\nα={r['alpha']:.3g}\nαp={r['alpha_pull']:.3g}"
                   for r in fn_rows]

        cmap   = plt.cm.RdYlGn_r
        lmin, lmax = min(losses), max(losses)
        norm_vals  = [(l - lmin) / (lmax - lmin + 1e-30) for l in losses]

        for i, (label, loss, nv) in enumerate(zip(labels, losses, norm_vals)):
            color = cmap(nv * 0.8 + 0.1)
            ax.scatter([0], [top_n - i], color=color, s=200, zorder=3)
            ax.text(0.12, top_n - i, f"{loss:.3e}", va="center",
                    fontsize=7.5, color=PALETTE["text"])
            ax.text(-0.12, top_n - i, label, va="center", ha="right",
                    fontsize=6.5, color=PALETTE["muted"])

        ax.set_xlim(-1.0, 0.7)
        ax.set_ylim(0, top_n + 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(fn_name, fontsize=9)
        ax.axvline(0, color=PALETTE["border"], linewidth=0.8)

    fig.tight_layout()
    path = os.path.join(out_dir, "best_per_function.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [saved] {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze EASGD2 grid-search results from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",       default="optimization_results_2.csv",
                   help="Path to results CSV (default: optimization_results_2.csv)")
    p.add_argument("--top_n",     type=int, default=3,
                   help="Top-N configs to show/plot per function (default: 3)")
    p.add_argument("--out_dir",   default="./analysis_figures",
                   help="Output directory for figures (default: ./analysis_figures)")
    p.add_argument("--no_plots",  action="store_true",
                   help="Skip all figure generation, console output only")
    return p.parse_args()


def main():
    args   = parse_args()
    rows   = load_csv(args.csv)

    # ── Console output ────────────────────────────────────────────────────────
    print_summary(rows, top_n=args.top_n)
    print_correlation(rows)

    if args.no_plots:
        return

    # ── Figures ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving figures to  {args.out_dir}/\n")

    plot_marginals(rows, args.out_dir)
    plot_heatmaps(rows, args.out_dir)
    plot_ranked_configs(rows, args.out_dir, top_n=15)
    plot_convergence_rate(rows, args.out_dir)
    plot_best_per_function(rows, args.out_dir, top_n=args.top_n)

    print(f"\nDone. All figures saved to  {args.out_dir}/")


if __name__ == "__main__":
    main()
