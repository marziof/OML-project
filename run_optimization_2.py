"""
run_optimization_2.py
─────────────────────
Refined grid search over alpha, alpha_pull for EASGD2 across ALL
benchmark functions. Search ranges narrowed from heatmap analysis:

  alpha      : [0.325, 1.0]    - linear     (alpha ≤ 0.1 is consistently bad)
  alpha_pull : [0.325, 1.0]    - linear     (outside this band loses ~10×)
  beta       : [0, 4]          - linear     


Locked hyperparameters (updated from heatmap evidence):
  num_workers = 4    (3–4 workers optimal; 1–2 unstable at any eta)
  num_epochs  = 10000
  eta         = 1e-3


Results are appended to  optimization_results_2.csv  so you can resume a
partial run without losing data.

Usage
-----
# Full refined grid
python run_optimization_2.py

# Quick smoke-test (3×3)
python run_optimization_2.py --quick

# Override locked params if needed
python run_optimization_2.py --num_workers 4 --beta 5 --num_epochs 3000

# Just re-print the summary from an existing CSV
python run_optimization_2.py --summary_only --top_n 5
"""

import argparse
import csv
import itertools
import os
import time
from collections import defaultdict


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locked hyperparameters (everything except the 3 we sweep) ────────────────
# Justification from heatmaps:
#   num_workers=4  : 3–4 workers are robust; 1–2 diverge at any eta > 0.001
#   beta=2.0       : beta 2–5 yields best losses; beta=10 previously locked was
#                    too aggressive — high beta + eta > 0.002 -> catastrophic divergence
#   num_epochs=2000: unchanged

LOCKED = dict(
    eta            = 0.001,
    num_workers    = 4,
    num_epochs     = 10000,
    eta_sgd        = 0.001,
    num_epochs_sgd = 10000,
    seed           = 42,
)

# ── Search grids ──────────────────────────────────────────────────────────────

BETA_GRID        = [round(v, 3) for v in np.linspace(4, 6, 5).tolist()]
ALPHA_GRID      = [round(v, 3) for v in np.linspace(0.7, 1.0, 6).tolist()] 
ALPHA_PULL_GRID = [round(v, 3) for v in np.linspace(0.0, 0.2, 6).tolist()]

# ── Benchmark functions ───────────────────────────────────────────────────────

def rosenbrock_fn(theta):
    x, y = theta
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gd(theta):
    x, y = theta
    dfdx = -2*(1 - x) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return np.clip(np.array([dfdx, dfdy]), -10.0, 10.0)

def rastrigin_fn(theta):
    A = 10
    return A*2 + sum(x**2 - A*np.cos(2*np.pi*x) for x in theta)

def rastrigin_gd(theta):
    A = 10
    return np.array([2*x + 2*np.pi*A*np.sin(2*np.pi*x) for x in theta])

def himmelblau_fn(theta):
    x, y = theta
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_gd(theta):
    x, y = theta
    dfdx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dfdy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.clip(np.array([dfdx, dfdy]), -10.0, 10.0)

def ackley_fn(theta):
    x, y = theta
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
            - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
            + np.e + 20)

def ackley_gd(theta):
    x, y = theta
    r = np.sqrt(0.5*(x**2+y**2))
    exp1 = np.exp(-0.2*r)
    exp2 = np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
    dfdx = (20*exp1*0.2*(0.5*x/r if r > 1e-10 else 0)
            + exp2*np.pi*np.sin(2*np.pi*x))
    dfdy = (20*exp1*0.2*(0.5*y/r if r > 1e-10 else 0)
            + exp2*np.pi*np.sin(2*np.pi*y))
    return np.clip(np.array([dfdx, dfdy]), -10.0, 10.0)

FUNCTIONS = {
    "rosenbrock":      (rosenbrock_fn, rosenbrock_gd, [-2.0, 2.0],   2.0),
    "rastrigin":       (rastrigin_fn,  rastrigin_gd,  [3.0,  3.0],   2.0),
    "himmelblau":      (himmelblau_fn, himmelblau_gd, [-3.0, -3.0],  4.0),
}

# ── EASGD2 ────────────────────────────────────────────────────────────────────

def easgd_2(grad, fn, start, num_workers, eta, alpha, alpha_pull, beta, num_epochs, rng):
    x_center = np.asarray(start, dtype=float)
    workers  = np.array([x_center + rng.randn(2) for _ in range(num_workers)])
    master_traj = [x_center.copy()]

    for e in range(num_epochs):
        if e % 50 == 0:
            losses = np.array([fn(workers[i]) for i in range(num_workers)])
            # Numerically stable softmax
            shifted = -beta * losses
            shifted -= shifted.max()
            w = np.exp(shifted) / np.exp(shifted).sum()

            workers_temp = workers.copy()
            workers  = workers - alpha_pull * (w @ (workers - x_center))
            x_center = (1 - alpha) * x_center + alpha * (w @ workers_temp)

        g = np.array([grad(workers[i]) for i in range(num_workers)])
        workers = workers - eta * g
        master_traj.append(x_center.copy())

    return x_center, master_traj

# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_FILE = "optimization_results_2.csv"
CSV_COLS = [
    "function", "eta", "alpha", "alpha_pull",
    "num_workers", "beta", "num_epochs",
    "final_loss", "best_loss", "converged", "elapsed_s",
]

def load_done(csv_file):
    """Return a set of (function, eta, alpha, alpha_pull) already in the CSV."""
    done = set()
    if not os.path.isfile(csv_file):
        return done
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["function"],
                      float(row["eta"]),
                      float(row["alpha"]),
                      float(row["alpha_pull"])))
    return done

def append_row(csv_file, row: dict):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ── Main grid search ──────────────────────────────────────────────────────────

def run_grid(locked, beta_grid, alpha_grid, alpha_pull_grid, csv_file, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    done = load_done(csv_file)
    combos = list(itertools.product(
        FUNCTIONS.keys(), beta_grid, alpha_grid, alpha_pull_grid
    ))
    total   = len(combos)
    skipped = sum(1 for (fn_name, eta, alpha, ap) in combos
                  if (fn_name, eta, alpha, ap) in done)

    print(f"\nGrid search — EASGD2  |  locked: {locked}")
    print(f"beta × alpha × alpha_pull × functions = "
          f"{len(beta_grid)} × {len(alpha_grid)} × {len(alpha_pull_grid)} × {len(FUNCTIONS)}"
          f" = {total} runs  ({skipped} already done, resuming)\n")

    current = 0
    for fn_name, beta, alpha, alpha_pull in combos:
        current += 1
        key = (fn_name, beta, alpha, alpha_pull)

        if key in done:
            print(f"  [{current:>4}/{total}] SKIP  {fn_name}  beta={beta}  alpha={alpha}  alpha_pull={alpha_pull}")
            continue

        fn, grad_fn, minimum, _ = FUNCTIONS[fn_name]
        rng = np.random.RandomState(locked["seed"])

        t0 = time.perf_counter()
        x_end, master_traj = easgd_2(
            grad       = grad_fn,
            fn         = fn,
            start      = minimum,           # start from the known minimum for fair comparison
            num_workers= locked["num_workers"],
            beta        = beta,
            alpha      = alpha,
            alpha_pull = alpha_pull,
            eta       = locked["eta"],
            num_epochs = locked["num_epochs"],
            rng        = rng,
        )
        elapsed = time.perf_counter() - t0

        losses      = [fn(p) for p in master_traj]
        final_loss  = losses[-1]
        best_loss   = min(losses)
        converged   = int(final_loss < 1e-3)

        print(f"  [{current:>4}/{total}]  {fn_name:<16}  beta={beta:<8}  "
              f"alpha={alpha:<6}  alpha_pull={alpha_pull:<6}  "
              f"loss={final_loss:.4e}  best={best_loss:.4e}  "
              f"conv={'YES' if converged else ' no'}  {elapsed:.1f}s")

        append_row(csv_file, {
            "function":    fn_name,
            "beta":         beta,
            "alpha":       alpha,
            "alpha_pull":  alpha_pull,
            "num_workers": locked["num_workers"],
            "eta":        locked["eta"],
            "num_epochs":  locked["num_epochs"],
            "final_loss":  round(final_loss, 8),
            "best_loss":   round(best_loss, 8),
            "converged":   converged,
            "elapsed_s":   round(elapsed, 3),
        })

    print(f"\nAll runs written to  {csv_file}")
    return csv_file

# ── Summary / best-config report ─────────────────────────────────────────────

def print_summary(csv_file, top_n=5):
    import csv as _csv
    from collections import defaultdict

    rows = []
    with open(csv_file, newline="") as f:
        rows = list(_csv.DictReader(f))

    if not rows:
        print("No results yet.")
        return

    print("\n" + "═"*70)
    print(f"  TOP-{top_n} CONFIGS PER FUNCTION  (by final_loss)")
    print("═"*70)

    by_fn = {}
    for r in rows:
        by_fn.setdefault(r["function"], []).append(r)

    for fn_name in sorted(by_fn):
        fn_rows = sorted(by_fn[fn_name], key=lambda r: float(r["final_loss"]))
        print(f"\n  {fn_name}")
        print(f"  {'beta':<10} {'alpha':<8} {'alpha_pull':<12} {'final_loss':<14} {'best_loss':<14} conv")
        print(f"  {'-'*8:<10} {'-'*6:<8} {'-'*10:<12} {'-'*12:<14} {'-'*12:<14} ----")
        for r in fn_rows[:top_n]:
            print(f"  {r['beta']:<10} {r['alpha']:<8} {r['alpha_pull']:<12} "
                  f"{float(r['final_loss']):<14.4e} {float(r['best_loss']):<14.4e} "
                  f"{'YES' if r['converged']=='1' else ' no'}")

    # --- avg final_loss across functions ---
    combo_losses = defaultdict(list)
    for r in rows:
        key = (r["beta"], r["alpha"], r["alpha_pull"])
        combo_losses[key].append(float(r["final_loss"]))

    combo_avg = sorted(
        ((key, sum(l) / len(l)) for key, l in combo_losses.items()),
        key=lambda x: x[1]
    )

    print("\n" + "═"*70)
    print("  GLOBAL TOP CONFIGS  (averaged final_loss across all functions)")
    print("═"*70)
    print(f"  {'beta':<10} {'alpha':<8} {'alpha_pull':<12} {'avg_final_loss':<14} {'n_fns'}")
    print(f"  {'-'*8:<10} {'-'*6:<8} {'-'*10:<12} {'-'*12:<14} ------")
    for (beta, alpha, alpha_pull), avg_loss in combo_avg[:top_n]:
        n = len(combo_losses[(beta, alpha, alpha_pull)])
        print(f"  {beta:<10} {alpha:<8} {alpha_pull:<12} {avg_loss:<14.4e} {n}")

    # --- rank aggregation across functions ---
    combo_ranks = defaultdict(list)
    for fn_name, fn_rows in by_fn.items():
        fn_rows_sorted = sorted(fn_rows, key=lambda r: float(r["final_loss"]))
        for rank, r in enumerate(fn_rows_sorted, start=1):
            key = (r["beta"], r["alpha"], r["alpha_pull"])
            combo_ranks[key].append(rank)

    combo_avg_rank = sorted(
        ((key, sum(ranks) / len(ranks)) for key, ranks in combo_ranks.items()),
        key=lambda x: x[1]
    )

    n_fns = len(by_fn)
    print("\n" + "═"*70)
    print("  GLOBAL TOP CONFIGS  (avg rank across all functions)")
    print("═"*70)
    print(f"  {'beta':<10} {'alpha':<8} {'alpha_pull':<12} {'avg_rank':<12} {'n_fns'}")
    print(f"  {'-'*8:<10} {'-'*6:<8} {'-'*10:<12} {'-'*10:<12} ------")
    for (beta, alpha, alpha_pull), avg_rank in combo_avg_rank[:top_n]:
        n = len(combo_ranks[(beta, alpha, alpha_pull)])
        print(f"  {beta:<10} {alpha:<8} {alpha_pull:<12} {avg_rank:<12.2f} {n}/{n_fns}")
    print()
    
# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search alpha/alpha_pull/eta for EASGD2 across all functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--quick", action="store_true",
                   help="Use a smaller grid for a fast smoke-test")
    p.add_argument("--summary_only", action="store_true",
                   help="Skip training, just print summary from existing CSV")
    p.add_argument("--top_n", type=int, default=3,
                   help="How many top configs to show per function (default: 3)")
    p.add_argument("--csv_file",    default=CSV_FILE)
    p.add_argument("--figures_dir", default="./figures_grid")

    # Allow overriding locked params
    p.add_argument("--num_workers", type=int,   default=LOCKED["num_workers"])
    p.add_argument("--eta",        type=float, default=LOCKED["eta"])
    p.add_argument("--num_epochs",  type=int,   default=LOCKED["num_epochs"])
    p.add_argument("--seed",        type=int,   default=LOCKED["seed"])
    return p.parse_args()


def main():
    args = parse_args()

    locked = dict(
        num_workers = args.num_workers,
        eta        = args.eta,
        num_epochs  = args.num_epochs,
        seed        = args.seed,
    )

    if args.quick:
        beta_grid        = [0, 2, 5]   # within refined safe zone
        alpha_grid      = [0.325, 0.6625, 1.0]       # low / mid / high of new range
        alpha_pull_grid = [0.45, 0.65, 0.85]          # low / mid / high of new range
        print("[--quick mode: 3x3x3 grid within refined ranges]")
    else:
        beta_grid        = BETA_GRID
        alpha_grid      = ALPHA_GRID
        alpha_pull_grid = ALPHA_PULL_GRID

    if not args.summary_only:
        run_grid(locked, beta_grid, alpha_grid, alpha_pull_grid,
                 args.csv_file, args.figures_dir)

    print_summary(args.csv_file, top_n=args.top_n)


if __name__ == "__main__":
    main()
