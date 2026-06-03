"""
run_optimization.py
───────────────────
Run EASGD, EASGD2, or plain SGD on any 2-D benchmark function.

Usage examples
--------------
# Default: EASGD on rosenbrock
python run_optimization.py

# Choose optimizer and function
python run_optimization.py --optimizer easgd2 --function himmelblau

# Full hyperparameter control
python run_optimization.py --optimizer easgd \
    --function rastrigin \
    --num_workers 6 \
    --eta 0.001 \
    --rho 0.01 \
    --num_epochs 1000 \
    --start -1.0 2.0 \
    --plot_range 3.0

# Run all three optimizers on one function for comparison
python run_optimization.py --function rosenbrock --compare_all

Available functions
-------------------
  rosenbrock  sphere  rastrigin  himmelblau
  styblinski_tang  ackley

Available optimizers
--------------------
  easgd  easgd2  sgd
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Gradient definitions ──────────────────────────────────────────────────────

def sphere_fn(theta):
    x, y = theta
    return x**2 + y**2

def sphere_gd(theta):
    x, y = theta
    return np.array([2*x, 2*y])

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

def styblinski_tang_fn(theta):
    return 0.5*sum(x**4 - 16*x**2 + 5*x for x in theta) + 80

def styblinski_tang_gd(theta):
    return np.array([0.5*(4*x**3 - 32*x + 5) for x in theta])

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
    "sphere":          (sphere_fn,          sphere_gd,          [0.0, 0.0],    2.0),
    "rosenbrock":      (rosenbrock_fn,       rosenbrock_gd,      [1.0, 1.0],    2.0),
    "rastrigin":       (rastrigin_fn,        rastrigin_gd,       [0.0, 0.0],    2.0),
    "himmelblau":      (himmelblau_fn,       himmelblau_gd,      [3.0, 2.0],    4.0),
    "styblinski_tang": (styblinski_tang_fn,  styblinski_tang_gd, [-2.9, -2.9],  4.0),
    "ackley":          (ackley_fn,           ackley_gd,          [0.0, 0.0],    4.0),
}

# Default hyperparameters per (optimizer, function) — used when --compare_all
DEFAULTS = {
    "easgd": dict(num_workers=4, eta=0.005, rho=50.0, num_epochs=2000, beta=None,
                  alpha=None, alpha_pull=None),
    "easgd2": dict(num_workers=4, eta=0.001, rho=None, num_epochs=2000, beta=10,
                   alpha=0.09, alpha_pull=0.5),
    "sgd": dict(eta=0.001, num_epochs=2000),
}


# ── Optimizers ────────────────────────────────────────────────────────────────

def easgd(grad, start, num_workers=4, eta=0.01, rho=0.1, num_epochs=300):
    alpha = eta * rho
    x_center = np.asarray(start, dtype=float)
    workers  = [x_center + np.random.randn(2) for _ in range(num_workers)]
    master_traj  = [x_center.copy()]
    worker_trajs = [[w.copy()] for w in workers]

    for e in range(num_epochs):
        for i in range(num_workers):
            if e % 50 == 0:
                workers[i] = workers[i] - alpha*(workers[i] - x_center)
                x_center   = x_center  + alpha*(workers[i] - x_center)
            g = grad(workers[i])
            workers[i] = workers[i] - eta*g
            worker_trajs[i].append(workers[i].copy())
        master_traj.append(x_center.copy())

    return x_center, master_traj, worker_trajs


def easgd_2(grad, fn, start, num_workers=4, eta=0.01, alpha=0.5,
            alpha_pull=0.5, beta=0, num_epochs=300):
    x_center = np.asarray(start, dtype=float)
    workers  = np.array([x_center + np.random.randn(2) for _ in range(num_workers)])
    master_traj  = [x_center.copy()]
    worker_trajs = [[w.copy()] for w in workers]

    for e in range(num_epochs):
        if e % 50 == 0:
            losses = np.array([fn(workers[i]) for i in range(num_workers)])
            w = np.exp(-beta*losses) / np.sum(np.exp(-beta*losses))
            workers_temp = workers.copy()
            workers = workers - alpha_pull * w @ (workers - x_center)
            x_center = (1 - alpha)*x_center + alpha*(w @ workers_temp)

        g = np.array([grad(workers[i]) for i in range(num_workers)])
        workers = workers - eta*g
        for i in range(num_workers):
            worker_trajs[i].append(workers[i].copy())
        master_traj.append(x_center.copy())

    return x_center, master_traj, worker_trajs


def plain_gd(grad, start, eta=0.01, num_epochs=300):
    x = np.asarray(start, dtype=float)
    traj = [x.copy()]
    for _ in range(num_epochs):
        x = x - eta*grad(x)
        traj.append(x.copy())
    return x, traj


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_path(ax, trajectory, color, label):
    pts = np.array(trajectory)
    tx, ty = pts[:, 0], pts[:, 1]
    ax.plot(tx, ty, color=color, linewidth=1.5, alpha=0.8, label=label)
    ax.scatter(tx[0],  ty[0],  s=100, color="lime",  zorder=5)
    ax.scatter(tx[-1], ty[-1], s=100, color=color,   zorder=5, marker="*")
    step = max(1, len(tx) // 15)
    for i in range(0, len(tx) - step, step):
        ax.annotate("",
            xy=(tx[i+step], ty[i+step]),
            xytext=(tx[i], ty[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )


def plot_trajectories(fn, master_traj, gd_traj, worker_trajs,
                      minimum, fn_name, opt_name, r, save_path):
    xs = np.linspace(-r, r, 400)
    ys = np.linspace(-r, r*1.5, 400)
    Z  = np.array([[fn(np.array([xi, yj])) for xi in xs] for yj in ys])

    fig, ax = plt.subplots(figsize=(8, 7))
    cp = ax.contourf(xs, ys, np.log1p(Z), levels=60, cmap="viridis")
    plt.colorbar(cp, ax=ax, label="log(1 + Loss)")

    labeled_worker = False
    for wt in worker_trajs:
        draw_path(ax, wt, color="royalblue",
                  label="Worker" if not labeled_worker else "_nolegend_")
        labeled_worker = True

    draw_path(ax, gd_traj,     color="tomato", label="Plain GD")
    draw_path(ax, master_traj, color="white",  label=f"{opt_name.upper()} master")

    ax.scatter([], [], s=100, color="lime", label="Start")
    ax.scatter(minimum[0], minimum[1], s=200, color="yellow",
               marker="*", zorder=6, label="Global min")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Plain GD vs {opt_name.upper()} — {fn_name}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_loss_curves(master_traj, gd_traj, fn, fn_name, opt_name, save_path):
    master_losses = [fn(p) for p in master_traj]
    gd_losses     = [fn(p) for p in gd_traj]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(master_losses, label=f"{opt_name.upper()} master", color="white",
                linewidth=1.5)
    ax.semilogy(gd_losses,     label="Plain GD",    color="tomato", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Loss curves — {fn_name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.tick_params(colors="gray"); ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray"); ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(opt_name, fn_name, fn, grad_fn, minimum, r, start, args, figures_dir):
    print(f"\n[{opt_name.upper()} | {fn_name}]")

    if opt_name == "easgd":
        x_end, master_traj, worker_trajs = easgd(
            grad_fn, start,
            num_workers=args.num_workers,
            eta=args.eta,
            rho=args.rho,
            num_epochs=args.num_epochs,
        )
    elif opt_name == "easgd2":
        x_end, master_traj, worker_trajs = easgd_2(
            grad_fn, fn, start,
            num_workers=args.num_workers,
            eta=args.eta,
            alpha=args.alpha,
            alpha_pull=args.alpha_pull,
            beta=args.beta,
            num_epochs=args.num_epochs,
        )
    else:  # sgd only
        master_traj = None
        worker_trajs = []

    _, gd_traj = plain_gd(grad_fn, start, eta=args.eta_sgd,
                           num_epochs=args.num_epochs_sgd)

    final_loss_sgd = fn(gd_traj[-1])
    print(f"  Plain GD  final loss: {final_loss_sgd:.6f}")

    if master_traj is not None:
        final_loss_opt = fn(master_traj[-1])
        print(f"  {opt_name.upper():8s} final loss: {final_loss_opt:.6f}")

        tag = f"{opt_name}_{fn_name}_numwork{args.num_workers}_eta{args.eta}_alpha{args.alpha}_alphap{args.alpha_pull}_beta{args.beta}"
        plot_trajectories(fn, master_traj, gd_traj, worker_trajs,
                          minimum, fn_name, opt_name, r,
                          os.path.join(figures_dir, f"{tag}_trajectory.png"))
        plot_loss_curves(master_traj, gd_traj, fn, fn_name, opt_name,
                         os.path.join(figures_dir, f"{tag}_loss.png"))
    # Define the CSV file name
    csv_file = "optimization_results.csv"

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(csv_file)

    # Append the hyperparameters and the final loss to the CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
    
        # Write header only if the file is being created for the first time
        if not file_exists:
            writer.writerow(["num_workers", "eta", "beta", "alpha", "alpha_pull", "loss"])
            
        # Write the current run's data (assuming 'args' holds your command-line arguments)
        writer.writerow([args.num_workers, args.eta, args.beta, args.alpha, args.alpha_pull, final_loss_opt])


def build_args_from_defaults(base_args, opt_name):
    """Copy base_args and fill missing hyperparams from DEFAULTS."""
    import copy
    a = copy.copy(base_args)
    d = DEFAULTS[opt_name]
    if not hasattr(a, 'num_workers') or a.num_workers is None:
        a.num_workers = d.get("num_workers", 4)
    if not hasattr(a, 'eta') or a.eta is None:
        a.eta = d.get("eta", 0.01)
    if not hasattr(a, 'rho') or a.rho is None:
        a.rho = d.get("rho", 1.0)
    if not hasattr(a, 'beta') or a.beta is None:
        a.beta = d.get("beta", 0)
    if not hasattr(a, 'alpha') or a.alpha is None:
        a.alpha = d.get("alpha", 0.5)
    if not hasattr(a, 'alpha_pull') or a.alpha_pull is None:
        a.alpha_pull = d.get("alpha_pull", 0.5)
    return a


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run EASGD / EASGD2 / SGD on 2-D benchmark functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--optimizer",    default="easgd",
                   choices=["easgd", "easgd2", "sgd"],
                   help="Optimizer to run (default: easgd)")
    p.add_argument("--function",     default="rosenbrock",
                   choices=list(FUNCTIONS.keys()),
                   help="Benchmark function (default: rosenbrock)")
    p.add_argument("--compare_all",  action="store_true",
                   help="Run all three optimizers for comparison")

    # Starting point
    p.add_argument("--start", nargs=2, type=float, default=[-0.5, 1.5],
                   metavar=("X0", "Y0"),
                   help="Starting point (default: -0.5 1.5)")
    p.add_argument("--plot_range", type=float, default=None,
                   help="Half-width of plot axes (default: auto per function)")

    # Shared hyperparams
    p.add_argument("--num_epochs",   type=int,   default=2000)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--eta",          type=float, default=0.005,
                   help="Learning rate for the chosen optimizer")

    # SGD baseline (always plotted alongside)
    p.add_argument("--eta_sgd",      type=float, default=0.001,
                   help="Learning rate for the plain GD baseline")
    p.add_argument("--num_epochs_sgd", type=int, default=2000,
                   help="Epochs for the plain GD baseline")

    # EASGD-specific
    p.add_argument("--rho",          type=float, default=50.0,
                   help="EASGD coupling strength")

    # EASGD2-specific
    p.add_argument("--beta",         type=float, default=10.0,
                   help="EASGD2 temperature beta")
    p.add_argument("--alpha",        type=float, default=0.09,
                   help="EASGD2 center update rate")
    p.add_argument("--alpha_pull",   type=float, default=0.5,
                   help="EASGD2 worker pull strength")

    p.add_argument("--figures_dir",  default="./figures",
                   help="Directory for saved figures (default: ./figures)")
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.figures_dir, exist_ok=True)

    fn, grad_fn, minimum, default_r = FUNCTIONS[args.function]
    r     = args.plot_range if args.plot_range is not None else default_r
    start = args.start

    if args.compare_all:
        for opt_name in ["easgd", "easgd2", "sgd"]:
            a = build_args_from_defaults(args, opt_name)
            run_one(opt_name, args.function, fn, grad_fn, minimum, r,
                    start, a, args.figures_dir)
    else:
        run_one(args.optimizer, args.function, fn, grad_fn, minimum, r,
                start, args, args.figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
