import numpy as np 
import matplotlib.pyplot as plt

# ── Synchronous EASGD ─────────────────────────────────────────────────────────
def seasgd(grad,fn, theta,num_workers=4, eta=0.01, alpha=0.5,alpha_pull = 0.5,beta=0, num_epochs=300, dim=2):

    x_center = np.asarray(theta, dtype=float)
    workers = [
        x_center + 1.0 * np.random.randn(dim)
        for _ in range(num_workers)
    ]

    master_trajectory  = [x_center.copy()]
    worker_trajectories = [[w.copy()] for w in workers]  # one list per worker

    for e in range(num_epochs):
        if e % 50 ==0:
            losses = np.array([fn(workers[i]) for i in range(num_workers)])
            scaled = -beta * losses
            scaled -= scaled.max()  # subtract max for numerical stability
            exp_scaled = np.exp(scaled)
            w = exp_scaled / exp_scaled.sum()
            workers = np.array(workers)
            workers_temp = workers
            workers = workers - alpha_pull*w @ (workers - np.array(x_center))
            x_center = (1-alpha)*x_center + alpha * (w @ workers_temp)

            
        g = np.array([grad(workers[i]) for i in range(num_workers)])
        workers = (workers - eta   * g)
        for i in range(num_workers):
            worker_trajectories[i].append(workers[i].copy())  # record after each step

        master_trajectory.append(x_center.copy())

    return x_center, master_trajectory, worker_trajectories

# ── Plain Gradient Descent ────────────────────────────────────────────────────
def plain_gd(grad, x_init, eta=0.01, num_epochs=300):
    x = np.asarray(x_init, dtype=float)

    trajectory = [x.copy()]
    for _ in range(num_epochs):
        x = x - eta * grad(x)
        trajectory.append(x.copy())

    return x, trajectory


# ── Vanilla EASGD ────────────────────────────────────────────────────
# def easgd(grad, theta,num_workers=4, eta=0.01, rho=0.1, num_epochs=300, dim=2):
#     alpha = eta * rho

#     x_center = np.asarray(theta,dtype=float)
#     workers = [
#         x_center + 1 * np.random.randn(dim)
#         for _ in range(num_workers)
#     ]

#     master_trajectory  = [x_center.copy()]
#     worker_trajectories = [[w.copy()] for w in workers]  # one list per worker


#     for e in range(num_epochs):
#         for i in range(num_workers):
#             if e % 50 ==0:
#                 workers[i] = (workers[i]
#                             - alpha * (workers[i] - x_center))
#                 x_center    = x_center + alpha*(workers[i] - x_center)

#             g = grad(workers[i])
#             workers[i] = (workers[i]
#                             - eta   * g)
#             worker_trajectories[i].append(workers[i].copy())  # record after each step

#         master_trajectory.append(x_center.copy())

#     return x_center, master_trajectory, worker_trajectories

def easgd(grad, theta, num_workers=4, eta=0.01, rho=0.1, num_epochs=300, tau=50, dim=2):
    alpha = eta * rho
    beta = num_workers * alpha  # paper's constraint: β = p·α

    x_center = np.asarray(theta, dtype=float)
    workers = [x_center + np.random.randn(dim) for _ in range(num_workers)]

    master_trajectory = [x_center.copy()]
    worker_trajectories = [[w.copy()] for w in workers]

    for e in range(num_epochs):
        # gradient + elastic pull together (Eq. 5)
        for i in range(num_workers):
            g = grad(workers[i])
            workers[i] = workers[i] - eta * g - alpha * (workers[i] - x_center)
            worker_trajectories[i].append(workers[i].copy())

        # master update every τ steps (Eq. 6)
        if e % tau == 0:
            x_center = (1 - beta) * x_center + beta * np.mean(workers, axis=0)

        master_trajectory.append(x_center.copy())

    return x_center, master_trajectory, worker_trajectories

def seasgd_fixed_param(grad, fn, theta, workers_init, eta=0.01, alpha=0.5, alpha_pull=0.5,
            beta=0, num_epochs=300, dim=2):

    x_center = np.asarray(theta, dtype=float)
    workers = [np.asarray(w, dtype=float).copy() for w in workers_init]  # don't mutate the originals

    master_trajectory   = [x_center.copy()]
    worker_trajectories = [[w.copy()] for w in workers]

    for e in range(num_epochs):
        if e % 50 == 0:
            losses = np.array([fn(workers[i]) for i in range(len(workers))])
            scaled = -beta * losses
            scaled -= scaled.max()
            exp_scaled = np.exp(scaled)
            w = exp_scaled / exp_scaled.sum()
            workers = np.array(workers)
            workers_temp = workers.copy()
            workers = workers - alpha_pull * w @ (workers - np.array(x_center))
            x_center = (1 - alpha) * x_center + alpha * (w @ workers_temp)

        g = np.array([grad(workers[i]) for i in range(len(workers))])
        workers = workers - eta * g
        for i in range(len(workers)):
            worker_trajectories[i].append(workers[i].copy())

        master_trajectory.append(x_center.copy())

    return x_center, master_trajectory, worker_trajectories

def noisy_grad(grad_fn, x, noise_scale=0.1):
    return grad_fn(x) + noise_scale * np.random.randn(len(x))


def plot_master_vs_sgd(
    fn,
    sgd_traj,
    center_traj,
    xlim=(-5, 5),
    ylim=(-5, 5),
    title=""
):

    xs = np.linspace(*xlim, 400)
    ys = np.linspace(*ylim, 400)

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))

    sgd_traj = np.asarray(sgd_traj)
    center_traj = np.asarray(center_traj)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        X, Y,
        np.log1p(Z),
        levels=50
    )

    plt.colorbar(label="log(1 + loss)")

    step = 20

    plt.plot(
        sgd_traj[::step, 0],
        sgd_traj[::step, 1],
        'r-o',
        markersize=3,
        label="SGD"
    )
    plt.plot(
        sgd_traj[0, 0],
        sgd_traj[0, 1],
        'ro',
        ms=10
    )

    # EASGD master
    plt.plot(
        center_traj[:, 0],
        center_traj[:, 1],
        'w-',
        lw=2,
        label="SEASGD master"
    )

    plt.plot(
        center_traj[-1, 0],
        center_traj[-1, 1],
        'w*',
        ms=14
    )

    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def benchmark(
    grad_fn,
    fn,
    easgd2_kwargs,
    easgd_kwargs,
    sgd_kwargs,
    n_inits=50,
    

):

    sgd_losses = []
    easgd_losses = []
    easgd2_losses = []

    for seed in range(n_inits):

        rng = np.random.default_rng(seed)

        x0 = rng.uniform(-5, 5, size=2)

        # SGD
        sgd_final, _ = plain_gd(
            grad_fn,
            x_init=x0.copy(),
            eta=sgd_kwargs["eta"],
            num_epochs=sgd_kwargs["num_epochs"]
        )

        # Original EASGD
        easgd_final, _, _ = easgd(
            grad_fn,
            theta=x0.copy(),
            num_workers=easgd_kwargs["num_workers"],
            eta=easgd_kwargs["eta"],
            rho=easgd_kwargs["rho"],
            num_epochs=easgd_kwargs["num_epochs"],
            dim=easgd_kwargs["dim"]
        )

        # EASGD_2
        easgd2_final, _, _ = seasgd(
            grad_fn,
            fn,
            theta=x0.copy(),
            eta = easgd2_kwargs["eta"],
            num_workers=easgd2_kwargs["num_workers"],
            alpha=easgd2_kwargs["alpha"],
            alpha_pull=easgd2_kwargs["alpha_pull"],
            beta = easgd2_kwargs["beta"],
            num_epochs=easgd2_kwargs["num_epochs"],
            dim=easgd2_kwargs["dim"]
        )

        sgd_losses.append(fn(sgd_final))
        easgd_losses.append(fn(easgd_final))
        easgd2_losses.append(fn(easgd2_final))

    return (
        np.array(sgd_losses),
        np.array(easgd_losses),
        np.array(easgd2_losses)
    )

def plot_himmelblau_trajectories(
    sgd_traj,
    master_traj,
    worker_trajs,
    xlim=(-6, 6),
    ylim=(-6, 6)
):

    # Himmelblau landscape
    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)

    X, Y = np.meshgrid(x, y)

    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    plt.figure(figsize=(10, 8))

    # Contours
    plt.contourf(
        X,
        Y,
        np.log1p(Z),
        levels=60,
        alpha=0.8
    )

    plt.colorbar(label="log(1 + f(x,y))")

    # SGD trajectory
    sgd = np.asarray(sgd_traj)

    plt.plot(
        sgd[:, 0],
        sgd[:, 1],
        color="red",
        linewidth=3,
        label="SGD"
    )

    plt.scatter(
        sgd[0, 0],
        sgd[0, 1],
        color="red",
        marker="o",
        s=80
    )

    plt.scatter(
        sgd[-1, 0],
        sgd[-1, 1],
        color="red",
        marker="*",
        s=200
    )

    # Master trajectory
    master = np.asarray(master_traj)

    step = 100
    plt.plot(
        master[::step, 0],
        master[::step, 1],
        color="white",
        linewidth=3,
        label="Master"
    )

    plt.scatter(
        master[-1, 0],
        master[-1, 1],
        color="white",
        marker="*",
        s=200
    )
    
    # Workers
    for i, worker in enumerate(worker_trajs):

        worker = np.asarray(worker)

        plt.plot(
            worker[::step, 0],
            worker[::step, 1],
            alpha=0.6,
            linewidth=1
        )
        plt.scatter(
            worker[-1, 0],
            worker[-1, 1],
            s=40
        )

    # Known Himmelblau minima
    minima = np.array([
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126]
    ])

    plt.scatter(
        minima[:, 0],
        minima[:, 1],
        marker="*",
        s=200,
        color="yellow",
        label="Global minima"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Himmelblau Function: SGD vs EASGD")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rosenbrock_trajectories(
    master_trajs,      # dict {beta: master_traj}
    worker_trajs_dict, # dict {beta: worker_trajs}
    xlim=(-2, 2),
    ylim=(-1, 3)
):
    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, np.log1p(Z), levels=60, alpha=0.8)
    plt.colorbar(label="log(1 + f(x,y))")

    colors = plt.cm.cool(np.linspace(0, 1, len(master_trajs)))

    step = 50
    for i, (color, (beta, master_traj)) in enumerate(zip(colors, master_trajs.items())):
        
        # Seed markers only, no trajectory lines
        for j, worker in enumerate(worker_trajs_dict[beta]):
            worker = np.asarray(worker)
            plt.scatter(worker[0, 0], worker[0, 1],
                        color='red', marker="X", s=50, zorder=5, label="Worker start" if (i == 0 and j == 0) else None)
        master = np.asarray(master_traj)
        plt.plot(master[::step, 0], master[::step, 1],
                 color=color, linewidth=3, label=f"Master β={beta}")
        plt.scatter(master[-1, 0], master[-1, 1],
                    color=color, marker="*", s=200)
    plt.scatter([1.0], [1.0], marker="*", s=200, color="yellow", label="Global minimum")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Function: EASGD (varying β)")
    plt.legend()
    plt.tight_layout()
    plt.show()