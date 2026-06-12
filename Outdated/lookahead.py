
# imports
import numpy as np
import time
from optimizers import numerical_grad, numerical_hessian

def lookahead_sgd(func, theta_init, lr=0.1, k=5, alpha=0.5, n_steps=100):
    phi = np.array(theta_init, dtype=float)
    theta = phi.copy()

    trajectory = [phi.copy()]
    start_time = time.time()

    step = 0
    while step < n_steps:
        theta = phi.copy()

        # fast weights inner loop
        for i in range(k):
            if step >= n_steps:
                break
            grad = numerical_grad(func, theta)
            theta = theta - lr * grad
            trajectory.append(theta.copy())
            step += 1

        # outer sync
        phi = phi + alpha * (theta - phi)

    end_time = time.time()

    eng_grad = numerical_grad(func, phi)
    end_hessian = numerical_hessian(func, phi)
    end_val = func(phi)

    print(f"Lookahead completed in {end_time - start_time:.2f} seconds.")

    return {
        "traj": np.array(trajectory),
        "loss": np.array([func(t) for t in trajectory]),
        "theta": phi,
        "loss_val": end_val,
        "grad": eng_grad,
        "hessian": end_hessian
    }


def lookahead_adaptive_alpha(func, theta_init, lr=0.1, k=5,
                            alpha0=0.5, n_steps=100,
                            beta=0.9):
    phi = np.array(theta_init, dtype=float)
    theta = phi.copy()

    trajectory = [phi.copy()]
    prev_loss = func(phi)
    ema_improve = 0.0

    start_time = time.time()
    step = 0

    while step < n_steps:
        theta = phi.copy()

        for i in range(k):
            if step >= n_steps:
                break
            grad = numerical_grad(func, theta)
            theta = theta - lr * grad
            trajectory.append(theta.copy())
            step += 1

        loss_phi = func(phi)
        loss_theta = func(theta)

        # improvement signal (positive = good fast movement)
        improve = loss_phi - loss_theta
        # ema_improve = beta * ema_improve + (1 - beta) * improve

        # # adaptive alpha (bounded)
        # alpha = alpha0 * (1 + ema_improve / (abs(loss_phi) + 1e-8))
        # alpha = np.clip(alpha, 0.05, 1.0)

        alpha = alpha0 * (1 + improve / (abs(loss_phi) + 1e-8))
        phi = phi + alpha * (theta - phi)

        prev_loss = loss_theta

    end_time = time.time()

    print(f"Adaptive Lookahead completed in {end_time - start_time:.2f} seconds.")

    return {
        "traj": np.array(trajectory),
        "loss": np.array([func(t) for t in trajectory]),
        "theta": phi
    }


def lookahead_continuous_sync(func, theta_init, lr=0.1,
                        alpha=0.5, lam=0.1,
                        n_steps=100):
    theta = np.array(theta_init, dtype=float)
    phi = theta.copy()

    trajectory = [phi.copy()]
    start_time = time.time()

    for t in range(n_steps):
        grad = numerical_grad(func, theta)

        # fast dynamics + pull to slow weights
        theta = theta - lr * grad - lam * (theta - phi)

        # slow tracking of fast
        phi = phi + alpha * (theta - phi)

        trajectory.append(phi.copy())

    end_time = time.time()

    print(f"Continuous sync completed in {end_time - start_time:.2f} seconds.")

    return {
        "traj": np.array(trajectory),
        "loss": np.array([func(t) for t in trajectory]),
        "theta": phi
    }



def lookahead_tanh(func, theta_init, lr=0.1,
                   alpha=0.5, k=5,
                   scale=1.0, n_steps=100):
    phi = np.array(theta_init, dtype=float)
    theta = phi.copy()

    trajectory = [phi.copy()]
    step = 0
    start_time = time.time()

    while step < n_steps:
        theta = phi.copy()

        for i in range(k):
            if step >= n_steps:
                break
            grad = numerical_grad(func, theta)
            theta = theta - lr * grad
            trajectory.append(theta.copy())
            step += 1

        diff = theta - phi

        # nonlinear saturation
        update = np.tanh(scale * diff)

        phi = phi + alpha * update

    end_time = time.time()

    print(f"Tanh Lookahead completed in {end_time - start_time:.2f} seconds.")

    return {
        "traj": np.array(trajectory),
        "loss": np.array([func(t) for t in trajectory]),
        "theta": phi
    }