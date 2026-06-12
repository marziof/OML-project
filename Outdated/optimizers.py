import numpy as np
import time

# Finite difference gradient
def numerical_grad(f, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    d = theta.shape[0]
    grad = np.zeros(d)
    f_theta = f(theta)  # compute once
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        grad[i] = (f(theta + e) - f_theta) / h
    return grad

def numerical_hessian(f, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    d = theta.shape[0]
    hessian = np.zeros((d, d))
    f_theta = f(theta)  # compute once
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = h
        for j in range(d):
            e_j = np.zeros(d)
            e_j[j] = h
            f_ij_plus = f(theta + e_i + e_j)
            f_ij_minus_i = f(theta - e_i + e_j)
            f_ij_minus_j = f(theta + e_i - e_j)
            f_ij_minus_both = f(theta - e_i - e_j)
            hessian[i, j] = (f_ij_plus - f_ij_minus_i - f_ij_minus_j + f_ij_minus_both) / (4 * h**2)
    return hessian

#### OPTIMIZERS

## Baseline sgd
def sgd_baseline(func, theta_init, lr=0.1, n_steps=100, lr_decay=None):
    theta = np.array(theta_init, dtype=float)
    trajectory = [theta.copy()]
    start_time = time.time()
    for t in range(n_steps):
        grad = numerical_grad(func, theta)
        # optional: decay LR
        eta = lr * (1 - t/n_steps) if lr_decay else lr
        theta = theta - eta * grad
        trajectory.append(theta.copy())
    end_time = time.time()
    print(f"SGD baseline completed in {end_time - start_time:.2f} seconds.")
    eng_grad = numerical_grad(func, theta)
    end_hessian = numerical_hessian(func, theta)
    end_val = func(theta)
    return {"traj": np.array(trajectory), "loss": np.array([func(t) for t in trajectory]), "theta": theta, "loss_val": end_val, "grad": eng_grad, "hessian": end_hessian}


## ADAM

def adam_baseline(func, theta_init, lr=0.1, n_steps=100, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=None):
    """
    Adam optimizer baseline for comparison with SGD.
    """
    theta = np.array(theta_init, dtype=float)
    m = np.zeros_like(theta)  # first moment
    v = np.zeros_like(theta)  # second moment
    trajectory = [theta.copy()]
    start_time = time.time()

    for t in range(1, n_steps + 1):
        grad = numerical_grad(func, theta)

        # update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        # bias-corrected moments
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # optional learning rate decay
        eta = lr * (1 - t/n_steps) if lr_decay else lr

        # parameter update
        theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)

        trajectory.append(theta.copy())

    end_time = time.time()
    print(f"Adam baseline completed in {end_time - start_time:.2f} seconds.")

    eng_grad = numerical_grad(func, theta)
    end_hessian = numerical_hessian(func, theta)
    end_val = func(theta)

    return {
        "traj": np.array(trajectory),
        "loss": np.array([func(t) for t in trajectory]),
        "theta": theta,
        "loss_val": end_val,
        "grad": eng_grad,
        "hessian": end_hessian
    }

# ----------------------------------------------------------------------------
## Noise decay sgd
def sgd_LH(f, theta_init, n_steps=1000, lr=0.01, sigma_0=0.1, lam=0.1, alpha=0.1, R_min=0.5, R_max=10.0, tau=1.0, max_step=1, eps=1e-8, seed=42, k=3):
    """
    Two-agent SGD with a Low-noise anchor (L) and High-noise explorer (H).
 
    Receives:
        f         : callable, loss function f(theta) -> scalar
        theta_init: np.ndarray, starting point (used for both L and H)
        n_steps   : int, number of iterations
        eta_L, eta_H : float, learning rates for L and H
        sigma_0   : float, base noise scale for H
        lam       : float, repulsion strength (H pushed away from L)
        alpha     : float in (0,1], relaxation on steepness criterion
        R_min, R_max : float, bounds for adaptive noise scaling and potential around L
        tau       : float, time constant for noise adaptation
        max_step  : float, maximum allowed step size for H to ensure stability
        eps       : float, numerical stability in sigma computation
        seed      : int, RNG seed for reproducibility
 
    Returns:
        dict with keys:
            theta_L      : final position of L
            theta_H      : final position of H
            loss_L       : loss trajectory of L   (n_steps,)
            loss_H       : loss trajectory of H   (n_steps,)
            promotions   : list of step indices where L was promoted
            sigma_history: noise level of H over time (n_steps,)
    """
    eta_L = lr
    eta_H = lr
    rng = np.random.default_rng(seed)
    traj_L = []
    traj_H = []
    theta_L = np.array(theta_init, dtype=float)
    theta_H = np.array(theta_init, dtype=float)
    loss_L = np.zeros(n_steps)
    loss_H = np.zeros(n_steps)
    sigma_history = np.zeros(n_steps)
    promotions = []
    promotion_counter = 0  # counts consecutive "better" steps
    diverged_counter = 0  # counts consecutive divergence steps
    start_time  = time.time()
    for t in range(n_steps):

        # ── Compute gradients ────────────────────────────────────────────────
        g_L = numerical_grad(f, theta_L)
        g_H = numerical_grad(f, theta_H)
        norm_g_L = np.linalg.norm(g_L)
        norm_g_H = np.linalg.norm(g_H)

        # ── Second agent update ──────────────────────────────
        grad_U = H_potential_grad(theta_H, theta_L, norm_g_L, A=lam, sigma_p=0.5, B=0.01, p=2)
        
        # --- weaken potential if in verification period ---
        if promotion_counter > 0:
            grad_U_effective = 0.1 * grad_U  # only 10% strength
        else:
            grad_U_effective = grad_U

        # Adaptive noise
        d = theta_H.shape[0]
        sigma_t = np.clip(sigma_0 / (norm_g_L + eps), R_min, R_max) / np.sqrt(d)
        
        # Stochastic update for H
        xi = rng.standard_normal(size=theta_H.shape)
        delta = - eta_H * g_H - eta_H * grad_U_effective + sigma_t * xi

        # Clip step
        step_norm = np.linalg.norm(delta)
        if step_norm > max_step:
            delta = delta / step_norm * max_step

        theta_H = theta_H + delta

        # ── Update L (standard SGD step) ─────────────────────────────────────
        theta_L = theta_L - eta_L * g_L

        # ── Promotion rule with verification ─────────────────────────────────
        f_L = f(theta_L)
        f_H = f(theta_H)
        
        if f_H < f_L and norm_g_H > alpha * norm_g_L:
            promotion_counter += 1
        else:
            promotion_counter = 0  # reset if condition fails

        if promotion_counter >= k:
            theta_L = theta_H.copy()
            promotions.append(t)
            promotion_counter = 0  # reset after promotion
        

        #h_stuck   = norm_g_H < stuck_tol
        
        loss_ratio = 2
        h_diverged = f_H > loss_ratio * f_L
        if h_diverged:
            diverged_counter += 1

        if diverged_counter >= 2 * k:
            theta_H = theta_L.copy()
            diverged_counter = 0  # reset after correction
 
        # ── Logging ──────────────────────────────────────────────────────────
        loss_L[t] = f_L
        loss_H[t] = f_H
        traj_L.append(theta_L.copy())
        traj_H.append(theta_H.copy())
    end_time = time.time()
    print(f"SGD LH completed in {end_time - start_time:.2f} seconds with {len(promotions)} promotions.")
    eng_grad = numerical_grad(f, theta_L)
    end_hessian = numerical_hessian(f, theta_L)
    end_val = f(theta_L)
    return {
        "traj": np.array(traj_L),
        "traj_H": np.array(traj_H),
        "theta": theta_L,
        "theta_H": theta_H,
        "loss_L": loss_L,
        "loss_H": loss_H,
        "promotions": promotions,
        "sigma_history": sigma_history,
        "loss_val": end_val,
        "grad": eng_grad,
        "hessian": end_hessian,
    }

def H_potential_grad(theta_H, theta_L, norm_g_L, A=0.1, sigma_p=0.5, B=0.01, p=2, R_min=0.5, R_max=3.0, tau=1.0):
    diff  = theta_H - theta_L
    dist2 = np.dot(diff, diff)
    dist  = np.sqrt(dist2) + 1e-8

    # ── Gaussian repulsion from L ────────────────────────────────────────────
    grad_repulsion = - (A / sigma_p**2) * diff * np.exp(-dist2 / (2 * sigma_p**2))

    # ── Adaptive leash radius: large when L is flat, small when L is steep ───
    R_t = R_min + (R_max - R_min) / (norm_g_L / tau + 1)

    # ── Soft confinement relative to L, centered at radius R_t ──────────────
    # U_conf = B * (dist - R_t)^p  for dist > R_t, else 0
    # → only pulls H back when it exceeds R_t
    excess = dist - R_t
    if excess > 0:
        grad_confinement = B * p * (excess ** (p - 1)) * (diff / dist)
    else:
        grad_confinement = np.zeros_like(diff)

    return grad_repulsion + grad_confinement

#----------------------------------------------------------------------------

# def sgd_rotational(f, theta_init, n_steps=1000, lr=0.01, beta=0.9, lam=0.5, eps=1e-8):
#     """
#     Poincaré-inspired rotational SGD
    
#     Args:
#         f: function to minimize, f(theta)
#         theta_init: initial parameters (1D array)
#         n_steps: number of optimization steps
#         lr: learning rate
#         beta: momentum coefficient
#         lam: rotational (angular) component weight
#         eps: small value to avoid division by zero
    
#     Returns:
#         traj: array of all theta positions (n_steps x dim)
#         theta: final parameters
#     """
#     theta = np.array(theta_init, dtype=float)
#     d = theta.shape[0]

#     v = np.zeros_like(theta)       # momentum/velocity
#     g_prev = np.zeros_like(theta)  # previous gradient
#     traj = []

#     for t in range(n_steps):
#         # --- compute gradient ---
#         g = numerical_grad(f, theta)

#         # --- compute rotational component (orthogonal to previous gradient) ---
#         if t > 0:
#             denom = np.dot(g_prev, g_prev) + eps
#             proj = (np.dot(g, g_prev) / denom) * g_prev  # projection onto previous gradient
#             r = g - proj  # orthogonal component
#         else:
#             r = np.zeros_like(theta)

#         # --- velocity update (momentum + rotational component) ---
#         v = beta * v + g + lam * r

#         # --- parameter update ---
#         theta = theta - lr * v

#         # --- store trajectory ---
#         traj.append(theta.copy())
#         g_prev = g.copy()

#     eng_grad = numerical_grad(f, theta)
#     end_hessian = numerical_hessian(f, theta)
#     end_val = f(theta)
#     return {"traj": np.array(traj), "theta": theta, "loss_val": end_val, "grad": eng_grad, "hessian": end_hessian}


# def qft_poincare_gd(f, theta_init, n_steps=1000, lr=0.01, beta=0.9, lam=0.5, sigma=0.01, eps=1e-8):
#     """
#     QFT-inspired Poincaré Rotational Gradient Descent
#     Combines momentum, rotational component, and stochastic fluctuations
    
#     Args:
#         f: function to minimize
#         theta_init: initial parameters (1D array)
#         n_steps: number of steps
#         lr: learning rate
#         beta: momentum coefficient
#         lam: rotational component weight
#         sigma: stochastic fluctuation magnitude
#         eps: small value to avoid division by zero
    
#     Returns:
#         traj: array of parameter trajectories
#         theta: final parameters
#     """
#     theta = np.array(theta_init, dtype=float)
#     d = theta.shape[0]

#     v = np.zeros_like(theta)       # velocity (momentum)
#     g_prev = np.zeros_like(theta)  # previous gradient
#     traj = []

#     for t in range(n_steps):
#         # --- compute gradient ---
#         g = numerical_grad(f, theta)

#         # --- rotational / angular component ---
#         if t > 0:
#             denom = np.dot(g_prev, g_prev) + eps
#             proj = (np.dot(g, g_prev) / denom) * g_prev
#             r = g - proj  # orthogonal component
#         else:
#             r = np.zeros_like(theta)

#         # --- stochastic fluctuation (quantum-inspired) ---
#         noise = sigma * np.random.randn(d)

#         # --- velocity update (momentum + rotation + noise) ---
#         v = beta * v + g + lam * r + noise

#         # --- parameter update ---
#         theta = theta - lr * v

#         # --- store trajectory ---
#         traj.append(theta.copy())
#         g_prev = g.copy()

#     eng_grad = numerical_grad(f, theta)
#     end_hessian = numerical_hessian(f, theta)
#     end_val = f(theta)
#     return {"traj": np.array(traj), "theta": theta, "loss_val": end_val, "grad": eng_grad, "hessian": end_hessian}


optimizers = [("sgd_baseline", sgd_baseline), ("adam_baseline", adam_baseline), ("sgd_LH", sgd_LH)] #, ("sgd_rotational", sgd_rotational), ("qft_poincare_gd", qft_poincare_gd)]




# optimizer:
import torch

class SGD_LH(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, sigma_0=0.1, lam=0.1,
                 alpha=0.1, R_min=0.5, R_max=10.0,
                 max_step=1.0, eps=1e-8, k=3, momentum=0.0):

        defaults = dict(lr=lr, sigma_0=sigma_0, lam=lam, alpha=alpha,
                        R_min=R_min, R_max=R_max, max_step=max_step,
                        eps=eps, k=k, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        # Evaluate loss + gradients at current (L) point
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']
            sigma_0   = group['sigma_0']
            lam       = group['lam']
            alpha     = group['alpha']
            R_min     = group['R_min']
            R_max     = group['R_max']
            max_step  = group['max_step']
            eps       = group['eps']
            k         = group['k']
            mu        = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                g_L = p.grad.data.clone()
                norm_g_L = torch.norm(g_L).clamp(min=eps)

                state = self.state[p]
                if len(state) == 0:
                    state['theta_H']            = p.data.clone()
                    state['promotion_counter']  = 0
                    state['diverged_counter']   = 0
                    state['momentum_buf']       = torch.zeros_like(p.data)

                theta_H = state['theta_H']

                # ── Repulsive potential (H pushed away from L) ──────────────
                diff   = theta_H - p.data
                dist2  = diff.pow(2).sum().clamp(min=eps)
                # Repulsive: gradient points away from p.data
                # Repulsion too strong at low lr — H barely moves but gets pushed hard
                #grad_U = -lam * diff / dist2
                grad_U = -lam * lr * diff / dist2  # scale repulsion with lr

                if state['promotion_counter'] > 0:
                    grad_U = grad_U * 0.1   # weaken near promotion

                # ── Re-estimate g_H via closure (if available) ──────────────
                # Without a closure we fall back to g_L (known approximation)
                if closure is not None:
                    # Temporarily move p to theta_H, recompute grad
                    p.data.copy_(theta_H)
                    with torch.enable_grad():
                        closure()
                    g_H = p.grad.data.clone()
                    p.data.copy_(p.data - lr * g_L)  # restore + L-step below
                    # restore L position before we do the L update
                    p.data.add_(lr * g_L)            # undo L step (done below)
                else:
                    g_H = g_L.clone()  # approximation

                norm_g_H = torch.norm(g_H).clamp(min=eps)

                # ── Adaptive noise ───────────────────────────────────────────
                #sigma_t = torch.clamp(sigma_0 / norm_g_L, R_min, R_max)
                sigma_t = torch.clamp(sigma_0 * lr / norm_g_L, R_min * lr, R_max * lr)
                noise   = torch.randn_like(p.data) * sigma_t

                # ── H update ─────────────────────────────────────────────────
                delta = -lr * g_H - lr * grad_U + noise
                step_norm = torch.norm(delta)
                if step_norm > max_step:
                    delta = delta * (max_step / step_norm)
                theta_H = theta_H + delta

                # ── L update (SGD + optional momentum) ───────────────────────
                buf = state['momentum_buf']
                buf.mul_(mu).add_(g_L)
                p.data.add_(-lr * buf)

                # ── Promotion: H → L if H is consistently better ─────────────
                # Use gradient norm as loss proxy (works without closure)
                if norm_g_H < norm_g_L and norm_g_H > alpha * norm_g_L:
                    state['promotion_counter'] += 1
                else:
                    state['promotion_counter'] = 0   # reset on any bad step

                if state['promotion_counter'] >= k:
                    p.data.copy_(theta_H)
                    theta_H = p.data.clone()  # re-anchor H at new L position
                    state['promotion_counter'] = 0

                # ── Divergence reset ─────────────────────────────────────────
                if norm_g_H > 2.0 * norm_g_L:
                    state['diverged_counter'] += 1
                else:
                    state['diverged_counter'] = max(0, state['diverged_counter'] - 1)

                if state['diverged_counter'] >= 2 * k:
                    theta_H = p.data.clone()
                    state['diverged_counter'] = 0

                state['theta_H']      = theta_H
                state['momentum_buf'] = buf

        return loss

class SGD_LH_old(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, sigma_0=0.1, lam=0.1,
                 alpha=0.1, R_min=0.5, R_max=10.0,
                 max_step=1.0, eps=1e-8, k=3):

        defaults = dict(lr=lr, sigma_0=sigma_0, lam=lam,
                        alpha=alpha, R_min=R_min, R_max=R_max,
                        max_step=max_step, eps=eps, k=k)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            sigma_0 = group['sigma_0']
            lam = group['lam']
            alpha = group['alpha']
            R_min = group['R_min']
            R_max = group['R_max']
            max_step = group['max_step']
            eps = group['eps']
            k = group['k']

            for p in group['params']:
                if p.grad is None:
                    continue

                g_L = p.grad.data

                state = self.state[p]

                # --- init state ---
                if len(state) == 0:
                    state['theta_H'] = p.data.clone()
                    state['promotion_counter'] = 0
                    state['diverged_counter'] = 0

                theta_H = state['theta_H']
                g_H = g_L.clone()  # same gradient (approximation)

                norm_g_L = torch.norm(g_L)
                norm_g_H = torch.norm(g_H)

                # --- potential gradient (repulsion) ---
                diff = theta_H - p.data
                dist2 = torch.sum(diff**2) + eps
                grad_U = lam * diff / dist2  # simple repulsive potential

                # weaken during promotion verification
                if state['promotion_counter'] > 0:
                    grad_U = 0.1 * grad_U

                # --- adaptive noise ---
                d = p.data.numel()
                sigma_t = torch.clamp(
                    sigma_0 / (norm_g_L + eps),
                    R_min, R_max
                ) / (d ** 0.5)

                noise = torch.randn_like(p.data)

                # --- H update ---
                delta = -lr * g_H - lr * grad_U + sigma_t * noise

                step_norm = torch.norm(delta)
                if step_norm > max_step:
                    delta = delta / step_norm * max_step

                theta_H = theta_H + delta

                # --- L update (standard SGD) ---
                p.data = p.data - lr * g_L

                # --- promotion rule ---
                f_L = torch.norm(g_L)  # proxy for loss
                f_H = torch.norm(g_H)

                if f_H < f_L and norm_g_H > alpha * norm_g_L:
                    state['promotion_counter'] += 1
                else:
                    state['promotion_counter'] = 0

                if state['promotion_counter'] >= k:
                    p.data = theta_H.clone()
                    state['promotion_counter'] = 0

                # --- divergence reset ---
                if f_H > 2 * f_L:
                    state['diverged_counter'] += 1

                if state['diverged_counter'] >= 2 * k:
                    theta_H = p.data.clone()
                    state['diverged_counter'] = 0

                state['theta_H'] = theta_H

        return loss