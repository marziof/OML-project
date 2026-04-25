import numpy as np

class Objective:
    def __init__(self, func, min_x):
        self.func = func
        self.min_x = min_x

    def __call__(self, theta):
        return self.func(theta[0], theta[1])
    
    def mse_to_minimum(self, trajectory):
        minimum = self.min_x
        final_point = trajectory[-1]
        mse = np.mean((final_point[0] - minimum[0])**2 + (final_point[1] - minimum[1])**2)
        return mse
    
# FUNCTIONS 

def rastrigin_fn(theta, A=10):
    x, y = theta
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def rosenbrock_fn(theta, a=1, b=100):
    x, y = theta
    return (a - x)**2 + b * (y - x**2)**2

def sphere_fn(theta):
    x, y = theta
    return x**2 + y**2

def beale_fn(theta):
    x, y = theta
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2)

def himmelblau_fn(theta):
    x, y = theta
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def griewank_fn(theta):
    x, y = theta
    part1 = (x**2 + y**2) / 4000
    part2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return part1 - part2 + 1

def easom_fn(theta):
    x, y = theta
    return -np.cos(x)*np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def schaffer2_fn(theta):
    x, y = theta
    num = (np.sin(x**2 - y**2))**2 - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2
    return 0.5 + num/den

def mccormick_fn(theta):
    x, y = theta
    return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

def holder_table_fn(theta):
    x, y = theta
    val = -np.abs(np.sin(x)*np.cos(y)*np.exp(abs(1 - np.sqrt(x**2 + y**2)/np.pi)))
    return val

def eggholder_fn(theta):
    x, y = theta
    return -((y + 47)*np.sin(np.sqrt(abs(x/2 + (y + 47)))) + x*np.sin(np.sqrt(abs(x - (y+47)))))

def cross_in_tray_fn(theta):
    x, y = theta
    exp_term = np.exp(abs(100 - np.sqrt(x**2 + y**2)/np.pi))
    return -0.0001*(abs(np.sin(x)*np.sin(y)*exp_term) + 1)**0.1


## OBJECTIVES

# Rastrigin: minimum at (0,0)
rastrigin = Objective(rastrigin_fn, np.array([0.0, 0.0]))

# Rosenbrock: minimum at (1,1)
rosenbrock = Objective(rosenbrock_fn, np.array([1.0, 1.0]))

# Sphere: minimum at (0,0)
sphere = Objective(sphere_fn, np.array([0.0, 0.0]))

# Beale: minimum at (3,0.5)
beale = Objective(beale_fn, np.array([3.0, 0.5]))

# Himmelblau: multiple minima, choose one (3,2)
himmelblau = Objective(himmelblau_fn, np.array([3.0, 2.0]))

# Griewank: minimum at (0,0)
griewank = Objective(griewank_fn, np.array([0.0, 0.0]))

# Easom: minimum at (π, π)
easom = Objective(easom_fn, np.array([np.pi, np.pi]))

# Schaffer function N.2: minimum at (0,0)
schaffer2 = Objective(schaffer2_fn, np.array([0.0, 0.0]))

# McCormick: minimum at (-0.54719, -1.54719)
mccormick = Objective(mccormick_fn, np.array([-0.54719, -1.54719]))

# Holder Table: one of its minima at (8.05502, 9.66459)
holder_table = Objective(holder_table_fn, np.array([8.05502, 9.66459]))

# Eggholder: minimum at (512, 404.2319)
eggholder = Objective(eggholder_fn, np.array([512.0, 404.2319]))

# Cross-in-tray: one of its minima at (1.34941, -1.34941)
cross_in_tray = Objective(cross_in_tray_fn, np.array([1.34941, -1.34941]))

## List of all objectives for easy iteration
objectives = [("rastrigin", rastrigin), ("rosenbrock", rosenbrock), ("sphere", sphere), ("beale", beale), ("himmelblau", himmelblau), ("griewank", griewank), ("easom", easom), ("schaffer2", schaffer2), ("mccormick", mccormick), ("holder_table", holder_table), ("eggholder", eggholder), ("cross_in_tray", cross_in_tray)]