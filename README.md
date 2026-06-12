# OML project: SeaSGD : a selective version of EASGD

Class: Optimization for machine learning CS-439

## Overview

This project implements and evaluates SeaSGD, a proposed selective EASGD variant in which workers contribute unevenly to the master update according to their validation performance.
The performance of this method is compared to SGD and EASGD in three contexts: synthetic multimodal landscapes, MNIST, and CIFAR-10.
The repository provides implementations of the SeaSGD algorithm as well as its application to the stated problems.

## Installation

### Prerequisites

- Python 3.11+
- Packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/marziof/OML-project.git
cd OML-project

# Create virtual environment (or conda env)
python -m venv OML_env
source OML_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```text
.
├── configs
│   └── default.py
├── results
│   ├── dfs
│   ├── past_tests
│   ├── plots
│   └── sensor_stats
├── scripts
│   ├── gen_results.py
│   ├── plot_pinf_profile.py
│   ├── plot_results.py
│   ├── plot_sensors.py
│   └── run_gen_results.slurm
├── src
│   ├── algorithms
│   │   ├── non_oracle_selection.py
│   │   ├── optimal_subset_selection.py
│   │   ├── sequential_sensor_selection.py
│   │   └── static_selection.py
│   ├── Analysis
│   ├── experiments
│   ├── helpers
│   └── utils
└── test_nb.ipynb
```

## Quick Start

### 1. Reproduce the figures of the report

Run the notebook:
```bash
results.ipynb
```

### 2. Reproduce tests on low dimensional landscapes


### 3. Reproduce tests on MNIST

```bash
python mnist_analysis.py
```

Results are saved in the `results/` directory.

Can then be plotted using:

```bash
python -m src/plotting.py
```

Plots are saved in `results/plots/`

### 4. Reproduce tests on CIFAR-10

```bash
python cifar_analysis.py
```
