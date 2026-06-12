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
├── data                                  # data for MNIST and CIFAR
│   ├── cifar-10-batches-py
│   └── MNIST
├── plots
├── results                               # plots + csv files with dfs generated for mnist and cifar (used for plots)
│   ├── cifar_simCLR
│   ├── plots
├── src
│   ├── data
│   │   └── MNIST
│   ├── ElasticOptim.py
│   ├── model.py
│   └── plotting.py
└── utils
    ├── plot_utils.py
    └── simCLR_helpers.py
├── README.md
├── requirements.txt
├── analysis_cifar.py
├── analysis_figures
├── analysis_mnist.py
├── analysis.ipynb
├── beta_alpha_training.csv
├── csv_analyzer_2.py
├── csv_analyzer.py
├── results.ipynb
├── elastic_gd.py
├── experiments.py
├── landscape_analysis_functions.py
├── landscape_analysis.ipynb
├── lookahead_ntb.ipynb
├── lookahead.py
├── objectives.py
├── optimization_results_2.csv
├── optimization_results.csv
├── optimizers.py
├── run_datasets.py
├── run_experiments.ps1
├── run_optimization_2.py
├── run_optimization.py
```

## Quick Start

### 1. Reproduce the figures of the report

Run the notebook:
```bash
results.ipynb
```

### 2. Reproduce tests on low dimensional landscapes

Run the notebook :
```bash
landscape_analysis.ipynb
```

### 3. Reproduce tests on MNIST

```bash
python analysis_mnist.py
```

Results are saved in the `results/` directory.

Can then be plotted using:

```bash
python -m src/plotting.py
```

Plots are saved in `results/plots/`

### 4. Reproduce tests on CIFAR-10

```bash
python analysis_cifar.py
```
