import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_param_comparison(df, param_x, param_y, value_column="loss"):
    """Correctly pivots the dataframe to compare two specific hyperparameters,

    taking the best (minimum) loss found strictly within each specific (X, Y) cell.
    """
    print(f"Comparing {param_x} (X-axis) vs {param_y} (Y-axis)...")

    # Pivot table isolates each unique X and Y pair, then applies 'min' to the losses in that cell
    pivot_df = df.pivot_table(
        values=value_column, index=param_y, columns=param_x, aggfunc="min"
    )

    # Plotting the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",  # Shines bright on the lowest loss values
        cbar_kws={"label": value_column},
        linewidths=0.5,
    )

    plt.title(f"Minimum {value_column} per Cell: {param_y} vs {param_x}")
    plt.ylabel(param_y)
    plt.xlabel(param_x)
    plt.tight_layout()

    # Create an output directory and save the plot
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/comparison_{param_y}_vs_{param_x}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}\n")


if __name__ == "__main__":
    csv_file = "optimization_results.csv"

    if not os.path.isfile(csv_file):
        print(
            f"Error: Could not find '{csv_file}'. Make sure your grid search has run."
        )
        exit(1)

    # Load data
    df = pd.read_csv(csv_file)

    # Clean data: ensure everything is numeric to prevent aggregation glitches
    for col in ["num_workers", "eta", "beta", "alpha", "alpha_pull", "loss"]:
        df[col] = pd.to_numeric(df[col])

    # --- Generate specific comparisons ---
    # 1. Learning rate vs Interaction strength
    plot_param_comparison(df, "eta", "beta")

    # 2. Coupling dynamics
    plot_param_comparison(df, "alpha", "alpha_pull")

    # 3. Scaling vs Learning rate
    plot_param_comparison(df, "eta", "num_workers")

    print("All analysis plots generated successfully in the 'plots/' folder!")