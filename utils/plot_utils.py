import os
import numpy as np
import matplotlib.pyplot as plt


def make_grid(elastic_df, alphas, betas, metric):
    grid = np.zeros((len(alphas), len(betas)))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            grid[i, j] = elastic_df.loc[
                (elastic_df["alpha"] == a)
                & (elastic_df["beta"] == b),
                metric,
            ].values[0]

    return grid


def plot_heatmaps(
    elastic_df,
    alphas,
    betas,
    sgd_acc,
    sgd_te_loss,
    save_path,
):
    acc_grid = make_grid(elastic_df, alphas, betas, "test_accuracy")
    loss_grid = make_grid(elastic_df, alphas, betas, "test_loss")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, grid, title, cmap, fmt in zip(
        axes,
        [acc_grid, loss_grid],
        ["Test Accuracy", "Test Loss"],
        ["YlGn", "YlOrRd"],
        [".4f", ".4f"],
    ):
        im = ax.imshow(grid, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"β={b}" for b in betas])

        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"α={a}" for a in alphas])

        ax.set_title(title)

        plt.colorbar(im, ax=ax)

        for i in range(len(alphas)):
            for j in range(len(betas)):
                ax.text(
                    j,
                    i,
                    format(grid[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    for ax, val, label in zip(
        axes,
        [sgd_acc, sgd_te_loss],
        ["SGD acc", "SGD loss"],
    ):
        ax.set_xlabel(
            f"(SGD baseline: {label} = {val:.4f})",
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_accuracy_heatmap(
    elastic_df,
    alphas,
    betas,
    sgd_acc,
    save_path,
    fontsize=12,
    cmap="viridis",
):
    acc_grid = make_grid(elastic_df, alphas, betas, "test_accuracy")

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(acc_grid, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(betas)))
    ax.set_yticks(range(len(alphas)))

    ax.set_xticklabels([f"β={b}" for b in betas])
    ax.set_yticklabels([f"α={a}" for a in alphas])

    ax.tick_params(axis="both", labelsize=fontsize - 2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Test Accuracy", fontsize=fontsize)

    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(
                j,
                i,
                f"{acc_grid[i, j]:.4f}",
                ha="center",
                va="center",
                fontsize=fontsize - 2,
            )

    ax.set_xlabel(
        f"SGD baseline accuracy = {sgd_acc:.4f}",
        fontsize=fontsize,
        color="gray",
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_loss_curves_grid(
    elastic_df,
    alphas,
    betas,
    sgd_tr_curve,
    sgd_te_curve,
    save_path,
):
    epochs = range(1, len(sgd_te_curve) + 1)

    fig, axes = plt.subplots(
        len(alphas),
        len(betas),
        figsize=(4 * len(betas), 3.5 * len(alphas)),
        sharex=True,
        sharey=True,
    )

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):

            row = elastic_df.loc[
                (elastic_df["alpha"] == alpha)
                & (elastic_df["beta"] == beta)
            ].iloc[0]

            ax = axes[i][j]

            ax.plot(epochs, row["train_loss_curve"], lw=1.8)
            ax.plot(epochs, row["test_loss_curve"], lw=1.8)

            ax.plot(epochs, sgd_tr_curve, ls=":", alpha=0.5)
            ax.plot(epochs, sgd_te_curve, ls=":", alpha=0.5)

            ax.set_title(
                f"α={alpha} β={beta}\nacc={row['test_accuracy']:.4f}",
                fontsize=8,
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_top_configs(
    elastic_df,
    sgd_acc,
    save_path,
    top_n=5,
):
    top = elastic_df.nlargest(top_n, "test_accuracy").copy()

    top["label"] = top.apply(
        lambda r: f"α={r['alpha']}\nβ={r['beta']}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        range(len(top)),
        top["test_accuracy"],
        width=0.5,
    )

    ax.axhline(
        sgd_acc,
        ls="--",
        color="black",
        label=f"SGD ({sgd_acc:.4f})",
    )

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"])

    ax.set_ylim(min(top["test_accuracy"].min(), sgd_acc) - 0.005, 1.0)

    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()