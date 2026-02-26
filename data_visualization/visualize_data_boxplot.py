import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# bi, tri-fBm
def plot_clean_boxplot_withK(csv_path, out_png=None, name="tfBm"):

    df = pd.read_csv(csv_path)
    df["H0"] = pd.to_numeric(df["H0"])
    df["K"] = pd.to_numeric(df["K"])
    df["H_hat"] = pd.to_numeric(df["H_hat"])
    df["sqerr"] = (df["H_hat"] - df["H0"] * df["K"])**2
    grouped_df = (
        df.groupby(["H0", "K"])["sqerr"]
          .apply(list)
          .reset_index()
          .sort_values(["H0", "K"])
    )

    grouped = grouped_df["sqerr"].tolist()
    labels = [
        rf"({row.H0:g}, {row.K:g})"
        for row in grouped_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    ax.boxplot(
        grouped,
        widths=0.55,
        showfliers=True,
        patch_artist=False,
        flierprops=dict(marker="+", markersize=5, markeredgecolor="red"),
        boxprops=dict(color="blue", linewidth=1.2),
        whiskerprops=dict(color="black", linestyle="--"),
        capprops=dict(color="black"),
        medianprops=dict(color="red", linewidth=1.2),
    )

    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel(r"$(H_0, K)$")
    ax.set_ylabel(r"Squared error $(\hat{H}-H_0*K)^2$")

    # Remove outliers since they make real box too small
    upper = np.percentile(df["sqerr"], 99)
    ax.set_ylim(0, upper * 1.05)

    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.title(f"Squared error vs $(H_0, K)$ pairs for {name}")

    if out_png:
        plt.savefig(out_png, bbox_inches="tight", dpi=250)

    plt.show()
# fbm, sfbm
def plot_clean_boxplot(csv_path, out_png=None, name="fBm"):
    df = pd.read_csv(csv_path)
    df["H0"] = pd.to_numeric(df["H0"])
    df["H_hat"] = pd.to_numeric(df["H_hat"])
    df["sqerr"] = (df["H_hat"] - df["H0"]) ** 2
    H0_vals = np.sort(df["H0"].unique())
    grouped = [df[df["H0"] == h]["sqerr"].values for h in H0_vals]
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=160)
    ax.boxplot(
        grouped,
        widths=0.55,
        showfliers=True,
        patch_artist=False,
        flierprops=dict(marker="+", markersize=5, markeredgecolor="red"),
        boxprops=dict(color="blue", linewidth=1.2),
        whiskerprops=dict(color="black", linestyle="--"),
        capprops=dict(color="black"),
        medianprops=dict(color="red", linewidth=1.2),
    )
    ax.set_xticks(np.arange(1, len(H0_vals) + 1))
    ax.set_xticklabels([f"{h:g}" for h in H0_vals], rotation=45)
    ax.set_xlabel(r"$H_0$")
    ax.set_ylabel(r"Squared error $(\hat{H}-H_0)^2$")
    upper = np.percentile(df["sqerr"], 99)
    ax.set_ylim(0, upper * 1.05)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.title(
        f"Squared error vs $H_0$ for {name}"
    )
    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    plot_clean_boxplot_withK("data_visualization/hurst_results_a2_DpwBiFbm.csv", 
                       out_png="{OUT}",
                       name="bfBm")