import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in ["N", "trials", "trial"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["H0", "H_hat", "sigma2_hat", "K", "L", "sigma"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "process" in df.columns:
        df["process"] = df["process"].astype(str)

    needed = ["process", "N", "H0", "H_hat", "K", "L"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=needed)
    return df

def plot_rmse_heatmap_N_vs_H0(
    df,
    *,
    process: str,
    out_png: str,
    K: float | None = None,
    L: float | None = None,
    sigma: float | None = None,
    y_tick_style: str = "pow2",   
    annotate_cells: bool = False,
    annotate_fmt: str = ".3f",
    tighten_contrast: bool = False, 
    contrast_lo: float = 5.0,
    contrast_hi: float = 95.0,
    name = "fBm",
):
    d = df[df["process"] == process].copy()
    if d.empty:
        raise ValueError(f"No rows found for process={process!r}. Available: {sorted(df['process'].unique())}")

    if K is not None:
        d = d[np.isclose(d["K"], K)]
    if L is not None:
        d = d[np.isclose(d["L"], L)]
    if sigma is not None:
        d = d[np.isclose(d["sigma"], sigma)]

    d["sqerr"] = (d["H_hat"] - d["H0"]) ** 2
    rmse = (
        d.groupby(["N", "H0"])["sqerr"]
         .mean()
         .pipe(np.sqrt)
         .reset_index(name="RMSE")
    )

    pivot = rmse.pivot_table(index="N", columns="H0", values="RMSE", aggfunc="mean")
    if pivot.empty:
        raise ValueError("Pivot table is empty. Check filters and whether N/H0 exist for this process.")
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    Z = pivot.values
    Ns = pivot.index.to_numpy()
    H0s = pivot.columns.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)

    im_kwargs = dict(aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    if tighten_contrast:
        vmin = np.nanpercentile(Z, contrast_lo)
        vmax = np.nanpercentile(Z, contrast_hi)
        im = ax.imshow(Z, vmin=vmin, vmax=vmax, **im_kwargs)
    else:
        im = ax.imshow(Z, **im_kwargs)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"RMSE = $\sqrt{\mathbb{E}[(\hat H - H_0)^2]}$")
    ax.set_xticks(np.arange(len(H0s)))
    ax.set_xticklabels([f"{h:g}" for h in H0s], rotation=45, ha="right")
    ax.set_xlabel(r"$H_0$")
    ax.set_yticks(np.arange(len(Ns)))

    if y_tick_style.lower() == "pow2":
        exps = [int(np.round(np.log2(n))) for n in Ns]
        ax.set_yticklabels([rf"$2^{{{e}}}$" for e in exps])
        ax.set_ylabel(r"Sample size $N$")
    elif y_tick_style.lower() == "log2":
        exps = [int(np.round(np.log2(n))) for n in Ns]
        ax.set_yticklabels([f"{e}" for e in exps])
        ax.set_ylabel(r"$\log_2 N$")
    else:
        ax.set_yticklabels([f"{int(n)}" for n in Ns])
        ax.set_ylabel(r"$N$")

    if annotate_cells:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isnan(val):
                    continue

                normalized = im.norm(val) 
                color = "white" if normalized < 0.5 else "black"

                ax.text(
                    j, i,
                    format(val, annotate_fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color
                )

    title_bits = [f"{name}", r"Estimator RMSE across $N$ and $H_0$"]
    if K is not None: title_bits.append(f"K={K:g}")
    if L is not None: title_bits.append(f"L={L:g}")
    if sigma is not None: title_bits.append(f"sigma={sigma:g}")
    ax.set_title(" | ".join(title_bits))

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

def plot_rmse_convergence(
    csv_path: str,
    *,
    process: str | None = None,
    H0_list: list[float] | None = None, 
    K: float | None = None,
    L: float | None = None,
    sigma: float | None = None,
    n_col: str = "N",
    H0_col: str = "H0",
    Hhat_col: str = "H_hat",
    fit_powerlaw: bool = True,
    fit_min_n: int | None = None,
    show_points: bool = True,
    out_png: str | None = None,
    name = "fBm",
):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for c in [n_col, H0_col, Hhat_col, "K", "L", "sigma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[n_col, H0_col, Hhat_col]).copy()

    if process is not None and "process" in df.columns:
        df = df[df["process"].astype(str) == str(process)]
    if K is not None and "K" in df.columns:
        df = df[np.isclose(df["K"], K)]
    if L is not None and "L" in df.columns:
        df = df[np.isclose(df["L"], L)]
    if sigma is not None and "sigma" in df.columns:
        df = df[np.isclose(df["sigma"], sigma)]

    if df.empty:
        raise ValueError("No data left after filtering.")

    if H0_list is None:
        H0_list = sorted(df[H0_col].unique())

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    for H0 in H0_list:
        d = df[np.isclose(df[H0_col], H0)]
        if d.empty:
            continue

        d["sqerr"] = (d[Hhat_col] - d[H0_col]) ** 2
        g = d.groupby(n_col)["sqerr"].mean().sort_index()

        n_vals = g.index.to_numpy(dtype=float)
        rmse_vals = np.sqrt(g.to_numpy(dtype=float))

        label = rf"$H_0={H0:g}$"

        ax.plot(
            n_vals,
            rmse_vals,
            marker="o" if show_points else None,
            linewidth=2,
            label=label
        )

        # Power law fitting for curve
        if fit_powerlaw:
            mask = np.isfinite(n_vals) & np.isfinite(rmse_vals) & (rmse_vals > 0)
            if fit_min_n is not None:
                mask &= (n_vals >= fit_min_n)

            n_fit = n_vals[mask]
            y_fit = rmse_vals[mask]

            if n_fit.size >= 2:
                x = np.log(n_fit)
                y = np.log(y_fit)
                alpha, logC = np.polyfit(x, y, 1)
                C = float(np.exp(logC))

                y_hat = C * (n_vals ** alpha)
                ax.plot(
                    n_vals,
                    y_hat,
                    linestyle="--",
                    linewidth=1.8
                )
                ax.text(
                    n_vals[-1],
                    y_hat[-1],
                    rf"$\alpha={alpha:.3f}$",
                    fontsize=9
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Path length $n$ (log$_2$ scale)")
    ax.set_ylabel(r"RMSE $= \sqrt{\mathbb{E}[(\hat H - H_0)^2]}$ (log$_2$ scale)")
    ax.set_title(f"log-log RMSE convergence vs sample size for {name}")

    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png}")

    plt.show()


df = load_results("data_visualization\\hurst_results_RSME_sfBm.csv")
plot_rmse_heatmap_N_vs_H0(
    df,
    process="DpwSubFbmSimulator",
    out_png=r"path",
    y_tick_style="pow2",
    annotate_cells=True,
    name = "Sub-fractional Brownian Motion",
)

# plot_rmse_convergence(
#     "data_visualization/hurst_results_RSME_fBm.csv",
#     process="WoodChanFbmSimulator",
#     H0_list = [0.3,0.5,0.7],
#     fit_powerlaw = False,
#     name = "fBm",
#     out_png=r"path",
# )