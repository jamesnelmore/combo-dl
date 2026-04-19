"""Plot DCE benchmark results: solve time for solved, normalized residual for unsolved."""

import glob
import json

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def load_results(base: str = "bench_output/dce-bench") -> list[dict]:
    rows = []
    for task_dir in sorted(glob.glob(f"{base}/???/")):
        run_dirs = sorted(glob.glob(f"{task_dir}*/results.json"))
        if not run_dirs:
            continue
        with open(run_dirs[-1]) as f:
            r = json.load(f)
        cfg = r["config"]
        n = cfg["n"]
        score = r["best_score"]
        solved = abs(score) < 1e-6
        rows.append({
            "n": n,
            "k": cfg["k"],
            "lambda": cfg["lambda"],
            "mu": cfg["mu"],
            "solved": solved,
            "wall_seconds": r["wall_seconds"],
            "norm_n4": -score / n**4,
            "norm_n2": -score / n**2,
            "label": f"({n},{cfg['k']},{cfg['lambda']},{cfg['mu']})",
        })
    rows.sort(key=lambda r: (r["n"], r["k"], r["lambda"], r["mu"]))
    return rows


def rolling_mean(vals: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.full_like(vals, np.nan)
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), lo + window)
        chunk = vals[lo:hi]
        if np.any(~np.isnan(chunk)):
            out[i] = np.nanmean(chunk)
    return out


def plot_residual_panel(ax, rows, x, key: str, ylabel: str) -> None:
    solved_color = "#4CAF50"
    unsolved_color = "#F44336"

    vals = np.array([r[key] if not r["solved"] else np.nan for r in rows])

    for xi, r in enumerate(rows):
        if r["solved"]:
            ax.bar(xi, 0, color=solved_color, width=0.6)
        else:
            ax.bar(xi, r[key], color=unsolved_color, width=0.6, alpha=0.7)

    trend = rolling_mean(vals, window=7)
    ax.plot(x, trend, color="black", linewidth=1.5, linestyle="--", label="Rolling mean (w=7)", zorder=5)

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    rows = load_results()
    x = np.arange(len(rows))
    labels = [r["label"] for r in rows]

    solved_color = "#4CAF50"
    unsolved_color = "#F44336"
    timeout = 28500

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.08},
    )
    ax_top, ax_n4, ax_n2 = axes

    # ── Top: wall time ────────────────────────────────────────────────────────
    for xi, r in enumerate(rows):
        if r["solved"]:
            ax_top.scatter(xi, r["wall_seconds"], color=solved_color, marker="o", s=50, zorder=3)
        else:
            ax_top.scatter(xi, timeout, color=unsolved_color, marker="x", s=50, zorder=3, alpha=0.6)

    ax_top.axhline(timeout, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Wall time (s, log scale)")
    ax_top.set_title("DCE benchmark: solve time and normalized residual by SRG")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.legend(handles=[
        mlines.Line2D([], [], color=solved_color, marker="o", linestyle="None", label="Solved"),
        mlines.Line2D([], [], color=unsolved_color, marker="x", linestyle="None", label="Timeout (8h)"),
    ], fontsize=9)

    # ── Middle: n^4 normalization ─────────────────────────────────────────────
    plot_residual_panel(ax_n4, rows, x, "norm_n4", "−score / n⁴")

    # ── Bottom: n^2 normalization ─────────────────────────────────────────────
    plot_residual_panel(ax_n2, rows, x, "norm_n2", "−score / n²")

    ax_n2.set_xlabel("SRG parameters (n, k, λ, μ)")
    ax_n2.set_xticks(x)
    ax_n2.set_xticklabels(labels, rotation=90, fontsize=6.5)

    plt.tight_layout()
    out = "plots/dce_benchmark.png"
    plt.savefig(out, dpi=200)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
