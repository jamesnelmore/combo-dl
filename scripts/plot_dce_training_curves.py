"""Plot DCE training curves: CE accuracy, best residual, and smoothed avg residual."""

import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb
from scipy import stats


CUTOFF = datetime(2026, 4, 12, tzinfo=timezone.utc)


def load_runs() -> list[dict]:
    api = wandb.Api()
    all_runs = api.runs("combo-dl")
    rows = []
    for r in all_runs:
        created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
        if created < CUTOFF:
            continue
        srg = r.config.get("srg_parameters", {})
        n, k, lam, mu = srg.get("n"), srg.get("k"), srg.get("lambda"), srg.get("mu")
        if n is None:
            continue
        history = r.history(keys=["accuracy", "best_score", "avg_score", "_step"], samples=5000)
        if history.empty:
            continue
        rows.append({
            "n": n, "k": k, "lambda": lam, "mu": mu,
            "label": f"({n},{k},{lam},{mu})",
            "history": history,
            "solved": r.summary.get("best_score", -1) >= 0,
        })
    rows.sort(key=lambda r: (r["n"], r["k"], r["lambda"], r["mu"]))
    seen = {}
    for r in rows:
        key = (r["n"], r["k"], r["lambda"], r["mu"])
        seen[key] = r
    return list(seen.values())


def smooth(y: np.ndarray, window: int = 30) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def analyze(runs: list[dict]) -> None:
    print("\n=== Numerical analysis: avg_score vs best_score ===\n")

    plateau_together, luck_only, avg_only, both_stuck = 0, 0, 0, 0
    spearman_avg_acc = []

    for r in runs:
        hist = r["history"].dropna(subset=["best_score", "avg_score", "accuracy"])
        if len(hist) < 20:
            continue
        n = r["n"]
        label = r["label"]

        best = hist["best_score"].values
        avg = hist["avg_score"].values
        acc = hist["accuracy"].values

        mid = len(hist) // 2
        best_improved = best[mid:].max() > best[:mid].max()
        avg_improved = avg[mid:].max() > avg[:mid].max()

        if best_improved and avg_improved:
            plateau_together += 1
            tag = "both improve in 2nd half"
        elif best_improved and not avg_improved:
            luck_only += 1
            tag = "best improves (luck), avg stuck"
        elif avg_improved and not best_improved:
            avg_only += 1
            tag = "avg improves, best stuck"
        else:
            both_stuck += 1
            tag = "both stuck in 2nd half"

        rho_avg_acc, _ = stats.spearmanr(avg, acc)
        spearman_avg_acc.append(rho_avg_acc)

        # gap: how far is avg below best (as fraction of best magnitude)
        gap = np.mean(best - avg)
        print(f"{label:20s} | {tag:35s} | avg-best gap={gap:.1f} | rho(avg,acc)={rho_avg_acc:.3f}")

    n_runs = len(runs)
    print(f"\n2nd-half improvement breakdown (n={n_runs}):")
    print(f"  Both improve:         {plateau_together} ({100*plateau_together/n_runs:.0f}%)")
    print(f"  Best only (luck):     {luck_only} ({100*luck_only/n_runs:.0f}%)")
    print(f"  Avg only:             {avg_only} ({100*avg_only/n_runs:.0f}%)")
    print(f"  Both stuck:           {both_stuck} ({100*both_stuck/n_runs:.0f}%)")
    print(f"\nMedian rho(avg_score, accuracy): {np.median(spearman_avg_acc):.3f}")
    high = sum(r > 0.7 for r in spearman_avg_acc)
    low = sum(abs(r) < 0.2 for r in spearman_avg_acc)
    print(f"rho(avg,acc) > 0.7:    {high}/{n_runs}")
    print(f"|rho(avg,acc)| < 0.2:  {low}/{n_runs}")


def main() -> None:
    print("Fetching runs from W&B...")
    runs = load_runs()
    print(f"{len(runs)} unique SRGs")

    analyze(runs)

    ncols = 6
    nrows = math.ceil(len(runs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.flatten()

    color_acc = "#2196F3"
    color_best = "#F44336"
    color_avg = "#FF9800"

    for ax, r in zip(axes, runs):
        hist = r["history"].dropna(subset=["accuracy", "best_score"])
        steps = hist["_step"].values
        accuracy = hist["accuracy"].values
        best_res = -hist["best_score"].values

        ax2 = ax.twinx()
        ax.plot(steps, accuracy, color=color_acc, linewidth=0.8, alpha=0.9)
        ax2.plot(steps, best_res, color=color_best, linewidth=0.8, alpha=0.7)

        if "avg_score" in hist.columns:
            avg_res = -hist["avg_score"].values
            avg_smooth = smooth(avg_res, window=max(10, len(avg_res) // 20))
            ax2.plot(steps, avg_smooth, color=color_avg, linewidth=1.0, alpha=0.9)

        title_color = "#2E7D32" if r["solved"] else "black"
        ax.set_title(r["label"], fontsize=7.5, color=title_color, pad=2)

        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax2.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

    for ax in axes[len(runs):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=color_acc, label="CE accuracy"),
        Line2D([], [], color=color_best, label="Best residual (−best_score)"),
        Line2D([], [], color=color_avg, label="Avg residual (−avg_score, smoothed)"),
        Line2D([], [], color="#2E7D32", label="Solved (green title)"),
    ], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("DCE training curves: CE accuracy vs best and avg residual", fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = "plots/dce_training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
