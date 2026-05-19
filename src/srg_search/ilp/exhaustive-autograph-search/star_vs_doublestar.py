"""
Compare λ₁ + μ for three families of trees against the conjectural bound
√(n−1) + 1, across a wide range of n.

Families
--------
1. **Star** K_{1,n-1}:
       λ₁ = √(n−1),  μ = 1  →  λ₁ + μ = √(n−1) + 1   (always equals bound)

2. **Bridged double star** BDS(a, b):
       Two hubs with a−1 and b−1 leaves respectively, joined by a bridge
       vertex (path of length 2 between hubs).  n = a + b + 1.
       μ = 2 (each hub matches one leaf; bridge vertex is stranded).
       λ₁ computed from the characteristic polynomial.

3. **Pure double star** S(a, b):
       Two hubs connected by a direct edge, with a−1 and b−1 leaves.
       n = a + b.  μ = 2.  λ₁ from closed-form quadratic.

For each n we find the optimal split for both double-star variants and
plot all three families against the bound.

Usage
-----
    python star_vs_doublestar.py [--nmax 200] [--out star_vs_doublestar.png]
"""

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


# ── Eigenvalue computations ─────────────────────────────────────────────────

def star_value(n: int) -> float:
    """λ₁ + μ for the star K_{1,n-1}.  Always equals √(n-1) + 1."""
    return math.sqrt(n - 1) + 1


def pure_double_star_eigenvalue(a: int, b: int) -> float:
    """
    Largest eigenvalue of the pure double star S(a, b).

    Two hubs u, v connected by edge u-v.  u has (a-1) leaves, v has (b-1)
    leaves.  n = a + b.

    By symmetry reduction the eigenvalue satisfies:
        (λ² − (a−1))(λ² − (b−1)) = λ²

    Let t = λ²:
        t² − (a+b−1)t + (a−1)(b−1) = 0
        t = [(a+b−1) ± √((a−b)² + 2(a+b) − 3)] / 2
    """
    s = a + b
    disc = (a - b) ** 2 + 2 * s - 3
    t = (s - 1 + math.sqrt(disc)) / 2.0
    return math.sqrt(t)


def bridged_double_star_eigenvalue(a: int, b: int) -> float:
    """
    Largest eigenvalue of the bridged double star BDS(a, b).

    Structure: hub_u — bridge — hub_v
    hub_u has (a-1) pendant leaves, hub_v has (b-1) pendant leaves.
    hub_u connects to bridge and its (a-1) leaves  → deg(hub_u) = a
    hub_v connects to bridge and its (b-1) leaves  → deg(hub_v) = b
    bridge connects to hub_u and hub_v              → deg(bridge) = 2
    Total vertices: 1 + (a-1) + 1 + (b-1) + 1 = a + b + 1

    By symmetry reduction on the three "types" (hub_u, hub_v, bridge,
    leaves-of-u all equal, leaves-of-v all equal), the eigenvector is
    (x, y, z, p, q) where p is the common value on u's leaves and q on v's.

    Eigenvalue equations:
        λx = (a-1)p + z       (hub_u)
        λy = (b-1)q + z       (hub_v)
        λz = x + y            (bridge)
        λp = x                (leaf of u)
        λq = y                (leaf of v)

    Substituting p = x/λ, q = y/λ:
        λx = (a-1)x/λ + z  →  (λ² − (a-1))x = λz   ... (i)
        λy = (b-1)y/λ + z  →  (λ² − (b-1))y = λz   ... (ii)
        λz = x + y                                    ... (iii)

    From (i): x = λz / (λ² − (a-1))
    From (ii): y = λz / (λ² − (b-1))
    Substitute into (iii):
        λz = λz/(λ²−(a-1)) + λz/(λ²−(b-1))

    Divide by λz (λ>0, z≠0):
        1 = 1/(λ²−(a-1)) + 1/(λ²−(b-1))

    Let u = λ²:
        1 = 1/(u−(a-1)) + 1/(u−(b-1))
        (u−(a-1))(u−(b-1)) = (u−(b-1)) + (u−(a-1))
        u² − (a+b−2)u + (a-1)(b-1) = 2u − (a+b−2)
        u² − (a+b)u + (a-1)(b-1) + (a+b-2) = 0
        u² − (a+b)u + (ab − a − b + 1 + a + b − 2) = 0
        u² − (a+b)u + (ab − 1) = 0

    So:  u = [(a+b) ± √((a+b)² − 4(ab−1))] / 2
            = [(a+b) ± √(a² + b² − 2ab + 4)] / 2
            = [(a+b) ± √((a−b)² + 4)] / 2

    Take the + root, then λ = √u.
    """
    disc = (a - b) ** 2 + 4
    u = (a + b + math.sqrt(disc)) / 2.0
    return math.sqrt(u)


# ── Optimal split finders ───────────────────────────────────────────────────

def best_bridged_double_star(n: int) -> dict | None:
    """
    Find the split (a, b) with a + b + 1 = n, a >= b >= 1, that minimises
    λ₁(BDS(a,b)) + 2.  Requires n >= 5 (a >= 2, b >= 2 for a real double star,
    but a=k,b=1 is just a star with an extra path vertex—still valid).
    """
    if n < 5:
        return None

    best_lam = float("inf")
    best_a, best_b = None, None

    # a + b = n - 1, a >= b >= 1
    total = n - 1  # a + b
    for b in range(1, total // 2 + 1):
        a = total - b
        lam = bridged_double_star_eigenvalue(a, b)
        if lam < best_lam:
            best_lam = lam
            best_a, best_b = a, b

    return dict(a=best_a, b=best_b, lam1=best_lam, mu=2, val=best_lam + 2)


def best_pure_double_star(n: int) -> dict | None:
    """
    Find the split (a, b) with a + b = n, a >= b >= 2, that minimises
    λ₁(S(a,b)) + 2.
    """
    if n < 4:
        return None

    best_lam = float("inf")
    best_a, best_b = None, None

    for b in range(2, n // 2 + 1):
        a = n - b
        lam = pure_double_star_eigenvalue(a, b)
        if lam < best_lam:
            best_lam = lam
            best_a, best_b = a, b

    return dict(a=best_a, b=best_b, lam1=best_lam, mu=2, val=best_lam + 2)


# ── Main computation ────────────────────────────────────────────────────────

def compute_curves(n_max: int):
    ns = np.arange(3, n_max + 1)
    bound = np.sqrt(ns - 1) + 1
    star_val = bound.copy()  # star always equals bound

    bds_val = np.full_like(ns, dtype=float, fill_value=np.nan)
    bds_splits = []
    pds_val = np.full_like(ns, dtype=float, fill_value=np.nan)
    pds_splits = []

    for i, n in enumerate(ns):
        rec = best_bridged_double_star(int(n))
        if rec is not None:
            bds_val[i] = rec["val"]
            bds_splits.append((rec["a"], rec["b"]))
        else:
            bds_splits.append((None, None))

        rec = best_pure_double_star(int(n))
        if rec is not None:
            pds_val[i] = rec["val"]
            pds_splits.append((rec["a"], rec["b"]))
        else:
            pds_splits.append((None, None))

    return ns, bound, star_val, bds_val, bds_splits, pds_val, pds_splits


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(ns, bound, star_val, bds_val, bds_splits,
                    pds_val, pds_splits, out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    # colours
    C_BOUND = "black"
    C_STAR = "#d62728"
    C_BDS = "#1f77b4"
    C_PDS = "#ff7f0e"

    # ── Panel 1: raw values ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ns, bound, "--", color=C_BOUND, linewidth=2,
            label="bound  √(n−1) + 1", zorder=3)
    ax.plot(ns, star_val, "o-", color=C_STAR, markersize=2, linewidth=1.0,
            label="star K₁,ₙ₋₁  (μ=1)", zorder=4, alpha=0.8)
    ax.plot(ns, bds_val, "s-", color=C_BDS, markersize=2, linewidth=1.0,
            label="bridged double star  (μ=2)", zorder=4, alpha=0.8)
    ax.plot(ns, pds_val, "^-", color=C_PDS, markersize=2, linewidth=1.0,
            label="pure double star  (μ=2)", zorder=4, alpha=0.8)
    ax.set_xlabel("n (number of vertices)", fontsize=11)
    ax.set_ylabel("λ₁ + μ", fontsize=11)
    ax.set_title("λ₁ + μ  vs  n", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: margin above bound ─────────────────────────────────────
    ax = axes[0, 1]
    star_margin = star_val - bound
    bds_margin = bds_val - bound
    pds_margin = pds_val - bound
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot(ns, star_margin, "o-", color=C_STAR, markersize=2, linewidth=1.0,
            label="star margin (≡ 0)", alpha=0.8)
    ax.plot(ns, bds_margin, "s-", color=C_BDS, markersize=2, linewidth=1.0,
            label="bridged double star margin", alpha=0.8)
    ax.plot(ns, pds_margin, "^-", color=C_PDS, markersize=2, linewidth=1.0,
            label="pure double star margin", alpha=0.8)
    ax.set_xlabel("n", fontsize=11)
    ax.set_ylabel("(λ₁ + μ) − (√(n−1) + 1)", fontsize=11)
    ax.set_title("Margin above bound", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight crossover for bridged double star
    valid_bds = ~np.isnan(bds_margin)
    cross_idx = np.where(valid_bds & (bds_margin < -1e-12))[0]
    if len(cross_idx) > 0:
        cross_n = ns[cross_idx[0]]
        ax.axvline(cross_n, color="#2ca02c", linewidth=1.5, linestyle=":",
                   alpha=0.7)
        ax.annotate(
            f"BDS beats bound\nat n={cross_n}",
            xy=(cross_n, bds_margin[cross_idx[0]]),
            xytext=(cross_n + max(3, len(ns) * 0.04),
                    bds_margin[cross_idx[0]] * 0.4),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#333"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#eeffee", ec="#2ca02c"),
        )

    # Highlight tie point for bridged double star
    tie_idx = np.where(valid_bds & (np.abs(bds_margin) < 1e-6))[0]
    if len(tie_idx) > 0:
        tie_n = ns[tie_idx[0]]
        ax.plot(tie_n, 0, "D", color="#2ca02c", markersize=8, zorder=5)
        ax.annotate(
            f"BDS ties star\nat n={tie_n}",
            xy=(tie_n, 0),
            xytext=(tie_n - max(5, len(ns) * 0.08), -0.15),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#333"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffee", ec="#999"),
        )

    # ── Panel 3: difference from star ───────────────────────────────────
    ax = axes[1, 0]
    bds_diff = bds_val - star_val
    pds_diff = pds_val - star_val

    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(ns, bds_diff, "s-", color=C_BDS, markersize=2, linewidth=1.2,
            label="bridged double star − star", alpha=0.85)
    ax.plot(ns, pds_diff, "^-", color=C_PDS, markersize=2, linewidth=1.2,
            label="pure double star − star", alpha=0.85)

    ax.fill_between(ns, bds_diff, 0, where=(bds_diff < 0),
                    color=C_BDS, alpha=0.15, interpolate=True)
    ax.fill_between(ns, bds_diff, 0, where=(bds_diff > 0),
                    color=C_STAR, alpha=0.10, interpolate=True)

    ax.set_xlabel("n", fontsize=11)
    ax.set_ylabel("(double star family) − (star)", fontsize=11)
    ax.set_title("Double star families vs star\n(negative = beats star)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: optimal splits ─────────────────────────────────────────
    ax = axes[1, 1]

    bds_valid = [(n, a, b) for (n, (a, b)) in zip(ns, bds_splits)
                 if a is not None]
    pds_valid = [(n, a, b) for (n, (a, b)) in zip(ns, pds_splits)
                 if a is not None]

    if bds_valid:
        bns, bas, bbs = zip(*bds_valid)
        bns, bas, bbs = np.array(bns), np.array(bas), np.array(bbs)
        ax.plot(bns, bas, "-", color=C_BDS, linewidth=1.2, alpha=0.8,
                label="BDS: a (larger hub)")
        ax.plot(bns, bbs, "--", color=C_BDS, linewidth=1.0, alpha=0.6,
                label="BDS: b (smaller hub)")

    if pds_valid:
        pns, pas, pbs = zip(*pds_valid)
        pns, pas, pbs = np.array(pns), np.array(pas), np.array(pbs)
        ax.plot(pns, pas, "-", color=C_PDS, linewidth=1.2, alpha=0.8,
                label="PDS: a (larger hub)")
        ax.plot(pns, pbs, "--", color=C_PDS, linewidth=1.0, alpha=0.6,
                label="PDS: b (smaller hub)")

    ax.plot(ns, (ns - 1) / 2, "k:", linewidth=0.8, alpha=0.4,
            label="(n−1)/2  (balanced)")
    ax.set_xlabel("n", fontsize=11)
    ax.set_ylabel("hub sizes a, b", fontsize=11)
    ax.set_title("Optimal splits for each double star family",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Star  vs  Bridged Double Star  vs  Pure Double Star  vs  Bound\n"
        "λ₁ + μ  ≥  √(n−1) + 1",
        fontsize=14, fontweight="bold", y=1.03,
    )

    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"\nSaved figure → {out_path}")
    else:
        plt.show()


# ── Textual summary ─────────────────────────────────────────────────────────

def print_table(ns, bound, star_val, bds_val, bds_splits, pds_val, pds_splits):
    header = (
        f"{'n':>4s}  {'bound':>9s}  {'star':>9s}  "
        f"{'BDS val':>9s} {'split':>8s}  "
        f"{'PDS val':>9s} {'split':>8s}  "
        f"{'BDS-bnd':>9s}  {'BDS-star':>9s}"
    )
    print()
    print(header)
    print("─" * len(header))

    for i, n in enumerate(ns):
        ba, bb = bds_splits[i]
        pa, pb = pds_splits[i]
        bds_str = f"({ba},{bb})" if ba is not None else "n/a"
        pds_str = f"({pa},{pb})" if pa is not None else "n/a"
        bds_v = bds_val[i] if not np.isnan(bds_val[i]) else float("nan")
        pds_v = pds_val[i] if not np.isnan(pds_val[i]) else float("nan")
        bds_margin = bds_v - bound[i]
        bds_vs_star = bds_v - star_val[i]
        print(
            f"{n:4d}  {bound[i]:9.4f}  {star_val[i]:9.4f}  "
            f"{bds_v:9.4f} {bds_str:>8s}  "
            f"{pds_v:9.4f} {pds_str:>8s}  "
            f"{bds_margin:+9.4f}  {bds_vs_star:+9.4f}"
        )
    print()

    # Report crossover
    valid_bds = ~np.isnan(bds_val)
    ties = np.where(valid_bds & (np.abs(bds_val - star_val) < 1e-6))[0]
    beats = np.where(valid_bds & (bds_val < star_val - 1e-9))[0]

    if len(ties) > 0:
        tn = ns[ties[0]]
        print(f"Bridged double star first TIES the star at n = {tn}")
        print(f"  star  = {star_val[ties[0]]:.6f}")
        print(f"  BDS   = {bds_val[ties[0]]:.6f}")
    if len(beats) > 0:
        bn = ns[beats[0]]
        print(f"Bridged double star first BEATS the star at n = {bn}")
        print(f"  star  = {star_val[beats[0]]:.6f}")
        print(f"  BDS   = {bds_val[beats[0]]:.6f}")
        print(f"  margin below bound = {bds_val[beats[0]] - bound[beats[0]]:+.6f}")

    valid_pds = ~np.isnan(pds_val)
    pds_beats = np.where(valid_pds & (pds_val < star_val - 1e-9))[0]
    if len(pds_beats) > 0:
        pn = ns[pds_beats[0]]
        print(f"Pure double star first BEATS the star at n = {pn}")
        print(f"  star  = {star_val[pds_beats[0]]:.6f}")
        print(f"  PDS   = {pds_val[pds_beats[0]]:.6f}")
    else:
        print("Pure double star never beats the star in this range.")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare star vs bridged/pure double star vs bound."
    )
    parser.add_argument("--nmax", type=int, default=200,
                        help="Largest n (default: 200)")
    parser.add_argument("--out", type=str, default="star_vs_doublestar.png",
                        help="Output image path ('' for interactive)")
    args = parser.parse_args()

    if args.nmax < 4:
        print("n_max must be >= 4", file=sys.stderr)
        sys.exit(1)

    ns, bound, star_val, bds_val, bds_splits, pds_val, pds_splits = \
        compute_curves(args.nmax)
    print_table(ns, bound, star_val, bds_val, bds_splits, pds_val, pds_splits)
    out = args.out if args.out else None
    plot_comparison(ns, bound, star_val, bds_val, bds_splits,
                    pds_val, pds_splits, out_path=out)


if __name__ == "__main__":
    main()
