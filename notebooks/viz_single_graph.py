import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    def _find_root():
        for anchor in [Path.cwd()]:
            d = anchor
            for _ in range(5):
                if (d / "cayley_data").is_dir():
                    return d
                d = d.parent
        raise FileNotFoundError("Cannot locate cayley_data/")

    ROOT = _find_root()
    DATA = ROOT / "cayley_data" / "hpc_cayley"
    OUT_DIR = ROOT / "notebooks"
    return DATA, OUT_DIR, np, plt, nx


@app.cell
def _(DATA, np, nx):
    import subprocess

    # Graph 361: (C3 x C3) : C4 only, Aut order 36, new construction
    adj = np.load(DATA / "dsrg_36_10_5_2_3_g9.npz")["adjacency"][0]
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    # Classify edges: mutual (both directions) vs one-way
    mutual_edges = set()
    oneway_edges = set()
    for u, v in G.edges():
        if G.has_edge(v, u):
            if (v, u) not in mutual_edges:
                mutual_edges.add((u, v))
        else:
            oneway_edges.add((u, v))

    # --- Get multiplication table from GAP for coset layout ---
    _root = DATA.parent.parent
    gap_script = _root / "src" / "cayley_search" / "group_tables.g"
    gap_input = "n := 36;;\ninclude_abelian := false;;\n" + gap_script.read_text()
    proc = subprocess.run(
        ["gap", "-q"], input=gap_input, capture_output=True, text=True, timeout=30
    )
    gap_lines = proc.stdout.strip().split("\n")

    def _collect_ints(start, stop):
        vals, j = [], start
        while j < len(gap_lines):
            s = gap_lines[j].strip()
            if s == stop:
                return vals, j
            vals.extend(int(t) for t in s.split())
            j += 1
        raise ValueError(f"Expected {stop}")

    mul = None
    idx = 0
    while idx < len(gap_lines):
        if gap_lines[idx].strip().startswith("GROUP_START 9 "):
            idx += 1  # IDENTITY line
            idx += 1  # INV line (may wrap)
            idx += 1
            while idx < len(gap_lines) and gap_lines[idx].strip() != "TABLE_START":
                idx += 1
            idx += 1  # skip TABLE_START
            table_vals, idx = _collect_ints(idx, "TABLE_END")
            mul = np.array(table_vals).reshape(36, 36)
            break
        idx += 1

    # Connection set
    S = list(np.where(adj[0] == 1)[0])

    # Generate order-6 subgroup from elements 2 and 8 of connection set
    def _gen_subgroup(gens):
        sub = {0}
        sub.update(gens)
        changed = True
        while changed:
            changed = False
            for a in list(sub):
                for b in list(sub):
                    p = int(mul[a, b])
                    if p not in sub:
                        sub.add(p)
                        changed = True
        return frozenset(sub)

    H = _gen_subgroup([int(S[0]), int(S[2])])  # elements 2,8 -> order 6

    # Compute left cosets
    cosets = []
    covered = set()
    for g in range(36):
        if g not in covered:
            coset = frozenset(int(mul[g, h]) for h in H)
            cosets.append(sorted(coset))
            covered |= coset
    coset_of = {}
    for ci, c in enumerate(cosets):
        for v in c:
            coset_of[v] = ci

    # --- Vertex ordering: cosets contiguous, sorted by coset rep ---
    # Within each coset, order elements by their index in H (consistent algebraic order)
    H_sorted = sorted(H)
    vertex_order = []
    for _c in cosets:
        # _c is already sorted; keep that order
        vertex_order.extend(_c)

    # Map vertex -> position on the line
    rank = {v: i for i, v in enumerate(vertex_order)}

    # --- Hex cluster positions for coset layout ---
    n_cosets = len(cosets)
    _cluster_radius = 4.0
    _node_radius = 0.8
    _center_angles = np.linspace(0, 2 * np.pi, n_cosets, endpoint=False) - np.pi / 2

    hex_pos = {}
    for _ci, _c in enumerate(cosets):
        _cx = _cluster_radius * np.cos(_center_angles[_ci])
        _cy = _cluster_radius * np.sin(_center_angles[_ci])
        _local_angles = np.linspace(0, 2 * np.pi, len(_c), endpoint=False)
        for _j, _v in enumerate(_c):
            hex_pos[_v] = np.array([
                _cx + _node_radius * np.cos(_local_angles[_j]),
                _cy + _node_radius * np.sin(_local_angles[_j]),
            ])

    # --- Vertex ordering for arc diagram: cosets contiguous ---
    vertex_order = []
    for _c in cosets:
        vertex_order.extend(_c)
    rank = {v: i for i, v in enumerate(vertex_order)}

    return G, coset_of, cosets, hex_pos, mutual_edges, oneway_edges, rank, vertex_order


@app.cell
def _(G, OUT_DIR, coset_of, cosets, hex_pos, mutual_edges, nx, oneway_edges, plt):
    # --- Hex cluster layout figure ---
    _coset_colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4"]
    _node_colors = [_coset_colors[coset_of[v]] for v in range(36)]

    fig_hex, ax_hex = plt.subplots(figsize=(10, 10), facecolor="white")

    # Mutual edges: blue, no arrows
    nx.draw_networkx_edges(
        G, hex_pos, edgelist=list(mutual_edges), ax=ax_hex,
        alpha=0.3, edge_color="#3b82f6", width=1.0,
        arrows=False, node_size=300,
    )
    # One-way edges: red with arrows
    nx.draw_networkx_edges(
        G, hex_pos, edgelist=list(oneway_edges), ax=ax_hex,
        alpha=0.15, edge_color="#ef4444", width=0.5,
        arrows=True, arrowsize=4, node_size=300,
        connectionstyle="arc3,rad=0.1", min_source_margin=7, min_target_margin=7,
    )
    # Nodes colored by coset
    nx.draw_networkx_nodes(
        G, hex_pos, ax=ax_hex, node_size=300,
        node_color=_node_colors, edgecolors="#333", linewidths=1.2, alpha=0.9,
    )

    _Line2D_h = __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D
    _Patch_h = __import__("matplotlib.patches", fromlist=["Patch"]).Patch
    _leg = [
        _Line2D_h([0], [0], color="#3b82f6", lw=2, alpha=0.5, label="Mutual (symmetric)"),
        _Line2D_h([0], [0], color="#ef4444", lw=1.5, alpha=0.35, label="One-way (asymmetric)"),
    ] + [
        _Patch_h(facecolor=_coset_colors[_i], edgecolor="#333", alpha=0.8, label=f"Coset {_i}")
        for _i in range(len(cosets))
    ]
    ax_hex.legend(handles=_leg, loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#ccc")
    ax_hex.set_title(
        "DSRG(36, 10, 5, 2, 3) over (C\u2083 \u00d7 C\u2083) \u22c9 C\u2084"
        "  \u2014  coset cluster layout",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax_hex.axis("off")
    fig_hex.tight_layout()

    fig_hex.savefig(OUT_DIR / "dsrg_36_10_cosets.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig_hex.savefig(OUT_DIR / "dsrg_36_10_cosets.pdf", bbox_inches="tight", facecolor="white")
    fig_hex
    return


@app.cell
def _(G, OUT_DIR, cosets, mutual_edges, nx, oneway_edges, plt):
    # --- Shell layout: cosets as concentric rings ---
    shell_pos = nx.shell_layout(G, nlist=cosets)

    fig_shell, ax_shell = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_edges(
        G, shell_pos, edgelist=list(mutual_edges), ax=ax_shell,
        alpha=0.3, edge_color="#3b82f6", width=1.0,
        arrows=False, node_size=250,
    )
    nx.draw_networkx_edges(
        G, shell_pos, edgelist=list(oneway_edges), ax=ax_shell,
        alpha=0.15, edge_color="#ef4444", width=0.5,
        arrows=True, arrowsize=4, node_size=250,
        connectionstyle="arc3,rad=0.1", min_source_margin=6, min_target_margin=6,
    )
    nx.draw_networkx_nodes(
        G, shell_pos, ax=ax_shell, node_size=250,
        node_color="#dbeafe", edgecolors="#1e3a5f", linewidths=1.4,
    )

    ax_shell.axis("off")
    fig_shell.tight_layout()

    fig_shell.savefig(OUT_DIR / "dsrg_36_10_shell.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig_shell.savefig(OUT_DIR / "dsrg_36_10_shell.pdf", bbox_inches="tight", transparent=True)
    fig_shell
    return


@app.cell
def _(OUT_DIR, coset_of, cosets, mutual_edges, np, oneway_edges, plt, rank, vertex_order):
    # --- Arc diagram figure ---
    _coset_colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4"]

    fig_arc, ax_arc = plt.subplots(figsize=(16, 7), facecolor="white")

    # x-positions: small gaps within cosets, larger gaps between cosets
    _xpos = np.zeros(36)
    _x = 0.0
    for _ci, _c in enumerate(cosets):
        for _v in _c:
            _xpos[rank[_v]] = _x
            _x += 1.0
        _x += 1.5  # extra gap between cosets

    def _draw_arc(_u, _v, _above, _color, _alpha, _lw):
        _xu, _xv = _xpos[rank[_u]], _xpos[rank[_v]]
        _mid = (_xu + _xv) / 2
        _span = abs(_xv - _xu)
        _height = _span * 0.5
        _sign = 1 if _above else -1
        _theta = np.linspace(0, np.pi, 40)
        _xs = _mid + (_span / 2) * np.cos(_theta) * (1 if _xv > _xu else -1)
        _ys = _sign * _height * np.sin(_theta)
        ax_arc.plot(_xs, _ys, color=_color, alpha=_alpha, lw=_lw, solid_capstyle="round")

    # Mutual edges above (blue)
    for _u, _v in mutual_edges:
        _draw_arc(_u, _v, True, "#3b82f6", 0.25, 0.8)

    # One-way edges below (red)
    for _u, _v in oneway_edges:
        _draw_arc(_u, _v, False, "#ef4444", 0.15, 0.5)

    # Nodes on the x-axis
    for _i, _v in enumerate(vertex_order):
        ax_arc.plot(_xpos[_i], 0, "o", color=_coset_colors[coset_of[_v]],
                    markersize=8, markeredgecolor="#333", markeredgewidth=1.0, zorder=5)

    # Coset labels
    _idx = 0
    for _ci, _c in enumerate(cosets):
        _xm = (_xpos[_idx] + _xpos[_idx + len(_c) - 1]) / 2
        ax_arc.text(_xm, -1.8, f"Coset {_ci}", ha="center", va="top",
                    fontsize=8, color=_coset_colors[_ci], fontweight="bold")
        _idx += len(_c)

    _Line2D_a = __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D
    ax_arc.legend(handles=[
        _Line2D_a([0], [0], color="#3b82f6", lw=2, alpha=0.5, label="Mutual (above)"),
        _Line2D_a([0], [0], color="#ef4444", lw=1.5, alpha=0.35, label="One-way (below)"),
    ], loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#ccc")

    ax_arc.set_title(
        "DSRG(36, 10, 5, 2, 3) over (C\u2083 \u00d7 C\u2083) \u22c9 C\u2084"
        "  \u2014  arc diagram, vertices ordered by coset of \u27e8s\u2081, s\u2083\u27e9",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax_arc.set_xlim(_xpos[0] - 1.5, _xpos[-1] + 1.5)
    ax_arc.axis("off")
    ax_arc.set_aspect("auto")
    fig_arc.tight_layout()

    fig_arc.savefig(OUT_DIR / "dsrg_36_10_arc.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig_arc.savefig(OUT_DIR / "dsrg_36_10_arc.pdf", bbox_inches="tight", facecolor="white")
    fig_arc
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
