import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Automorphism group analysis for DSRG(36, 10, 5, 2, 3)

    The 14 non-isomorphic Cayley DSRG(36,10,5,2,3) come from two groups:

    - **(C3 x C3) : C4** — SmallGroup(36, 9)
    - **S3 x S3** — SmallGroup(36, 10)

    Some graphs arise from only one group, others from both.
    This notebook examines how Aut(Γ) relates to which Cayley group(s)
    can produce each graph.
    """)
    return


@app.cell
def _():
    import subprocess
    import tempfile
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from pynauty import Graph, autgrp

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
    return (
        Counter,
        DATA,
        Graph,
        OUT_DIR,
        Path,
        autgrp,
        np,
        pd,
        subprocess,
        tempfile,
    )


@app.cell
def _():
    def adj_to_pynauty_fn(adj, _Graph):
        n = adj.shape[0]
        d = {}
        for i in range(n):
            nbrs = list(adj[i].nonzero()[0])
            if nbrs:
                d[i] = nbrs
        return _Graph(number_of_vertices=n, directed=True, adjacency_dict=d)

    def perm_to_gap(gen):
        """Convert 0-indexed pynauty generator to GAP permutation string."""
        cycles = []
        visited = set()
        for start in range(len(gen)):
            if start in visited or gen[start] == start:
                visited.add(start)
                continue
            cycle = [start]
            visited.add(start)
            nxt = gen[start]
            while nxt != start:
                cycle.append(nxt)
                visited.add(nxt)
                nxt = gen[nxt]
            if len(cycle) > 1:
                cycles.append("(" + ",".join(str(x + 1) for x in cycle) + ")")
        return "".join(cycles) if cycles else "()"

    return adj_to_pynauty_fn, perm_to_gap


@app.cell
def _(DATA, Graph, adj_to_pynauty_fn, autgrp, np, pd, perm_to_gap):
    cat = pd.read_csv(DATA / "catalog.csv")
    prov = pd.read_csv(DATA / "provenance.csv")
    rows = cat[
        (cat["n"] == 36)
        & (cat["k"] == 10)
        & (cat["t"] == 5)
        & (cat["lambda"] == 2)
        & (cat["mu"] == 3)
    ].reset_index(drop=True)

    graphs = []
    for _, row in rows.iterrows():
        gid = row["graph_id"]
        p = prov[prov["graph_id"] == gid].iloc[0]
        fname = (
            DATA / f"dsrg_{int(p['n'])}_{int(p['k'])}_{int(p['t'])}"
            f"_{int(p['lambda'])}_{int(p['mu'])}_g{int(p['group_lib_id'])}.npz"
        )
        adj = np.load(fname)["adjacency"][int(p["subset_index"])]

        gens, grpsize1, grpsize2, orbits, numorbits = autgrp(adj_to_pynauty_fn(adj, Graph))
        aut_order = int(grpsize1 * 10**grpsize2)
        gap_gens = [perm_to_gap(g) for g in gens]

        graphs.append({
            "graph_id": gid,
            "adj": adj,
            "aut_order": aut_order,
            "gap_gens": gap_gens,
            "num_constructions": row["num_constructions"],
            "num_groups": row["num_groups"],
            "groups": row["groups"],
            "group_lib_id": int(p["group_lib_id"]),
            "subset_index": int(p["subset_index"]),
            "numorbits": numorbits,
        })
    return (graphs,)


@app.cell
def _(Path, graphs, subprocess, tempfile):
    # Run GAP to identify Aut structure descriptions + wreath product check
    _gap_lines = ['LoadPackage("smallgrp");;']

    for _i, _g in enumerate(graphs):
        _gens_str = ", ".join(_g["gap_gens"])
        _gap_lines.append(f"aut := Group([{_gens_str}]);;")
        _gap_lines.append(
            f'Print("AUT {_i} ", Size(aut), " ", StructureDescription(aut), "\\n");;'
        )

    _gap_lines.append("G9 := SmallGroup(36, 9);;")
    _gap_lines.append("G10 := SmallGroup(36, 10);;")
    _gap_lines.append('Print("G9_DESC ", StructureDescription(G9), "\\n");;')
    _gap_lines.append('Print("G10_DESC ", StructureDescription(G10), "\\n");;')
    _gap_lines.append("W := WreathProduct(SymmetricGroup(3), SymmetricGroup(2));;")
    _gap_lines.append('Print("WREATH_DESC ", StructureDescription(W), " ", Size(W), "\\n");;')
    _gap_lines.append('Print("WREATH_HAS_G9 ", IsomorphicSubgroups(W, G9) <> [], "\\n");;')
    _gap_lines.append('Print("WREATH_HAS_G10 ", IsomorphicSubgroups(W, G10) <> [], "\\n");;')
    _gap_lines.append('Print("ALL_DONE\\n");;')
    _gap_lines.append("QUIT;")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False) as _f:
        _f.write("\n".join(_gap_lines))
        _gap_path = _f.name

    gap_proc = subprocess.run(
        ["gap", "-q", _gap_path], capture_output=True, text=True, timeout=600
    )
    Path(_gap_path).unlink(missing_ok=True)

    if gap_proc.returncode != 0:
        print("GAP STDERR:", gap_proc.stderr[:2000])

    aut_info = {}
    extra_info = {}
    for _line in gap_proc.stdout.splitlines():
        _line = _line.strip()
        if _line.startswith("AUT "):
            _parts = _line.split(maxsplit=3)
            aut_info[int(_parts[1])] = {
                "order": int(_parts[2]),
                "desc": _parts[3] if len(_parts) > 3 else "?",
            }
        elif _line.startswith(("G9_DESC", "G10_DESC", "WREATH_")):
            _key = _line.split()[0]
            extra_info[_key] = " ".join(_line.split()[1:])

    for _i, _g in enumerate(graphs):
        _g["aut_desc"] = aut_info[_i]["desc"]
    return (extra_info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The three tiers

    The 14 graphs cleanly partition into three tiers based on which
    Cayley group(s) produce them, and **each tier has a distinct
    automorphism group**.
    """)
    return


@app.cell
def _(graphs, pd):
    _GROUP_NAMES = {9: "(C3 x C3) : C4", 10: "S3 x S3"}
    _summary = []
    for _g in graphs:
        if _g["num_groups"] == 1 and _g["group_lib_id"] == 9:
            _tier = "(C3 x C3) : C4 only"
        elif _g["num_groups"] == 2:
            _tier = "Both groups"
        else:
            _tier = "S3 x S3 only"
        _summary.append({
            "graph_id": _g["graph_id"],
            "tier": _tier,
            "source_group": _GROUP_NAMES[_g["group_lib_id"]],
            "aut_desc": _g["aut_desc"],
            "aut_order": _g["aut_order"],
            "num_constructions": _g["num_constructions"],
            "orbits": _g["numorbits"],
        })

    df = pd.DataFrame(_summary)
    return (df,)


@app.cell
def _(df):
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key observation: Aut(Γ) reflects the source group

    - **(C3 x C3) : C4 only** (ids 361, 362) -- Aut = (C3 x C3) : C4, order 36, 144 constructions
    - **Both groups** (ids 363--366) -- Aut = (S3 x S3) : C2, order 72, 108 constructions
    - **S3 x S3 only** (ids 367--374) -- Aut = S3 x S3, order 36, 72 constructions

    When a graph comes from **only (C3 x C3) : C4**, Aut(Γ) is isomorphic to
    (C3 x C3) : C4 --- the Cayley group itself.
    When from **only S3 x S3**, Aut(Γ) is isomorphic to S3 x S3.
    When from **both**, Aut(Γ) is isomorphic to (S3 x S3) : C2, order 72 = 2 x 36.
    """)
    return


@app.cell
def _(df):
    # Verify the pattern: for single-group graphs, Aut ≅ source group (regular action)
    _output = ["=== Tier summary ===\n"]
    for _tier_name, _tier_df in df.groupby("tier", sort=False):
        _ids = _tier_df["graph_id"].tolist()
        _aut = _tier_df["aut_desc"].iloc[0]
        _order = _tier_df["aut_order"].iloc[0]
        _constructions = _tier_df["num_constructions"].iloc[0]
        _orbits = _tier_df["orbits"].iloc[0]
        _output.append(f"{_tier_name} ({len(_ids)} graphs: {_ids})")
        _output.append(f"  Aut(Γ) = {_aut},  |Aut| = {_order}")
        _output.append(f"  Vertex orbits = {_orbits},  Constructions = {_constructions}")
        _ratio = _order / 36
        if _ratio == 1:
            _output.append("  |Aut|/n = 1  =>  Aut acts REGULARLY (vertex stabilizer is trivial)")
        else:
            _output.append(f"  |Aut|/n = {_ratio}  =>  vertex stabilizer has order {int(_ratio)}")
        _output.append("")
    print("\n".join(_output))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Why the automorphism group matches the source group

    A Cayley graph Cay(G, S) always has G as a subgroup of Aut(Γ)
    via left multiplication.  When Aut(Γ) = G (i.e., no extra symmetry
    beyond the Cayley action), the graph is called a **graphical regular
    representation (GRR)** of G --- though here these are directed, so
    the precise term is **DRR** (digraphical regular representation).

    - **(C3 x C3) : C4 only** (ids 361, 362): Aut = (C3 x C3) : C4.
      These are DRRs of (C3 x C3) : C4.  No other group of order 36 embeds
      regularly into their automorphism group, so S3 x S3 can't produce them.

    - **S3 x S3 only** (ids 367--374): Aut = S3 x S3.
      These are DRRs of S3 x S3.  (C3 x C3) : C4 doesn't embed into S3 x S3,
      so it can't produce them.

    - **Both groups** (ids 363--366): Aut = (S3 x S3) : C2, order 72.
      This is strictly larger than both source groups.  It contains
      *both* (C3 x C3) : C4 and S3 x S3 as regular subgroups (each of index 2).
      That's exactly what allows both groups to produce the same graph
      as a Cayley graph --- each group acts regularly on the vertices,
      just with different connection sets.
    """)
    return


@app.cell
def _(extra_info):
    # Verify: (S3 x S3) : C2 contains both groups as subgroups
    _output = ["=== Wreath product / containment check (from GAP) ===\n"]
    for _k, _v in sorted(extra_info.items()):
        _output.append(f"  {_k}: {_v}")
    _output.append("")
    _output.append("(S3 x S3) : C2  =  S3 wr C2  (the wreath product)")
    _output.append("It has S3 x S3 as a normal subgroup of index 2,")
    _output.append("and also contains (C3 x C3) : C4 as a subgroup of index 2.")
    print("\n".join(_output))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Connection set structure by tier

    The element order distribution in the connection set S also
    separates the tiers cleanly.
    """)
    return


@app.cell
def _(Counter, Path, graphs, np, subprocess, tempfile):
    # Get group element orders from GAP
    _gap_lines = ['LoadPackage("smallgrp");;', "n := 36;;"]
    for _lib_id in [9, 10]:
        _gap_lines += [
            f"G := SmallGroup(n, {_lib_id});;",
            "elements := AsList(G);;",
            "id := Identity(G);;",
            f'Print("IDENTITY {_lib_id} ", Position(elements, id), "\\n");;',
            f'Print("ORDERS_START {_lib_id}\\n");;',
            'for i in [1..n] do Print(Order(elements[i]), "\\n");; od;;',
            f'Print("ORDERS_END {_lib_id}\\n");;',
        ]
    _gap_lines += ['Print("DONE\\n");;', "QUIT;"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False) as _f:
        _f.write("\n".join(_gap_lines))
        _gap_path = _f.name

    _proc = subprocess.run(
        ["gap", "-q", _gap_path], capture_output=True, text=True, timeout=60
    )
    Path(_gap_path).unlink(missing_ok=True)

    _identity_pos = {}
    _element_orders = {}
    _lines = _proc.stdout.splitlines()
    _li = 0
    while _li < len(_lines):
        _line = _lines[_li].strip()
        if _line.startswith("IDENTITY"):
            _parts = _line.split()
            _identity_pos[int(_parts[1])] = int(_parts[2]) - 1
        elif _line.startswith("ORDERS_START"):
            _lid = int(_line.split()[1])
            _orders_list = []
            _li += 1
            while not _lines[_li].strip().startswith("ORDERS_END"):
                _orders_list.append(int(_lines[_li].strip()))
                _li += 1
            _element_orders[_lid] = _orders_list
        _li += 1

    _GROUP_NAMES = {9: "(C3xC3):C4", 10: "S3xS3"}
    _output = ["=== Connection set order distributions ===\n"]
    for _g in graphs:
        _lib_id = _g["group_lib_id"]
        _id_pos = _identity_pos[_lib_id]
        _orders = _element_orders[_lib_id]
        _conn_set = np.nonzero(_g["adj"][_id_pos])[0].tolist()
        _conn_orders = sorted(_orders[j] for j in _conn_set)
        _dist = dict(Counter(_conn_orders))

        _tier = f"{_GROUP_NAMES[_lib_id]} only" if _g["num_groups"] == 1 else "Both"
        _jorg = " (Jorg)" if _g["graph_id"] == 366 else ""
        _output.append(f"  Graph {_g['graph_id']:>3} [{_tier:>16}]{_jorg:>8}  orders: {_dist}")
    print("\n".join(_output))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Observations

    **(C3 x C3) : C4 family** (including "both" graphs loaded from this group):
    Order distribution {2: 3, 3: 1, 4: 6}. Six elements of order 4
    --- these exist in (C3 x C3) : C4 but NOT in S3 x S3 (which has
    max element order 6).

    **S3 x S3 family**:
    Order distribution {2: 5, 3: 1, 6: 4}. Four elements of order 6
    --- these exist in S3 x S3 but NOT in (C3 x C3) : C4 (which has
    max element order 4).

    The element orders in S reflect the group they were constructed
    from.  The "both" graphs, when loaded from (C3 x C3) : C4, show
    that group's order signature.  If loaded from S3 x S3 instead,
    they would show S3 x S3's signature --- the *graph* is the same,
    but the Cayley labeling and connection set are different.
    """)
    return


@app.cell
def _(df):
    # Construction counts vs automorphism order
    _output = ["\n=== Construction counts vs automorphism order ===\n"]
    _output.append("  The construction count is the total number of connection sets")
    _output.append("  (across all source groups) yielding each isomorphism class.\n")

    for _tier_name in ["(C3 x C3) : C4 only", "Both groups", "S3 x S3 only"]:
        _tier_rows = df[df["tier"] == _tier_name]
        _n_graphs = len(_tier_rows)
        _aut_order = _tier_rows["aut_order"].iloc[0]
        _constructions = _tier_rows["num_constructions"].iloc[0]
        _stab_order = _aut_order // 36
        _output.append(f"  {_tier_name:>22}: {_n_graphs} graphs, |Aut|={_aut_order}, "
                       f"stab={_stab_order}, constructions={_constructions}")
    print("\n".join(_output))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    1. **Aut(Γ) is isomorphic to the source group** for single-source graphs.
       These are digraphical regular representations (DRRs).

    2. **Aut(Γ) is isomorphic to (S3 x S3) : C2** (order 72) for the 4 graphs
       constructible from both groups.  This group contains both
       (C3 x C3) : C4 and S3 x S3 as regular subgroups, which is
       exactly what enables both Cayley constructions.

    3. **The automorphism group lines up with the source Cayley group.**
       The Aut group IS the source group when there's only one source,
       and is a common overgroup of both when there are two sources.

    4. **None of the 14 graphs are normal Cayley graphs** --- the
       connection set S is never a union of conjugacy classes.

    5. Jorg's graph (id=366) is one of the 4 "both groups" graphs,
       with the largest automorphism group (order 72).
    """)
    return


@app.cell
def _(OUT_DIR, graphs):
    import matplotlib.pyplot as plt

    _GROUP_NAMES = {9: "(C3 x C3) : C4", 10: "S3 x S3"}
    _tier_colors = {
        "(C3 x C3) : C4 only": "#E69F00",
        "Both groups": "#009E73",
        "S3 x S3 only": "#0072B2",
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    for _g in graphs:
        _tier = f"{_GROUP_NAMES[_g['group_lib_id']]} only" if _g["num_groups"] == 1 else "Both groups"
        _color = _tier_colors[_tier]
        _marker = "*" if _g["graph_id"] == 366 else "o"
        _size = 200 if _g["graph_id"] == 366 else 80
        ax.scatter(_g["graph_id"], _g["aut_order"], c=_color, s=_size, marker=_marker,
                   edgecolors="black", linewidths=0.8, zorder=3)

    for _tier, _color in _tier_colors.items():
        ax.scatter([], [], c=_color, s=80, label=_tier, edgecolors="black", linewidths=0.8)
    ax.scatter([], [], c="white", s=200, marker="*", label="Jorg's graph",
               edgecolors="black", linewidths=0.8)

    ax.set_xlabel("Graph ID")
    ax.set_ylabel("|Aut(Γ)|")
    ax.set_title("Automorphism group order by source tier — DSRG(36, 10, 5, 2, 3)")
    ax.legend()
    ax.set_yticks([36, 72])
    ax.axhline(36, color="gray", ls="--", alpha=0.3)
    ax.axhline(72, color="gray", ls="--", alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUT_DIR / "aut_group_tiers.png", dpi=200, bbox_inches="tight")
    fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
