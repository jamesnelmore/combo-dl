import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # DSRG(36, 10, 5, 2, 3) — Load and compute automorphism groups

    Loads the 14 non-isomorphic Cayley graphs, computes Aut via nauty,
    then shells out to GAP for structure descriptions and conjugacy class data.
    """)
    return


@app.cell
def _():
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from pynauty import Graph, certificate, autgrp

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
        DATA,
        Graph,
        OUT_DIR,
        Path,
        autgrp,
        certificate,
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
def _(Graph, adj_to_pynauty_fn, certificate, np):
    _JORG_ADJ = [
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1],
        [0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0],
        [1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,1,0,1,0,0,1,1],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,1,0,0],
        [0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0],
        [1,1,0,0,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
        [1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1],
        [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1],
        [0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0],
        [0,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [1,1,0,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
        [0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0],
        [1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0],
        [1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    ]
    jorg_adj = np.array(_JORG_ADJ, dtype=np.uint8)
    jorg_cert = certificate(adj_to_pynauty_fn(jorg_adj, Graph))
    return (jorg_cert,)


@app.cell
def _(
    DATA,
    Graph,
    adj_to_pynauty_fn,
    autgrp,
    certificate,
    jorg_cert,
    np,
    pd,
    perm_to_gap,
):
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
        cert = certificate(adj_to_pynauty_fn(adj, Graph))

        gens, grpsize1, grpsize2, orbits, numorbits = autgrp(adj_to_pynauty_fn(adj, Graph))
        aut_order = int(grpsize1 * 10**grpsize2)
        gap_gens = [perm_to_gap(g) for g in gens]

        graphs.append({
            "graph_id": gid,
            "is_jorg": cert == jorg_cert,
            "adj": adj,
            "aut_order": aut_order,
            "gap_gens": gap_gens,
            "num_constructions": row["num_constructions"],
            "groups": row["groups"],
            "group_lib_id": int(p["group_lib_id"]),
            "subset_index": int(p["subset_index"]),
            "numorbits": numorbits,
        })
    return (graphs,)


@app.cell
def _(Path, graphs, subprocess, tempfile):
    # Build GAP script — gets StructureDescription and conjugacy class data.
    _gap_lines = [
        'LoadPackage("smallgrp");;',
        "n := 36;;",
    ]

    for _lib_id in [9, 10]:
        _gap_lines += [
            f"G := SmallGroup(n, {_lib_id});;",
            "elements := AsList(G);;",
            "id := Identity(G);;",
            f'Print("GROUP_START {_lib_id}\\n");;',
            'Print("NAME ", StructureDescription(G), "\\n");;',
            'Print("IDENTITY ", Position(elements, id), "\\n");;',
            'Print("ORDERS_START\\n");;',
            'for i in [1..n] do Print(Order(elements[i]), "\\n");; od;;',
            'Print("ORDERS_END\\n");;',
            'Print("INV_START\\n");;',
            'for i in [1..n] do Print(Position(elements, elements[i]^-1), "\\n");; od;;',
            'Print("INV_END\\n");;',
            "cc := ConjugacyClasses(G);;",
            'Print("CLASSES_START ", Length(cc), "\\n");;',
            "for i in [1..Length(cc)] do",
            '    Print("CLASS ", i, " ", Size(cc[i]), " ", Order(Representative(cc[i])));;',
            '    for x in AsList(cc[i]) do Print(" ", Position(elements, x));; od;;',
            '    Print("\\n");;',
            "od;;",
            'Print("CLASSES_END\\n");;',
            'Print("GROUP_END\\n");;',
        ]

    for _i, _g in enumerate(graphs):
        _gens_str = ", ".join(_g["gap_gens"])
        _gap_lines.append(f"aut := Group([{_gens_str}]);;")
        _gap_lines.append(
            'Print("AUT ' + str(_i)
            + ' ", Size(aut), " ", StructureDescription(aut), "\\n");;'
        )

    _gap_lines += ['Print("ALL_DONE\\n");;', "QUIT;"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False) as _f:
        _f.write("\n".join(_gap_lines))
        _gap_path = _f.name

    gap_proc = subprocess.run(
        ["gap", "-q", _gap_path],
        capture_output=True, text=True, timeout=600,
    )
    Path(_gap_path).unlink(missing_ok=True)

    if gap_proc.returncode != 0:
        print("GAP STDERR:", gap_proc.stderr[:2000])
    return (gap_proc,)


@app.cell
def _(gap_proc):
    # Parse GAP output
    group_info = {}
    aut_info = {}

    _lines = gap_proc.stdout.splitlines()
    _li = 0

    while _li < len(_lines):
        _line = _lines[_li].strip()

        if _line.startswith("GROUP_START"):
            _lib_id = int(_line.split()[1])
            _gi = {"orders": [], "classes": [], "inv": [], "identity": 0}
            _li += 1

            while _li < len(_lines):
                _l2 = _lines[_li].strip()
                if _l2.startswith("NAME "):
                    _gi["name"] = _l2[5:]
                elif _l2.startswith("IDENTITY "):
                    _gi["identity"] = int(_l2.split()[1])
                elif _l2 == "ORDERS_START":
                    _li += 1
                    while _lines[_li].strip() != "ORDERS_END":
                        _gi["orders"].append(int(_lines[_li].strip()))
                        _li += 1
                elif _l2 == "INV_START":
                    _li += 1
                    while _lines[_li].strip() != "INV_END":
                        _gi["inv"].append(int(_lines[_li].strip()))
                        _li += 1
                elif _l2.startswith("CLASSES_START"):
                    _li += 1
                    while _lines[_li].strip() != "CLASSES_END":
                        _parts = _lines[_li].strip().split()
                        _gi["classes"].append({
                            "idx": int(_parts[1]),
                            "size": int(_parts[2]),
                            "order": int(_parts[3]),
                            "members": [int(x) for x in _parts[4:]],
                        })
                        _li += 1
                elif _l2 == "GROUP_END":
                    group_info[_lib_id] = _gi
                    break
                _li += 1

        elif _line.startswith("AUT "):
            _parts = _line.split(maxsplit=3)
            _idx = int(_parts[1])
            _order = int(_parts[2])
            _desc = _parts[3] if len(_parts) > 3 else "?"
            aut_info[_idx] = {"order": _order, "desc": _desc}

        elif _line == "ALL_DONE":
            break

        _li += 1
    return aut_info, group_info


@app.cell
def _(aut_info, graphs, group_info, np):
    # Recover connection sets and analyze
    _output = []
    _output.append("=" * 80)
    _output.append("DSRG(36, 10, 5, 2, 3) — 14 non-isomorphic Cayley graphs")
    _output.append("=" * 80)

    for _i, _g in enumerate(graphs):
        _lib_id = _g["group_lib_id"]
        _gi = group_info[_lib_id]
        _orders = _gi["orders"]
        _inv_map = _gi["inv"]
        _classes = _gi["classes"]
        _identity_pos = _gi["identity"] - 1

        _adj = _g["adj"]

        _conn_set = np.nonzero(_adj[_identity_pos])[0].tolist()
        _conn_orders = [_orders[j] for j in _conn_set]

        _conn_set_1idx = set(j + 1 for j in _conn_set)
        _classes_in_S = []
        _classes_partial = []
        for _cl in _classes:
            _members_set = set(_cl["members"])
            _overlap = _members_set & _conn_set_1idx
            if _overlap == _members_set:
                _classes_in_S.append(_cl)
            elif _overlap:
                _classes_partial.append((_cl, len(_overlap)))

        _is_normal = len(_classes_partial) == 0 and len(_classes_in_S) > 0

        _inverses_in_S = sum(1 for j in _conn_set if _inv_map[j] in _conn_set_1idx)
        _involutions_in_S = sum(1 for j in _conn_set if _orders[j] == 2)

        _ai = aut_info.get(_i, {"order": _g["aut_order"], "desc": "?"})

        _tag = " *** JORG'S GRAPH ***" if _g["is_jorg"] else ""
        _output.append(f"\n{'─' * 70}")
        _output.append(f"Graph {_g['graph_id']}{_tag}")
        _output.append(f"  Source: group {_lib_id} ({_gi['name']}), subset index {_g['subset_index']}")
        _output.append(f"  Constructions: {_g['num_constructions']}  |  Groups: {_g['groups']}")
        _output.append(f"  Aut group: {_ai['desc']}  (order {_ai['order']})")
        _output.append(f"  Vertex orbits under Aut: {_g['numorbits']}")
        _output.append(f"  Element orders in S: {sorted(_conn_orders)}")
        _order_dist = {}
        for _o in _conn_orders:
            _order_dist[_o] = _order_dist.get(_o, 0) + 1
        _output.append(f"  Order distribution: {dict(sorted(_order_dist.items()))}")
        _output.append(f"  |S ∩ S⁻¹| = {_inverses_in_S} (t=5 expected)")
        _output.append(f"  Involutions in S: {_involutions_in_S}")
        if _is_normal:
            _class_desc = ", ".join(
                f"class {c['idx']}(order {c['order']}, size {c['size']})"
                for c in _classes_in_S
            )
            _output.append(f"  NORMAL Cayley graph — S is union of classes: {_class_desc}")
        else:
            _output.append(f"  NOT a normal Cayley graph")
            if _classes_in_S:
                _full_desc = ", ".join(
                    f"class {c['idx']}(order {c['order']}, size {c['size']})"
                    for c in _classes_in_S
                )
                _output.append(f"    Full classes in S: {_full_desc}")
            if _classes_partial:
                _partial_desc = ", ".join(
                    f"class {c['idx']}(order {c['order']}, size {c['size']}, {count}/{c['size']} in S)"
                    for c, count in _classes_partial
                )
                _output.append(f"    Partial classes in S: {_partial_desc}")

    _output.append(f"\n{'=' * 80}")
    _output.append("Done")
    print("\n".join(_output))
    return


@app.cell
def _(OUT_DIR, aut_info, graphs, mo):
    import matplotlib.pyplot as plt
    import networkx as nx

    graphs_sorted = sorted(graphs, key=lambda g: (not g["is_jorg"], g["graph_id"]))

    fig, axes = plt.subplots(3, 5, figsize=(25, 16))
    _axes = axes.flat

    for _idx, _ax in enumerate(_axes):
        if _idx >= len(graphs_sorted):
            _ax.axis("off")
            continue

        _g = graphs_sorted[_idx]
        _ai = aut_info.get(graphs.index(_g), {"order": _g["aut_order"], "desc": "?"})
        _G = nx.from_numpy_array(_g["adj"], create_using=nx.DiGraph)

        _pos = nx.spring_layout(_G, seed=42, k=1.5, iterations=100)

        _is_jorg = _g["is_jorg"]
        _edge_color = "#DAA520" if _is_jorg else "#4477AA"
        _node_color = "#FFD700" if _is_jorg else "#88CCEE"

        nx.draw_networkx_edges(
            _G, _pos, ax=_ax, alpha=0.15, edge_color=_edge_color,
            arrows=True, arrowsize=3, width=0.4, node_size=120,
            connectionstyle="arc3,rad=0.1", min_source_margin=4, min_target_margin=4,
        )
        nx.draw_networkx_nodes(
            _G, _pos, ax=_ax, node_size=120, node_color=_node_color,
            edgecolors="black", linewidths=0.8,
        )

        _title = f"id={_g['graph_id']}"
        if _is_jorg:
            _title += " (Jorg)"
        _title += f"\nAut: {_ai['desc']} (order {_ai['order']})"
        _title += f"\n{_g['num_constructions']} constructions"
        _ax.set_title(_title, fontsize=10, fontweight="bold" if _is_jorg else "normal")
        _ax.axis("off")

    _axes[14].axis("off")

    fig.suptitle(
        "All 14 non-isomorphic DSRG(36, 10, 5, 2, 3) Cayley graphs",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(OUT_DIR / "dsrg_36_10_5_2_3_all.pdf", bbox_inches="tight")
    _png_path = OUT_DIR / "dsrg_36_10_5_2_3_all.png"
    fig.savefig(_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mo.image(src=_png_path)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
