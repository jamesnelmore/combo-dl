#!/usr/bin/env python3
"""Analyze the 14 unique DSRG(36,10,5,2,3) Cayley graphs."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pynauty import Graph, certificate, autgrp


def adj_to_pynauty(adj):
    n = adj.shape[0]
    d = {}
    for i in range(n):
        nbrs = np.nonzero(adj[i])[0].tolist()
        if nbrs:
            d[i] = nbrs
    return Graph(number_of_vertices=n, directed=True, adjacency_dict=d)


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


# ── Jorg's adjacency matrix for DSRG(36,10,5,2,3) ────────────────────────────
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

jorg_cert = certificate(adj_to_pynauty(jorg_adj))

# ── Load the 14 unique graphs ────────────────────────────────────────────────
cat = pd.read_csv("cayley_data/hpc_cayley/catalog.csv")
prov = pd.read_csv("cayley_data/hpc_cayley/provenance.csv")
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
        f"cayley_data/hpc_cayley/dsrg_{int(p['n'])}_{int(p['k'])}_{int(p['t'])}"
        f"_{int(p['lambda'])}_{int(p['mu'])}_g{int(p['group_lib_id'])}.npz"
    )
    adj = np.load(fname)["adjacency"][int(p["subset_index"])]
    cert = certificate(adj_to_pynauty(adj))

    gens, grpsize1, grpsize2, orbits, numorbits = autgrp(adj_to_pynauty(adj))
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

# ── Build GAP script ─────────────────────────────────────────────────────────
# Use JSON output from GAP to avoid line-wrapping issues.
# GAP's GapDoc package has no JSON, so we emit structured text with
# one-value-per-Print and explicit sentinel lines.

gap_lines = [
    'LoadPackage("smallgrp");;',
    "n := 36;;",
]

for lib_id in [9, 10]:
    gap_lines += [
        f"G := SmallGroup(n, {lib_id});;",
        "elements := AsList(G);;",
        "id := Identity(G);;",
        # Group name on its own line
        f'Print("GROUP_START {lib_id}\\n");;',
        'Print("NAME ", StructureDescription(G), "\\n");;',
        # Identity position
        'Print("IDENTITY ", Position(elements, id), "\\n");;',
        # Element orders: one per line to avoid wrapping
        'Print("ORDERS_START\\n");;',
        'for i in [1..n] do Print(Order(elements[i]), "\\n");; od;;',
        'Print("ORDERS_END\\n");;',
        # Inverse map: one per line
        'Print("INV_START\\n");;',
        'for i in [1..n] do Print(Position(elements, elements[i]^-1), "\\n");; od;;',
        'Print("INV_END\\n");;',
        # Conjugacy classes
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

# Identify each automorphism group
for i, g in enumerate(graphs):
    gens_str = ", ".join(g["gap_gens"])
    gap_lines.append(f"aut := Group([{gens_str}]);;")
    gap_lines.append(
        'Print("AUT ' + str(i)
        + ' ", Size(aut), " ", StructureDescription(aut), "\\n");;'
    )

gap_lines += ['Print("ALL_DONE\\n");;', "QUIT;"]

gap_script = "\n".join(gap_lines)

with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False) as f:
    f.write(gap_script)
    gap_path = f.name

print("Running GAP...")
proc = subprocess.run(
    ["gap", "-q", gap_path],
    capture_output=True,
    text=True,
    timeout=600,
)

if proc.returncode != 0:
    print("GAP STDERR:", proc.stderr[:2000])

# ── Parse GAP output ─────────────────────────────────────────────────────────
group_info = {}
aut_info = {}

lines = proc.stdout.splitlines()
li = 0

while li < len(lines):
    line = lines[li].strip()

    if line.startswith("GROUP_START"):
        lib_id = int(line.split()[1])
        gi = {"orders": [], "classes": [], "inv": [], "identity": 0}
        li += 1

        while li < len(lines):
            l2 = lines[li].strip()
            if l2.startswith("NAME "):
                gi["name"] = l2[5:]
            elif l2.startswith("IDENTITY "):
                gi["identity"] = int(l2.split()[1])
            elif l2 == "ORDERS_START":
                li += 1
                while lines[li].strip() != "ORDERS_END":
                    gi["orders"].append(int(lines[li].strip()))
                    li += 1
            elif l2 == "INV_START":
                li += 1
                while lines[li].strip() != "INV_END":
                    gi["inv"].append(int(lines[li].strip()))
                    li += 1
            elif l2.startswith("CLASSES_START"):
                li += 1
                while lines[li].strip() != "CLASSES_END":
                    parts = lines[li].strip().split()
                    # CLASS idx size order member1 member2 ...
                    gi["classes"].append({
                        "idx": int(parts[1]),
                        "size": int(parts[2]),
                        "order": int(parts[3]),
                        "members": [int(x) for x in parts[4:]],
                    })
                    li += 1
            elif l2 == "GROUP_END":
                group_info[lib_id] = gi
                break
            li += 1

    elif line.startswith("AUT "):
        parts = line.split(maxsplit=3)
        idx = int(parts[1])
        order = int(parts[2])
        desc = parts[3] if len(parts) > 3 else "?"
        aut_info[idx] = {"order": order, "desc": desc}

    elif line == "ALL_DONE":
        break

    li += 1

# ── Recover connection sets and analyze ──────────────────────────────────────
print("\n" + "=" * 80)
print("DSRG(36, 10, 5, 2, 3) — 14 non-isomorphic Cayley graphs")
print("=" * 80)

for i, g in enumerate(graphs):
    lib_id = g["group_lib_id"]
    gi = group_info[lib_id]
    orders = gi["orders"]    # 0-indexed python list, but values are 1-indexed GAP positions
    inv_map = gi["inv"]      # inv_map[i] = 1-indexed position of inverse of element at position i+1
    classes = gi["classes"]
    identity_pos = gi["identity"] - 1  # convert to 0-indexed

    adj = g["adj"]
    n = adj.shape[0]

    # Connection set S: neighbors of identity in the adjacency matrix
    conn_set = np.nonzero(adj[identity_pos])[0].tolist()  # 0-indexed

    # Element orders in the connection set
    conn_orders = [orders[j] for j in conn_set]

    # Check if S is a union of conjugacy classes
    conn_set_1idx = set(j + 1 for j in conn_set)
    classes_in_S = []
    classes_partial = []
    for cl in classes:
        members_set = set(cl["members"])
        overlap = members_set & conn_set_1idx
        if overlap == members_set:
            classes_in_S.append(cl)
        elif overlap:
            classes_partial.append((cl, len(overlap)))

    is_normal = len(classes_partial) == 0 and len(classes_in_S) > 0

    # Check inverse structure
    inverses_in_S = sum(1 for j in conn_set if inv_map[j] in conn_set_1idx)
    involutions_in_S = sum(1 for j in conn_set if orders[j] == 2)

    # Automorphism group
    ai = aut_info.get(i, {"order": g["aut_order"], "desc": "?"})

    tag = " *** JORG'S GRAPH ***" if g["is_jorg"] else ""
    print(f"\n{'─' * 70}")
    print(f"Graph {g['graph_id']}{tag}")
    print(f"  Source: group {lib_id} ({gi['name']}), subset index {g['subset_index']}")
    print(f"  Constructions: {g['num_constructions']}  |  Groups: {g['groups']}")
    print(f"  Aut group: {ai['desc']}  (order {ai['order']})")
    print(f"  Vertex orbits under Aut: {g['numorbits']}")
    print(f"  Element orders in S: {sorted(conn_orders)}")
    order_dist = {}
    for o in conn_orders:
        order_dist[o] = order_dist.get(o, 0) + 1
    print(f"  Order distribution: {dict(sorted(order_dist.items()))}")
    print(f"  |S ∩ S⁻¹| = {inverses_in_S} (t=5 expected)")
    print(f"  Involutions in S: {involutions_in_S}")
    if is_normal:
        class_desc = ", ".join(
            f"class {c['idx']}(order {c['order']}, size {c['size']})"
            for c in classes_in_S
        )
        print(f"  NORMAL Cayley graph — S is union of classes: {class_desc}")
    else:
        print(f"  NOT a normal Cayley graph")
        if classes_in_S:
            full_desc = ", ".join(
                f"class {c['idx']}(order {c['order']}, size {c['size']})"
                for c in classes_in_S
            )
            print(f"    Full classes in S: {full_desc}")
        if classes_partial:
            partial_desc = ", ".join(
                f"class {c['idx']}(order {c['order']}, size {c['size']}, {count}/{c['size']} in S)"
                for c, count in classes_partial
            )
            print(f"    Partial classes in S: {partial_desc}")

Path(gap_path).unlink(missing_ok=True)
print(f"\n{'=' * 80}")
print("Done")
