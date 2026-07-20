# Data schema

This directory is the **source of truth**, held as human-readable CSVs. A SQLite
database (or any other index) is a *build artifact* derived from these files, not
the other way around. The guiding rule: **store only what cannot be recomputed.**

Everything here is about directed strongly regular graphs (DSRGs) realized as
Cayley graphs of small groups.

## Group identity

Groups are never given a surrogate id. A group is named by its GAP **SmallGroup
value** `(order, lib_id)`, i.e. `SmallGroup(order, lib_id)`. Because a Cayley
DSRG on a group `G` has exactly `|G|` vertices, the group order always equals the
DSRG parameter `n`. So wherever `n` is already in scope (every table below), a
group is pinned by its **`lib_id`** alone, and reconstructed in GAP as
`SmallGroup(n, lib_id)`.

Anything else about a group — its `StructureDescription` (e.g. `S3`, `D8`),
whether it is abelian, its automorphism group, its element ordering — is
recovered from GAP on demand and is never stored.

## Multicolumn (composite) keys

None of these tables uses a surrogate id. A row is identified by a **tuple of
columns**, and uniqueness is a property of the whole tuple, not of any one column:

- A **parameter** is identified by `(n, k, t, lambda, mu)`. No single field is
  unique — many rows share the same `n`, or the same `k` — but the five together
  are.
- A **group** is identified by `(n, lib_id)`: `lib_id` alone is meaningless
  (every order restarts at `1`); the pair is the SmallGroup value.

Tables refer to each other by **carrying the referenced tuple's columns**, the
composite-key analogue of a foreign key. For example a `searches` row references
both a group and a parameter, so it carries `lib_id` (the group) and
`(k, t, lambda, mu)` (the parameter) — and the shared `n` does double duty,
pinning the parameter *and*, with `lib_id`, the group. A join is then "match on
all the key columns at once" (e.g. `searches ⋈ parameters` on
`n, k, t, lambda, mu`).

When a relational database is built from these CSVs, each such tuple becomes a
composite `PRIMARY KEY (...)` / `FOREIGN KEY (...) REFERENCES ...` over exactly
those columns. Surrogate integer ids may be minted at build time for
convenience, but they are local to that build and never written back to the CSVs.

## Element encoding

A connection set (`members`) is a space-separated list of **1-based indices into
the group's element list**, using GAP's deterministic enumeration
`Elements(SmallGroup(n, lib_id))`. Index `1` is the identity and is therefore
never a member. So `members = "2 3"` denotes `{ g_2, g_3 }`.

---

## Source-of-truth CSVs

### `parameters.csv` — the DSRG parameter space

The enumerated feasible-parameter table. This is external mathematical input; it
cannot be derived from anything else in the repo.

| column   | notes                                             |
|----------|---------------------------------------------------|
| `n`      | number of vertices                                |
| `k`      | out-degree; **invariant `k < n/2`** (see below)   |
| `t`      | number of `s` in the connection set with `s⁻¹` also in it |
| `lambda` |                                                   |
| `mu`     |                                                   |
| `status` | `open` \| `known` \| `impossible`                 |

Natural key: `(n, k, t, lambda, mu)`.

`t` is stored. It is in fact functionally determined by the others through the
DSRG identity `k(k + mu - lambda) = t + (n - 1)*mu` (verified to hold for every
row), so `t = k(k + mu - lambda) - (n - 1)*mu` — but it is kept in the file for
readability and as a consistency check rather than recomputed on every read.

(This file replaces the previous split between `parameters.csv` and
`dsrg_parameters.csv`; the former was a strict subset of the latter. There is now
one file, carrying the full `open`/`known`/`impossible` enumeration.)

#### Invariant: `k < n/2`

Only parameters with out-degree `k < n/2` are stored. The **complement** of a
DSRG is again a DSRG: replacing a connection set `S` (of size `k`, excluding the
identity) with `(G \ {e}) \ S` gives a DSRG of out-degree `n - 1 - k`. So degrees
`k` and `n - 1 - k` come in complementary pairs, and `k < n/2` keeps exactly one
representative of each pair — for `n` even it excludes `k = n/2`; for `n` odd the
self-complementary degree `k = (n-1)/2` satisfies `k < n/2` and is kept. The
`k ≥ n/2` half of the space is recovered by complementation and is therefore not
stored (the same reduction applies to `searches` and `dpds` rows).

> The current CSVs predate this rule and still contain `k ≥ n/2` rows; enforcing
> the invariant (dropping them, and their complement-redundant searches/dpds) is
> part of regenerating the data to this schema.

### `searches.csv` — the computational record

One row per `(group, parameter, method)` attempt. This is a log of work
performed and cannot be regenerated without re-running every search, so it is
kept verbatim. Note this is the only place **abelian** groups appear (as negative
results); the candidate grid otherwise concerns nonabelian groups only.

| column    | notes                                                           |
|-----------|-----------------------------------------------------------------|
| `lib_id`  | group is `SmallGroup(n, lib_id)`                                 |
| `n`,`k`,`t`,`lambda`,`mu` | the parameter set (natural key into `parameters.csv`)|
| `method`  | `exhaustive` \| `ilp` \| `rrhc`                                 |
| `outcome` | `found` \| `infeasible_proof` \| `empty_proof` \| `timeout` \| `heuristic_none` |
| `num_dpds`| number of constructions found (0 for negative outcomes)         |

`is_proof` is **not stored** — an outcome definitively resolves the pair iff it is
one of `found`, `infeasible_proof`, `empty_proof` (the `timeout` /
`heuristic_none` outcomes do not).

### `dpds/nNNN.csv` — the constructions found

The actual DSRGs discovered, sharded by vertex count `n` (the zero-padded number
in the filename, e.g. `n012.csv`). Each row is one connection set found on one
group. These are search outputs and are kept verbatim.

| column          | notes                                                    |
|-----------------|----------------------------------------------------------|
| `lib_id`        | group is `SmallGroup(n, lib_id)`, with `n` from filename |
| `members`       | connection set, as element indices (see *Element encoding*) |
| `source_method` | how it was found: `exhaustive` \| `inverse` \| `rrhc` \| `ilp` (occasionally a comma-joined set, e.g. `ilp, exhaustive`) |

The parameter set and the resulting graph are **not stored** — build the Cayley
digraph of `SmallGroup(n, lib_id)` on `members` and read off `(n, k, t, lambda,
mu)` and its canonical form directly.

---

## Derived tables (rebuilt, not stored)

These were previously materialized as CSVs. They add no information and are
regenerated when a database is built:

- **`groups`** — `(n, lib_id)` → `StructureDescription`, `is_abelian`,
  `aut_order`, … Reconstructed for each `SmallGroup(n, lib_id)` via GAP. The set
  of groups needed is exactly those referenced by `searches.csv` and the dpds
  shards.

- **`group_param`** — the candidate grid. Verified to be *exactly* every
  nonabelian group of order `n` paired with every parameter of that `n`, so it is
  produced by a GAP filter (`Filtered(AllSmallGroups(n), G -> not IsAbelian(G))`)
  rather than stored.

- **`graphs`** — the distinct DSRG isomorphism classes with invariants
  (`digraph6`, `aut_order`, `is_drr`, `is_self_converse`, `num_dpds`,
  `num_groups`). Every one is reachable from a construction in `dpds/`, so the
  table is obtained by canonicalizing each construction's Cayley digraph,
  deduplicating, and computing invariants. Optionally materialize it as a cache,
  since recomputing ~2k canonical forms over ~380k constructions is expensive —
  but it holds no irreducible data.

---

## Building a database

A DB build reads the three source-of-truth CSV families above and regenerates the
derived tables via GAP + a canonical-form step. Assign surrogate integer keys at
build time if a relational schema wants them; they never round-trip back into the
CSVs.
