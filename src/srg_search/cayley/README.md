# Exhaustive Cayley Graph DSRG Search
This app exhaustively searches the subsets of a given group to see if any produce a Cayley digraph which is also an $(n,k,t,\lambda,\mu)$ strongly regular digraph (DSRG).

## Layout
- `cli.py` — Typer command definitions (`single`, `one-group`, `cleanup`).
- `orchestration.py` — top-level workflows: `run_single` (all groups for one
  parameter set, with progress.csv resume) and `run_one_group` (one group).
  Handles file I/O and progress bars.
- `subset_enumeration.py` — pure-compute core: GAP loader, t-valid subset
  generation on GPU, DSRG check. `search_one_group` does the per-group work.
- `aggregate.py`, `dedup.py` — post-search aggregation and isomorphism
  deduplication (used by the `cleanup` command, currently a stub).
- `group_tables.g`, `metadata.g` — GAP scripts invoked by `load_group_tables`.
- `analysis.py`, `pure_gap_search.py` — exploratory / reference code, not on
  the live path.

## Running

```
srg-search cayley single <n> <k> <t> <lambda> <mu> [--output-dir DIR]
srg-search cayley one-group <n> <k> <t> <lambda> <mu> <group_id>
```
