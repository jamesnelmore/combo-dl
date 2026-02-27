#!/usr/bin/env python3
"""Parallelized Cayley graph DSRG search.

Splits the connector-set search space into blocks of configurable size and
dispatches them across worker processes, each of which runs a parameterised
GAP subprocess.  Progress is reported via tqdm as blocks complete.

Usage::

    python -m ilp.cayley_search.search n k t lambda mu \
        [--block-size 100000] [--workers NUM_CPUS] \
        [--timeout 3600] [--log FILE] [--stop-on-first]
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from math import comb
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from tqdm import tqdm

# ---------------------------------------------------------------------------
# GAP script templates
# ---------------------------------------------------------------------------

_WORKER_TEMPLATE = r"""
LoadPackage("smallgrp");;

n         := {n};;
k         := {k};;
t         := {t};;
lambda    := {lambda_};;
mu        := {mu};;
libId     := {group_lib_id};;
startIdx  := {start_idx};;
endIdx    := {end_idx};;

G        := SmallGroup(n, libId);;
elements := AsList(G);;
id       := Identity(G);;
nonId    := Filtered(elements, x -> x <> id);;

# ── Precompute a fast element → list-position lookup ─────────────────────────
# Position(list, x) scans the list in O(n).  For an n=30 group with k=12,
# that call appears k²=144 times per candidate — paying O(n·k²) per candidate.
# A dictionary gives O(1) amortised lookup, cutting the inner loop to O(k²).
elemPos := NewDictionary(id, true);;
for i in [1..n] do
    AddDictionary(elemPos, elements[i], i);
od;;

idPos := LookupDictionary(elemPos, id);;   # position of the identity element

# ── OPTIMIZATION 2 (t-filter) + OPTIMIZATION 3 (Aut(G)-orbit reduction) ──────
#
# Instead of iterating over all C(n-1, k) raw subsets, we build the much
# smaller list of Aut(G)-orbit representatives restricted to t-valid subsets.
#
# Step 1 — t-filter: keep only k-subsets S with |S ∩ S⁻¹| = t.
#   Aut(G) preserves this count (φ(s)⁻¹ = φ(s⁻¹)), so t-valid subsets form
#   a union of complete Aut(G)-orbits and the restriction is well-defined.
#
# Step 2 — orbit reduction: two connection sets S and φ(S) give isomorphic
#   Cayley graphs, so checking one representative per orbit is sufficient.
#   The factor of reduction is roughly |Aut(G)| / average-stabiliser-size,
#   often 6–54× for the group orders we care about.
#
# Combined, these two steps reduce C(n-1,k) candidates to a list of orbit
# representatives that is typically 10–100× smaller.  The startIdx/endIdx
# block boundaries now refer to indices in this *reduced* list, whose size
# was computed once in the metadata phase and communicated to this worker.
tValid  := Filtered(Combinations(nonId, k),
                    S -> Size(Filtered(S, s -> s^-1 in S)) = t);;
AutG    := AutomorphismGroup(G);;
orbs    := Orbits(AutG, tValid, OnSets);;
allReps := List(orbs, orb -> orb[1]);;   # one canonical rep per orbit

slice := allReps{{[startIdx..endIdx]}};;

Print("BLOCK_START ", startIdx, " ", endIdx, "\n");

ProcessBlock := function()
    local S, inS, fTable, s, sp, h, pos, ok, i, fh, A, row, col, checked;
    checked := 0;
    for S in slice do
        checked := checked + 1;
        # Note: every S in slice already satisfies |S ∩ S⁻¹| = t (from tValid),
        # so the i = idPos branch of the check below is always satisfied.
        # We keep it for correctness and as documentation of the invariant.

        # ── OPTIMIZATION 1: check DSRG condition at the identity only ────────
        #
        # Every Cayley graph Cay(G, S) is vertex-transitive: left-multiplication
        # by any g ∈ G is a graph automorphism taking vertex x to vertex g·x.
        # Therefore the DSRG equation
        #
        #   A² = t·I + λ·A + μ·(J − I − A)
        #
        # holds at ALL vertices iff it holds at the identity e alone.
        #
        # Row e of A² counts, for each h ∈ G, the number of directed 2-paths
        # e → s → h.  Such a path exists iff s ∈ S and s⁻¹·h ∈ S, i.e.
        # h = s·s' for some s, s' ∈ S.  Define:
        #
        #   f(h) := |{{(s, s') ∈ S × S : s·s' = h}}|
        #
        # The three DSRG conditions at e are then:
        #   f(e)  = t      (t reciprocal neighbours: both s and s⁻¹ in S)
        #   f(h)  = λ      for every h ∈ S      (out-neighbour of e)
        #   f(h)  = μ      for every h ∉ S∪{{e}}  (non-neighbour of e)
        #
        # Cost: O(k²) to build f, O(n) to check — versus O(n³) for A·A.

        # Boolean membership table: inS[i] is true iff elements[i] ∈ S.
        # BlistList builds this in one pass from the integer positions.
        inS := BlistList([1..n], List(S, s -> LookupDictionary(elemPos, s)));;

        # Accumulate f(h) over all ordered pairs (s, s') ∈ S × S.
        fTable := ListWithIdenticalEntries(n, 0);;
        for s in S do
            for sp in S do
                h   := s * sp;
                pos := LookupDictionary(elemPos, h);
                fTable[pos] := fTable[pos] + 1;
            od;
        od;

        # Check the three conditions; break on the first violation.
        ok := true;;
        for i in [1..n] do
            fh := fTable[i];
            if i = idPos then
                if fh <> t      then ok := false; break; fi;
            elif inS[i] then
                if fh <> lambda then ok := false; break; fi;
            else
                if fh <> mu     then ok := false; break; fi;
            fi;
        od;

        if ok then
            Print("FOUND\n");
            # Build the full adjacency matrix only for confirmed solutions —
            # this cost is negligible relative to the search.
            A := List([1..n], i -> ListWithIdenticalEntries(n, 0));;
            for row in [1..n] do
                for s in S do
                    h   := elements[row] * s;
                    col := LookupDictionary(elemPos, h);
                    A[row][col] := 1;
                od;
            od;
            Print("ADJ_START\n");
            for row in A do
                Print(JoinStringsWithSeparator(List(row, String), ","), "\n");
            od;
            Print("ADJ_END\n");
        fi;
    od;
    Print("BLOCK_DONE ", checked, "\n");
end;;

ProcessBlock();;
QUIT;
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupInfo:
    """Metadata for a single nonabelian group."""

    filtered_index: int
    library_id: int
    name: str
    # Number of Aut(G)-orbit representatives among t-valid k-subsets.
    # This is the true search-space size for this group after optimizations 2+3.
    num_reps: int


@dataclass(frozen=True, slots=True)
class Job:
    """One unit of work: a block of connector sets for a specific group."""

    n: int
    k: int
    t: int
    lambda_: int
    mu: int
    group_lib_id: int
    group_name: str
    start_idx: int
    end_idx: int
    timeout: float


@dataclass(slots=True)
class BlockResult:
    """Result returned by a single worker."""

    status: str
    group_lib_id: int
    group_name: str
    start_idx: int
    end_idx: int
    checked: int = 0
    adjacency: list[list[int]] | None = None


# ---------------------------------------------------------------------------
# Shared state for stop-on-first
# ---------------------------------------------------------------------------

_found_flag: mp.Value | None = None  # type: ignore[type-arg]


def _init_pool(flag: mp.Value) -> None:  # type: ignore[type-arg]
    """Pool initializer — stash the shared flag in a module global."""
    global _found_flag  # noqa: PLW0603
    _found_flag = flag


# ---------------------------------------------------------------------------
# Metadata phase
# ---------------------------------------------------------------------------


def _run_metadata(n: int, k: int, t: int, logger: logging.Logger) -> list[GroupInfo]:
    """Run metadata.g to enumerate nonabelian groups of order *n*.

    Also computes, for each group, the number of Aut(G)-orbit representatives
    among t-valid k-subsets (the effective search-space size after opts 2+3).

    Returns:
        List of ``GroupInfo`` for each nonabelian group found.
    """
    script = Path(__file__).with_name("metadata.g")
    # Inject n, k, t so metadata.g can compute the orbit counts.
    script_text = f"n := {n};; k := {k};; t := {t};;\n" + script.read_text()
    proc = subprocess.run(
        ["gap", "-q"],
        input=script_text,
        capture_output=True,
        text=True,
        timeout=300,  # orbit computation can take longer than simple enumeration
        check=False,
    )
    if proc.returncode != 0:
        logger.error("metadata.g failed:\n%s", proc.stderr)
        sys.exit(1)

    groups: list[GroupInfo] = []
    for line in proc.stdout.splitlines():
        if line.startswith("GROUP_COUNT"):
            count = int(line.split()[1])
            logger.info("%d nonabelian group(s) of order %d", count, n)
        elif line.startswith("GROUP "):
            # Format: GROUP <filtered_index> <library_id> <num_reps> <name...>
            # num_reps precedes the name because StructureDescription can contain spaces.
            parts = line.split(maxsplit=4)
            groups.append(
                GroupInfo(
                    filtered_index=int(parts[1]),
                    library_id=int(parts[2]),
                    num_reps=int(parts[3]),
                    name=parts[4] if len(parts) > 4 else "?",
                )
            )
        elif line == "META_DONE":
            break
    return groups


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_worker(job: Job) -> BlockResult:
    """Execute a single GAP block-check in a subprocess.

    Returns:
        A ``BlockResult`` with status, checked count, and optional adjacency.
    """
    if _found_flag is not None and _found_flag.value:
        return BlockResult(
            status="skipped",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )

    code = _WORKER_TEMPLATE.format(
        n=job.n,
        k=job.k,
        t=job.t,
        lambda_=job.lambda_,
        mu=job.mu,
        group_lib_id=job.group_lib_id,
        start_idx=job.start_idx,
        end_idx=job.end_idx,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".g", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.Popen(
            ["gap", "-q", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Drain stderr so the pipe buffer never blocks GAP.
        def _drain() -> None:
            assert proc.stderr is not None
            for _ in proc.stderr:
                pass

        t = threading.Thread(target=_drain, daemon=True)
        t.start()

        adjacency: list[list[int]] = []
        capturing = False
        checked = 0
        found = False

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BLOCK_DONE"):
                checked = int(line.split()[1])
            elif line == "FOUND":
                found = True
            elif line == "ADJ_START":
                capturing = True
                adjacency = []
            elif line == "ADJ_END":
                capturing = False
            elif capturing:
                adjacency.append([int(x) for x in line.split(",")])

        proc.wait(timeout=job.timeout)
        t.join(timeout=2)

        if found and _found_flag is not None:
            _found_flag.value = 1

        return BlockResult(
            status="found" if found else "done",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
            checked=checked,
            adjacency=adjacency if found else None,
        )

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return BlockResult(
            status="timeout",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )
    except Exception as exc:
        proc.kill()
        proc.wait()
        return BlockResult(
            status=f"error: {exc}",
            group_lib_id=job.group_lib_id,
            group_name=job.group_name,
            start_idx=job.start_idx,
            end_idx=job.end_idx,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _save_adjacency(
    adj: list[list[int]],
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    group_lib_id: int,
    logger: logging.Logger,
) -> Path:
    filename = Path(f"adjacency_{n}_{k}_{t}_{lambda_}_{mu}_g{group_lib_id}.json")
    filename.write_text(json.dumps(adj, indent=2), encoding="utf-8")
    logger.info("Adjacency matrix saved to %s", filename)
    return filename


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def search(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    *,
    block_size: int = 100_000,
    num_workers: int | None = None,
    timeout: float = 3600,
    logfile: Path = Path("cayley_search.log"),
    stop_on_first: bool = False,
) -> list[BlockResult]:
    """Run the parallelized Cayley DSRG search.

    Args:
        n: Number of vertices.
        k: Degree (in- and out-).
        t: Number of reciprocal neighbours.
        lambda_: Lambda DSRG parameter.
        mu: Mu DSRG parameter.
        block_size: Connector sets per job.
        num_workers: Worker processes (defaults to CPU count).
        timeout: Wall-clock limit in seconds.
        logfile: Path for the log file.
        stop_on_first: Stop after the first DSRG is found.

    Returns:
        List of BlockResult objects for every dispatched block.
    """
    # -- Logger ---------------------------------------------------------------
    logger = logging.getLogger("cayley_search")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # -- Feasibility ----------------------------------------------------------
    lhs = k * (k - lambda_) - t
    rhs = (n - k - 1) * mu
    if lhs != rhs:
        logger.error(
            "Parameters (n=%d, k=%d, t=%d, lambda=%d, mu=%d) fail the necessary condition "
            "k(k - lambda) - t = (n - k - 1) * mu  (%d != %d)",
            n, k, t, lambda_, mu, lhs, rhs,
        )
        return []

    # -- Metadata -------------------------------------------------------------
    start_time = time.perf_counter()
    logger.info("DSRG(%d, %d, %d, %d, %d) — starting search", n, k, t, lambda_, mu)

    groups = _run_metadata(n, k, t, logger)
    if not groups:
        logger.info("No nonabelian groups of order %d — nothing to search", n)
        return []

    # Log raw search-space sizes for comparison.
    sets_per_group_raw = comb(n - 1, k)
    total_reps = sum(g.num_reps for g in groups)
    logger.info(
        "C(%d, %d) = %s raw subsets/group → %s orbit-rep(s) total after Aut(G) + t-filter",
        n - 1, k, f"{sets_per_group_raw:,}", f"{total_reps:,}",
    )

    # -- Job list -------------------------------------------------------------
    # Each group's search space is now g.num_reps orbit representatives (not
    # the full C(n-1,k)).  Workers slice into the ordered list of orbit reps
    # recomputed deterministically in GAP from the same (n, k, t, libId) inputs.
    worker_timeout = max(timeout / (num_workers or mp.cpu_count()) * 2, 60)
    jobs: list[Job] = []
    for g in groups:
        if g.num_reps == 0:
            logger.info("Group %s (lib_id=%d): 0 orbit reps — skipping", g.name, g.library_id)
            continue
        for start in range(1, g.num_reps + 1, block_size):
            end = min(start + block_size - 1, g.num_reps)
            jobs.append(
                Job(
                    n=n, k=k, t=t, lambda_=lambda_, mu=mu,
                    group_lib_id=g.library_id,
                    group_name=g.name,
                    start_idx=start,
                    end_idx=end,
                    timeout=worker_timeout,
                )
            )

    total_blocks = len(jobs)
    workers = num_workers or mp.cpu_count()
    logger.info(
        "%d block(s) across %d group(s), %d worker(s), block size %s",
        total_blocks, len(groups), workers, f"{block_size:,}",
    )

    # -- Dispatch -------------------------------------------------------------
    flag = mp.Value("b", 0) if stop_on_first else None

    results: list[BlockResult] = []
    total_checked = 0
    found_count = 0

    with mp.Pool(
        processes=workers,
        initializer=_init_pool,
        initargs=(flag if flag is not None else mp.Value("b", 0),),
    ) as pool:
        imap_it = pool.imap_unordered(_run_worker, jobs, chunksize=1)
        try:
            for result in tqdm(imap_it, total=total_blocks, desc="Blocks", unit="blk"):
                results.append(result)
                total_checked += result.checked

                if result.status == "found":
                    found_count += 1
                    assert result.adjacency is not None
                    tqdm.write(
                        f"  DSRG found!  group={result.group_name} "
                        f"(lib_id={result.group_lib_id})  "
                        f"block=[{result.start_idx}..{result.end_idx}]"
                    )
                    _save_adjacency(
                        result.adjacency, n, k, t, lambda_, mu,
                        result.group_lib_id, logger,
                    )
                elif result.status not in {"done", "skipped"}:
                    tqdm.write(f"  Worker issue: {result.status}")

        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating workers")
            pool.terminate()

    # -- Summary --------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    logger.info(
        "Search complete in %s — %s sets checked, %d DSRG(s) found",
        _fmt_elapsed(elapsed), f"{total_checked:,}", found_count,
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallelized Cayley-graph DSRG search via GAP.",
    )
    # Positional params — optional when --params-file is used
    parser.add_argument("n", type=int, nargs="?", help="Number of vertices")
    parser.add_argument("k", type=int, nargs="?", help="Degree (in- and out-)")
    parser.add_argument("t", type=int, nargs="?", help="Number of reciprocal neighbours")
    parser.add_argument("lambda_", type=int, nargs="?", metavar="lambda", help="lambda parameter")
    parser.add_argument("mu", type=int, nargs="?", help="mu parameter")
    parser.add_argument(
        "--params-file", type=Path, default=None, metavar="FILE",
        help="File with one 'n k t lambda mu' per line; runs a search for each",
    )
    parser.add_argument(
        "--block-size", type=int, default=100_000,
        help="Connector sets per worker block (default: 100000)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--timeout", type=float, default=3600,
        help="Wall-clock limit in seconds (default: 3600)",
    )
    parser.add_argument(
        "--log", type=Path, default=Path("cayley_search.log"),
        help="Log file path (default: cayley_search.log); ignored when --params-file is used",
    )
    parser.add_argument(
        "--stop-on-first", action="store_true",
        help="Stop after the first DSRG is found",
    )
    args = parser.parse_args()

    shared = dict(
        block_size=args.block_size,
        num_workers=args.workers,
        timeout=args.timeout,
        stop_on_first=args.stop_on_first,
    )

    if args.params_file is not None:
        param_sets: list[tuple[int, int, int, int, int]] = []
        with args.params_file.open() as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    parser.error(f"{args.params_file}:{lineno}: expected 5 values, got {len(parts)}")
                try:
                    param_sets.append(tuple(int(p) for p in parts))  # type: ignore[misc]
                except ValueError as exc:
                    parser.error(f"{args.params_file}:{lineno}: {exc}")

        for pn, pk, pt, pl, pm in param_sets:
            logfile = Path(f"cayley_search_{pn}_{pk}_{pt}_{pl}_{pm}.log")
            search(pn, pk, pt, pl, pm, logfile=logfile, **shared)  # type: ignore[arg-type]
    else:
        missing = [name for name, val in [("n", args.n), ("k", args.k), ("t", args.t),
                                           ("lambda", args.lambda_), ("mu", args.mu)]
                   if val is None]
        if missing:
            parser.error(f"missing positional argument(s): {', '.join(missing)} "
                         f"(or use --params-file)")
        search(
            args.n, args.k, args.t, args.lambda_, args.mu,
            logfile=args.log,
            **shared,  # type: ignore[arg-type]
        )
