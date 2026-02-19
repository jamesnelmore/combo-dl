"""
Check whether every DAG on 6 vertices admits a lexicographically ordered
adjacency matrix under some simultaneous row-column permutation.
"""

from itertools import permutations
import multiprocessing as mp
import os
import time
import numpy as np
from tqdm import tqdm

N = 6

# All possible directed edges (i -> j), no self-loops
all_edges = [(i, j) for i in range(N) for j in range(N) if i != j]
num_edges = len(all_edges)  # 30 for N=6

CHUNK_SIZE = 1 << 14  # 16384 masks per chunk


def has_cycle(adj: np.ndarray) -> bool:
    """Check if directed graph has a cycle using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * N

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in range(N):
            if adj[u, v]:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(N))


def is_lex_ordered(adj: np.ndarray) -> bool:
    """Check if rows of adj are in non-decreasing lexicographic order."""
    for i in range(N - 1):
        for k in range(N):
            if adj[i, k] < adj[i + 1, k]:
                break
            if adj[i, k] > adj[i + 1, k]:
                return False
    return True


def permute_matrix(adj: np.ndarray, perm: tuple) -> np.ndarray:
    """Apply simultaneous row-column permutation."""
    return adj[np.ix_(perm, perm)]


def admits_lex_ordering(adj: np.ndarray) -> bool:
    """Check if any vertex permutation yields a lex-ordered matrix."""
    return any(
        is_lex_ordered(permute_matrix(adj, perm))
        for perm in permutations(range(N))
    )


def process_chunk(args: tuple[int, int]) -> tuple[int, list[np.ndarray]]:
    """Process masks in [start, end); return (dag_count, counterexamples)."""
    start, end = args
    dag_count = 0
    counterexamples = []
    for mask in range(start, end):
        adj = np.zeros((N, N), dtype=int)
        for bit, (i, j) in enumerate(all_edges):
            if mask & (1 << bit):
                adj[i, j] = 1

        if has_cycle(adj):
            continue

        dag_count += 1

        if not admits_lex_ordering(adj):
            counterexamples.append(adj.copy())

    return dag_count, counterexamples


def main() -> None:
    total = 1 << num_edges
    num_workers = mp.cpu_count()

    # Build (start, end) pairs — tiny to pickle, no list-of-masks needed
    ranges = [(s, min(s + CHUNK_SIZE, total)) for s in range(0, total, CHUNK_SIZE)]
    num_chunks = len(ranges)

    print(f"N={N}, edges={num_edges}, total masks={total:,}")
    print(f"Workers: {num_workers}, chunks: {num_chunks:,}, chunk size: {CHUNK_SIZE:,}")
    print()

    total_dags = 0
    counterexamples: list[np.ndarray] = []
    start_time = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_chunk, ranges, chunksize=2)
        for i, (dag_count, ces) in enumerate(
            tqdm(results, total=num_chunks, desc="Chunks", ncols=80, unit="chunk"), 1
        ):
            total_dags += dag_count
            counterexamples.extend(ces)

            if i % max(1, num_chunks // 20) == 0:
                elapsed = time.time() - start_time
                rate = i * CHUNK_SIZE / elapsed
                eta = (total - i * CHUNK_SIZE) / rate if rate > 0 else 0
                tqdm.write(
                    f"  [{i}/{num_chunks}] DAGs so far: {total_dags:,} | "
                    f"counterexamples: {len(counterexamples)} | "
                    f"{rate/1e6:.2f}M masks/s | ETA {eta:.0f}s"
                )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({total/elapsed/1e6:.2f}M masks/s)")
    print(f"Total DAGs on {N} vertices: {total_dags}")
    print(f"DAGs without lex ordering: {len(counterexamples)}")

    if counterexamples:
        print("\nCounterexamples:")
        for i, adj in enumerate(counterexamples[:10]):
            print(f"\n--- Counterexample {i + 1} ---")
            print(adj)
    else:
        print("\nAll DAGs on 6 vertices admit a lex-ordered adjacency matrix.")


if __name__ == "__main__":
    main()
