# cayley_search/metadata.g
#
# Enumerate nonabelian groups of order n and, for each, compute the number of
# Aut(G)-orbit representatives among k-subsets of G\{e} that also satisfy the
# t-reciprocity constraint |S ∩ S⁻¹| = t.  The Python orchestrator uses this
# count to create correctly-sized job blocks.
#
# Usage: pipe via stdin with the preamble "n := X;; k := Y;; t := Z;;" injected
# by the Python caller, then the contents of this file.
#
# Output tokens (one per line):
#   GROUP_COUNT <count>
#   GROUP <filtered_index> <library_id> <num_reps> <StructureDescription>
#   ...
#   META_DONE

LoadPackage("smallgrp");;

groups    := Filtered(AllSmallGroups(n), G -> not IsAbelian(G));;
numGroups := Size(groups);;

Print("GROUP_COUNT ", numGroups, "\n");

for i in [1..numGroups] do
    Gi  := groups[i];
    gid := IdGroup(Gi);

    id_i    := Identity(Gi);;
    nonId_i := Filtered(AsList(Gi), x -> x <> id_i);;

    # ── OPTIMIZATION 2 (pre-filter by t) ────────────────────────────────────
    # Any connection set S must satisfy |S ∩ S⁻¹| = t, so discard all k-subsets
    # that fail this before doing any further work.  Aut(G) preserves this count
    # (φ(s)⁻¹ = φ(s⁻¹), so |φ(S) ∩ φ(S)⁻¹| = |S ∩ S⁻¹|), meaning the
    # t-valid subsets form a union of complete Aut(G)-orbits — safe to restrict.
    tValid := Filtered(Combinations(nonId_i, k),
                       S -> Size(Filtered(S, s -> s^-1 in S)) = t);;

    # ── OPTIMIZATION 3 (Aut(G)-orbit reduction) ──────────────────────────────
    # Two connection sets S and φ(S) (for φ ∈ Aut(G)) yield isomorphic Cayley
    # graphs.  We therefore only need one representative per Aut(G)-orbit.
    # |Aut(G)| can be large (often 54+ for order-27 groups), giving a
    # proportional reduction in the number of candidates to check.
    AutGi   := AutomorphismGroup(Gi);;
    orbs    := Orbits(AutGi, tValid, OnSets);;
    numReps := Size(orbs);;

    # num_reps is placed before the name because StructureDescription can contain
    # spaces, so the Python parser splits on maxsplit=4 and takes the remainder.
    Print("GROUP ", i, " ", gid[2], " ", numReps, " ", StructureDescription(Gi), "\n");
od;

Print("META_DONE\n");
QUIT;
