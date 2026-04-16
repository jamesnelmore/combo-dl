# combo-dl/src/ilp/verify_new_dsrgs.g
#
# Verify two new DSRGs found by Cayley-graph ILP on GL(2,3) = SmallGroup(48,29).
#
# Connection sets S are given as 1-indexed positions in AsList(SmallGroup(48,29)).
#
# DSRG(48,19,14,5,9) on GL(2,3)
# DSRG(48,22,17,11,9) on GL(2,3)
#
# Usage:  gap -q verify_new_dsrgs.g

LoadPackage("smallgrp");;
LoadPackage("GRAPE");;

VerifyCayleyDSRG := function(n, k, t, lambda, mu, lib_id, S_indices)
    local G, elements, id_G, S, A, I_mat, J_mat, ASquared, RHS,
          row, col, g, s, h, AutG, AutS, phi, gamma, autGamma;

    G := SmallGroup(n, lib_id);
    elements := AsList(G);
    id_G := Identity(G);

    # Build connection set from 1-indexed element positions.
    S := List(S_indices, i -> elements[i]);

    # Verify basic properties of S.
    if Size(S) <> k then
        Print("FAIL: |S| = ", Size(S), " but k = ", k, "\n");
        return false;
    fi;
    if id_G in S then
        Print("FAIL: identity is in S\n");
        return false;
    fi;

    # Build adjacency matrix A of Cay(G, S).
    A := NullMat(n, n);
    for row in [1..n] do
        g := elements[row];
        for s in S do
            h := g * s;
            col := Position(elements, h);
            A[row][col] := 1;
        od;
    od;

    # Check DSRG condition: A^2 = t*I + lambda*A + mu*(J - I - A).
    I_mat := IdentityMat(n);
    J_mat := List([1..n], i -> List([1..n], j -> 1));
    ASquared := A * A;
    RHS := t * I_mat + lambda * A + mu * (J_mat - I_mat - A);

    if ASquared = RHS then
        Print("VERIFIED: DSRG(", n, ",", k, ",", t, ",", lambda, ",", mu,
              ") on SmallGroup(", n, ",", lib_id, ") = ",
              StructureDescription(G), "\n");
        Print("  Connection set S (1-indexed): ", S_indices, "\n");
    else
        Print("FAIL: DSRG condition not satisfied for (",
              n, ",", k, ",", t, ",", lambda, ",", mu, ")\n");
        return false;
    fi;

    # Aut(G) and stabilizer of S.
    AutG := AutomorphismGroup(G);
    Print("  Aut(G) = ", StructureDescription(AutG),
          ", order ", Size(AutG), "\n");
    AutS := Filtered(AsList(AutG),
        phi -> Set(List(S, s -> Image(phi, s))) = Set(S));
    Print("  Aut(G, S) order: ", Size(AutS), "\n");

    # Full digraph automorphism group via GRAPE/nauty.
    gamma := Graph(Group(()), [1..n], OnPoints,
        function(x, y) return A[x][y] = 1; end, true);
    autGamma := AutGroupGraph(gamma);
    Print("  Aut(Gamma) order: ", Size(autGamma), "\n");
    Print("  Aut(Gamma) structure: ", StructureDescription(autGamma), "\n");

    return true;
end;;

# --- DSRG(48,19,14,5,9) on GL(2,3) ---
Print("\n=== DSRG(48,19,14,5,9) on GL(2,3) ===\n");
VerifyCayleyDSRG(48, 19, 14, 5, 9, 29,
    [3, 5, 8, 10, 16, 17, 18, 20, 23, 24, 29, 32, 35, 36, 37, 41, 44, 45, 47]);

# --- DSRG(48,22,17,11,9) on GL(2,3) ---
Print("\n=== DSRG(48,22,17,11,9) on GL(2,3) ===\n");
VerifyCayleyDSRG(48, 22, 17, 11, 9, 29,
    [2, 4, 6, 7, 9, 10, 14, 15, 19, 22, 23, 26, 28, 30, 31, 34, 38, 40, 42, 43, 45, 48]);

QUIT;
