# combo-dl/src/ilp/cayley_graphs.g
#
# This script searches for a directed strongly regular graph (DSRG)
# with parameters (n, k, t, lambda, mu) among the Cayley graphs of
# all nonabelian groups of order n.
#
# A directed graph with adjacency matrix A is a DSRG with these parameters if:
# 1. It is a k-regular directed graph on n vertices.
# 2. The matrix equation
#        A^2 = t*I + lambda*A + mu*(J - I - A)
#    holds, where I is the identity matrix and J is the all-ones matrix.
#
# Usage:
#     gap -q cayley_graphs.g n k t lambda mu

# Bind ARGS to the positional integer arguments from the command line.
# GAPInfo.CommandLineArguments contains all args passed to the gap binary
# (including the binary name, flags, and the script filename itself).
# We extract only the entries that look like non-negative integers, which
# are the n, k, t, lambda, mu values the caller passes after the script name.
ARGS := Filtered(GAPInfo.SystemCommandLine,
    s -> Length(s) > 0 and ForAll(s, c -> c in "0123456789"));

CheckCayleyGraphsForDSRG := function()
    local n, k, t, lambda, mu, groups, numGroups, J, I, foundGraph, i, G, groupId, elements, id, nonIdElements, S, numSets, setCounter, A, g, s, h, row, col, ASquared, RHS;

    if Length(ARGS) < 5 then
        Error("Usage: gap cayley_graphs.g n k t lambda mu");
    fi;

    n := Int(ARGS[1]);
    k := Int(ARGS[2]);
    t := Int(ARGS[3]);
    lambda := Int(ARGS[4]);
    mu := Int(ARGS[5]);

    # Early feasibility check for DSRG parameters:
    # Necessary condition derived from row-summing A^2 = tI + lambda*A + mu*(J - I - A):
    # each row of A^2 sums to k^2, and the right-hand side sums to t + lambda*k + mu*(n-1-k),
    # giving k^2 = t + lambda*k + mu*(n - k - 1), i.e. k*(k - lambda) - t = (n - k - 1)*mu.
    if k * (k - lambda) - t <> (n - k - 1) * mu then
        Print("Parameter set (n=", n, ", k=", k, ", t=", t,
              ", lambda=", lambda, ", mu=", mu, ") fails the necessary condition\n");
        Print("k*(k - lambda) - t = (n - k - 1)*mu.\n");
        Print("Aborting search.\n");
        return;
    fi;

    foundGraph := false;

    # Load the small groups library if it's not already loaded.
    if not IsBound(SmallGroupsInformation) then
        RequirePackage("smallgrp");
    fi;

    groups := Filtered(AllSmallGroups(n), G -> not IsAbelian(G));
    numGroups := Size(groups);
    Print("Found ", numGroups, " nonabelian groups of order ", n, ".\n");

    # Pre-calculate the all-ones matrix J and identity matrix I.
    J := List([1..n], i -> List([1..n], j -> 1));
    I := IdentityMat(n);

    # Loop through all groups.
    for i in [1..numGroups] do
        G := groups[i];
        groupId := IdGroup(G);
        Print("\n--- Checking Group ", i, "/", numGroups, ": Id=", groupId, " ---\n");

        elements := AsList(G);
        id := Identity(G);
        nonIdElements := Filtered(elements, el -> el <> id);

        numSets := Binomial(n - 1, k);
        Print("Iterating through ", numSets, " possible connection sets...\n");
        setCounter := 0;

        # Loop through all combinations for the connection set.
        for S in Combinations(nonIdElements, k) do
            setCounter := setCounter + 1;
            if setCounter mod 100000 = 0 and setCounter > 0 then
                Print("  ... checked ", setCounter, "/", numSets, " sets\n");
            fi;

            # Construct the adjacency matrix A.
            A := List([1..n], i -> List([1..n], j -> 0));
            for row in [1..n] do
                g := elements[row];
                for s in S do
                    h := g * s;
                    col := Position(elements, h);
                    A[row][col] := 1;
                od;
            od;

            # Check the DSRG condition: A^2 = t*I + lambda*A + mu*(J - I - A)
            ASquared := A * A;
            RHS := t * I + lambda * A + mu * (J - I - A);

            if ASquared = RHS then
                Print("\n==================================================\n");
                Print("!!! FOUND A MATCHING CAYLEY GRAPH !!!\n");
                Print("Group ID: ", groupId, "\n");
                Print("Group Structure: ", StructureDescription(G), "\n");
                Print("Connection Set S (elements are 1-based indices from AsList(G)):\n");
                Print(Sort(List(S, s -> Position(elements, s))), "\n");
                Print("Adjacency Matrix A:\n");
                Display(A);
                Print("\n==================================================\n\n");
                foundGraph := true;
                # return; # Uncomment to stop after the first find.
            fi;
        od;
    od;

    if not foundGraph then
        Print("\n--- Search Complete ---\n");
        Print("No Cayley graph for any nonabelian group of order ", n, " was found to be a\n");
        Print("DSRG with parameters (n=", n, ", k=", k, ", t=", t, ", lambda=", lambda, ", mu=", mu, ").\n");
    fi;
end;

# Run the search.
CheckCayleyGraphsForDSRG();

# Quit GAP cleanly.
QUIT;
