# combo-dl/src/ilp/cayley_graphs.g
#
# This script searches for a directed strongly regular graph (DSRG) with parameters
# (n=24, k=10, t=5, lambda=3, mu=5) among the Cayley graphs of all groups of order 24.
#
# A directed graph with adjacency matrix A is a DSRG with these parameters if it satisfies:
# 1. It is a 10-regular directed graph on 24 vertices.
# 2. The matrix equation A^2 = t*I + lambda*A + mu*(J - I - A) holds,
#    where I is the identity matrix and J is the all-ones matrix.
#
# For the given parameters, this simplifies to:
# A^2 = 5*I + 3*A + 5*(J - I - A)
# A^2 = 5*I + 3*A + 5*J - 5*I - 5*A
# A^2 = 5*J - 2*A
#
# To run this script, execute it from your terminal using the GAP interpreter.
# Make sure to provide the full path to the script, for example:
# > gap /path/to/combo-dl/src/ilp/cayley_graphs.g

CheckCayleyGraphsForDSRG := function()
    local n, k, groups, numGroups, J, foundGraph, i, G, groupId, elements, id, nonIdElements, S, numSets, setCounter, A, g, s, h, row, col, ASquared, RHS;

    n := 24;
    k := 10;
    foundGraph := false;

    # Load the small groups library if it's not already loaded.
    if not IsBound(SmallGroupsInformation) then
        RequirePackage("smallgrp");
    fi;

    groups := AllSmallGroups(n);
    numGroups := Size(groups);
    Print("Found ", numGroups, " groups of order ", n, ".\n");

    # Pre-calculate the all-ones matrix J.
    J := List([1..n], i -> List([1..n], j -> 1));

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

            # Check the DSRG condition.
            ASquared := A * A;
            RHS := (5 * J) - (2 * A);

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
        Print("No Cayley graph for any group of order 24 was found to be a\n");
        Print("DSRG with parameters (n=24, k=10, t=5, lambda=3, mu=5).\n");
    fi;
end;

# Run the search.
CheckCayleyGraphsForDSRG();

# Quit GAP cleanly.
QUIT;
