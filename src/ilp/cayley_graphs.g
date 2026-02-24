# combo-dl/src/ilp/cayley_graphs.g
#
# This script searches for a directed strongly regular graph (DSRG) with parameters
# (n=24, k=10, t=5, lambda=3, mu=5) among the Cayley graphs of a specific group of order 24.
#
# This version is designed for parallel execution. It accepts a command-line
# argument specifying which of the 15 groups of order 24 to check.
#
# The DSRG condition is A^2 = 5*J - 2*A.
#
# To run this script for a single group (e.g., the 3rd group):
# > gap cayley_graphs.g 3

# This function contains the logic to check a single group.
# It takes the index of the group (from 1 to 15) as an argument.
ProcessSingleGroup := function(groupIndex)
    local n, k, J, foundGraph, G, groupId, elements, id, nonIdElements, S, numSets, setCounter, A, g, s, h, row, col, ASquared, RHS, groups, numGroups;

    # --- 1. Initialization ---
    n := 24;
    k := 10;
    foundGraph := false;

    # Load the small groups library.
    if not IsBound(SmallGroupsInformation) then
        RequirePackage("smallgrp");
    fi;

    # Get the list of all groups of order 24.
    groups := AllSmallGroups(n);
    numGroups := Size(groups);

    # Validate the provided group index.
    if groupIndex < 1 or groupIndex > numGroups then
        Print("Error: Group index ", groupIndex, " is out of range. Must be between 1 and ", numGroups, ".\n");
        return;
    fi;

    # Select the single group to process based on the index.
    G := groups[groupIndex];
    groupId := IdGroup(G);
    Print("\n--- Checking Group Index ", groupIndex, "/", numGroups, ": Id=", groupId, " ---\n");

    # Pre-calculate the all-ones matrix J.
    J := List([1..n], i -> List([1..n], j -> 1));

    elements := AsList(G);
    id := Identity(G);
    nonIdElements := Filtered(elements, el -> el <> id);

    # --- 2. Iterate Through Connection Sets for the Single Group ---
    numSets := Binomial(n - 1, k);
    Print("Iterating through ", numSets, " possible connection sets...\n");
    setCounter := 0;

    for S in Combinations(nonIdElements, k) do
        setCounter := setCounter + 1;
        if setCounter mod 500000 = 0 and setCounter > 0 then
            Print("  ... [Group ", groupIndex, "] checked ", setCounter, "/", numSets, " sets\n");
        fi;

        # --- 3. Construct the Adjacency Matrix A ---
        A := List([1..n], i -> List([1..n], j -> 0));
        for row in [1..n] do
            g := elements[row];
            for s in S do
                h := g * s;
                col := Position(elements, h);
                A[row][col] := 1;
            od;
        od;

        # --- 4. Check the DSRG Condition ---
        ASquared := A * A;
        RHS := (5 * J) - (2 * A);

        if ASquared = RHS then
            Print("\n==================================================\n");
            Print("!!! FOUND A MATCHING CAYLEY GRAPH !!!\n");
            Print("Group Index: ", groupIndex, " (Id=", groupId, ")\n");
            Print("Group Structure: ", StructureDescription(G), "\n");
            Print("Connection Set S (elements are 1-based indices from AsList(G)):\n");
            Print(Sort(List(S, s -> Position(elements, s))), "\n");
            Print("Adjacency Matrix A:\n");
            Display(A);
            Print("\n==================================================\n\n");
            foundGraph := true;
            # return; # Uncomment to stop after the first find for this group.
        fi;
    od;

    # --- 5. Final Report for this Group ---
    if not foundGraph then
        Print("\n--- Search Complete for Group Index ", groupIndex, " (Id=", groupId, ") ---\n");
        Print("No matching Cayley graph found for this group.\n");
    fi;
end;


# --- Script Execution ---
# GAP stores command-line arguments in `GAPInfo.CommandLineOptions`.
# We check if an argument was provided.
if Size(GAPInfo.CommandLineOptions) < 1 then
    Print("Usage: gap cayley_graphs.g <group_index>\n");
    Print("Please provide a group index between 1 and 15.\n");
else
    # Convert the argument (which is a string) to an integer.
    groupIndex := Int(GAPInfo.CommandLineOptions[1]);
    ProcessSingleGroup(groupIndex);
fi;

# Quit GAP cleanly.
QUIT;
