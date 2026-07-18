# ilp/models/cayley_dsrg.g
#
# For a given group SmallGroup(n, lib_id), output:
#   1. Element ordering (non-identity elements) with identity position
#   2. Inverse map for non-identity elements
#   3. Product table: for each pair (g, h) of non-identity elements,
#      the index of g^{-1}h (used for the DSRG 2-path constraint)
#   4. Classification: involutions vs non-self-inverse pairs
#   5. Generators of Aut(G) as permutations of non-identity element indices
#
# Expects `n` and `lib_id` to be defined before this file is loaded.
#
# Output format:
#   CAYLEY_START <n> <lib_id> <name>
#   IDENTITY <0-indexed position of identity in GAP element list>
#   NUM_NONID <n-1>
#   NONID_ORDER <space-separated 0-indexed positions of non-identity elements>
#   INV_MAP <space-separated: for each non-identity element index i, the
#            non-identity index of its inverse>
#   INVOLUTIONS <space-separated non-identity indices that are involutions>
#   PAIRS <space-separated: pairs (i, j) with i < j where elements are inverses>
#   PRODUCTS_START
#   <for each non-identity g (row), for each non-identity h (col):
#    the non-identity index of g^{-1}h, or -1 if g^{-1}h = identity>
#   PRODUCTS_END
#   AUT_GENERATORS <num_generators>
#   <for each generator: space-separated permutation of non-identity indices>
#   CAYLEY_END

LoadPackage("smallgrp");;

G := SmallGroup(n, lib_id);;
elements := AsList(G);;
nn := Size(elements);;
id_G := Identity(G);;

# Build element -> 1-indexed position lookup.
elemPos := NewDictionary(id_G, true);;
for i in [1..nn] do
    AddDictionary(elemPos, elements[i], i);
od;;

idPos := LookupDictionary(elemPos, id_G);;

# Build non-identity element list (0-indexed within the non-identity array).
nonIdElems := [];;
nonIdOrigIdx := [];;   # original 0-indexed positions in elements list
for i in [1..nn] do
    if i <> idPos then
        Add(nonIdElems, elements[i]);
        Add(nonIdOrigIdx, i - 1);  # 0-indexed
    fi;
od;;

numNonId := Size(nonIdElems);;

# Build non-identity element -> non-identity 0-index lookup.
nonIdPos := NewDictionary(id_G, true);;
for i in [1..numNonId] do
    AddDictionary(nonIdPos, nonIdElems[i], i - 1);  # 0-indexed
od;;

Print("CAYLEY_START ", n, " ", lib_id, " ", StructureDescription(G), "\n");
Print("IDENTITY ", idPos - 1, "\n");
Print("NUM_NONID ", numNonId, "\n");

# Non-identity element ordering (original 0-indexed positions).
Print("NONID_ORDER");
for i in [1..numNonId] do
    Print(" ", nonIdOrigIdx[i]);
od;
Print("\n");

# Inverse map within non-identity indices.
Print("INV_MAP");
for i in [1..numNonId] do
    inv_elem := nonIdElems[i]^-1;
    Print(" ", LookupDictionary(nonIdPos, inv_elem));
od;
Print("\n");

# Involutions (non-identity elements equal to their own inverse).
involutions := [];;
for i in [1..numNonId] do
    if nonIdElems[i]^-1 = nonIdElems[i] then
        Add(involutions, i - 1);  # 0-indexed
    fi;
od;;

Print("INVOLUTIONS");
for i in involutions do
    Print(" ", i);
od;
Print("\n");

# Non-self-inverse pairs (i < j).
pairs := [];;
seen := [];;
for i in [1..numNonId] do
    idx := i - 1;
    if not (idx in seen) then
        inv_idx := LookupDictionary(nonIdPos, nonIdElems[i]^-1);
        if inv_idx <> idx then
            if idx < inv_idx then
                Add(pairs, [idx, inv_idx]);
            else
                Add(pairs, [inv_idx, idx]);
            fi;
            Add(seen, idx);
            Add(seen, inv_idx);
        fi;
    fi;
od;;

Print("PAIRS");
for p in pairs do
    Print(" ", p[1], " ", p[2]);
od;
Print("\n");

# Product table: for each non-identity g (row i), non-identity h (col j),
# compute g^{-1}h. Output the non-identity index, or -1 if result is identity.
Print("PRODUCTS_START\n");
for i in [1..numNonId] do
    g_inv := nonIdElems[i]^-1;
    for j in [1..numNonId] do
        prod := g_inv * nonIdElems[j];
        if prod = id_G then
            if j > 1 then Print(" "); fi;
            Print(-1);
        else
            if j > 1 then Print(" "); fi;
            Print(LookupDictionary(nonIdPos, prod));
        fi;
    od;
    Print("\n");
od;
Print("PRODUCTS_END\n");

# Automorphism group generators as permutations of non-identity indices.
AutG := AutomorphismGroup(G);;
gens := GeneratorsOfGroup(AutG);;

Print("AUT_GENERATORS ", Size(gens), "\n");
for phi in gens do
    for i in [1..numNonId] do
        img := Image(phi, nonIdElems[i]);
        if i > 1 then Print(" "); fi;
        Print(LookupDictionary(nonIdPos, img));
    od;
    Print("\n");
od;

Print("CAYLEY_END\n");
QUIT;
