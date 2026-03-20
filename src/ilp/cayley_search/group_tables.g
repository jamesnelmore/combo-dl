# cayley_search/group_tables.g
#
# For a given group order n, output the Cayley (multiplication) table and
# inverse map for every nonabelian group.  Elements are 0-indexed for
# direct use as tensor indices in Python.
#
# Expects `n` to be defined before this file is loaded (injected by caller).
#
# Output format (one group per block):
#   GROUP_START <library_id> <name>
#   IDENTITY <0-indexed position of identity>
#   INV <space-separated 0-indexed inverse positions>
#   TABLE_START
#   <row 0: space-separated 0-indexed product positions>
#   ...
#   <row n-1>
#   TABLE_END
#   GROUP_END
#   ...
#   ALL_DONE

LoadPackage("smallgrp");;

groups := Filtered(AllSmallGroups(n), G -> not IsAbelian(G));;

for Gi in groups do
    gid      := IdGroup(Gi);;
    elements := AsList(Gi);;
    id_i     := Identity(Gi);;
    nn       := Size(elements);;

    # Build element -> 1-indexed position lookup.
    elemPos := NewDictionary(id_i, true);;
    for i in [1..nn] do
        AddDictionary(elemPos, elements[i], i);
    od;;

    idPos := LookupDictionary(elemPos, id_i);;

    Print("GROUP_START ", gid[2], " ", StructureDescription(Gi), "\n");
    Print("IDENTITY ", idPos - 1, "\n");

    # Inverse map.
    Print("INV");
    for i in [1..nn] do
        Print(" ", LookupDictionary(elemPos, elements[i]^-1) - 1);
    od;
    Print("\n");

    # Multiplication table.
    Print("TABLE_START\n");
    for i in [1..nn] do
        for j in [1..nn] do
            if j > 1 then Print(" "); fi;
            Print(LookupDictionary(elemPos, elements[i] * elements[j]) - 1);
        od;
        Print("\n");
    od;
    Print("TABLE_END\n");
    Print("GROUP_END\n");
od;

Print("ALL_DONE\n");
QUIT;
