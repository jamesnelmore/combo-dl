# cayley_search/metadata.g
#
# Quick metadata extraction: enumerate nonabelian groups of order n
# and output their SmallGroup library IDs for the Python orchestrator.
#
# Usage:  gap -q metadata.g <n>
#
# Output tokens (one per line):
#   GROUP_COUNT <count>
#   GROUP <filtered_index> <library_id> <StructureDescription>
#   ...
#   META_DONE

LoadPackage("smallgrp");;

ARGS := Filtered(GAPInfo.SystemCommandLine,
    s -> Length(s) > 0 and ForAll(s, c -> c in "0123456789"));;

if Length(ARGS) < 1 then
    Print("ERROR missing argument: n\n");
    QUIT;
fi;

n := Int(ARGS[1]);;

groups := Filtered(AllSmallGroups(n), G -> not IsAbelian(G));;
numGroups := Size(groups);;

Print("GROUP_COUNT ", numGroups, "\n");

for i in [1..numGroups] do
    G   := groups[i];
    gid := IdGroup(G);
    # gid = [n, library_index]
    Print("GROUP ", i, " ", gid[2], " ", StructureDescription(G), "\n");
od;

Print("META_DONE\n");
QUIT;
