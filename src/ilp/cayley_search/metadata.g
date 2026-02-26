# cayley_search/metadata.g
#
# Quick metadata extraction: enumerate nonabelian groups of order n
# and output their SmallGroup library IDs for the Python orchestrator.
#
# Usage: pipe via stdin with "n := <value>;;" prepended, e.g.:
#   echo "n := 6;;" | cat - metadata.g | gap -q
#
# Output tokens (one per line):
#   GROUP_COUNT <count>
#   GROUP <filtered_index> <library_id> <StructureDescription>
#   ...
#   META_DONE

LoadPackage("smallgrp");;

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
