e2:= function(r)

local coeff, length, i;

coeff:= CoefficientsAndMagmaElements(r);;
length:= Length(coeff);;


if length = 0 then
	return 0;
else
	return Sum(List([1..(length/2)], i -> coeff[2*i]^2));
fi;

end;

#################################################
#################################################

Diff:= function(D, emb, sumG, one, t, lambda, mu)

local imD;

imD:= Sum(List(D, i -> i^emb));;

return imD^2 - t*one - lambda*imD - mu*(sumG - imD - one);

end;

#################################################
#################################################

RandomSubset:= function(set, size)

local temp, g;

temp:= [];

while Size(temp) < size do
	g:= Random(set);
	if not g in temp then
		Add(temp, g);
	fi;
od;

return temp;

end;

#################################################
#################################################

HC:= function(G, preD, k, t, lambda, mu)

local test, R, emb, tempD, newD, i, best, trials, conj, g, h, besterror, bestneigh, one, sumG, error, compD;

conj:= List(ConjugacyClasses(G), AsSet);;

R:= GroupRing(Integers, G);;

emb:= Embedding(G, R);;

one:= Identity(G)^emb;;

sumG:= Sum(List(AsList(G), i -> i^emb));;

best:= e2(Diff(preD, emb, sumG, one, t, lambda, mu));;

if best = 0 then
	return [preD, 0];
else
	test:= false;
	tempD:= preD;
	trials:= 0;
fi;

while not test do
	besterror:= best;
	for g in tempD do
		compD:= Filtered(AsList(G), i -> not i in Union(tempD, [Identity(G)]));
		for h in compD do
			newD:= ShallowCopy(tempD);
			Remove(newD, Position(newD, g));
			Add(newD, h);
			error:= e2(Diff(newD, emb, sumG, one, t, lambda, mu));
			if error = 0 then
				test:= true;
				return [newD, error];
			fi;
			if error < besterror then
				besterror:= error;
				bestneigh:= newD;
			fi;
		od;
	od;
	if besterror < best then
		tempD:= bestneigh;
		best:= besterror;
	else
		test:= true;
	fi;
od;

return [tempD, best];

end;


#################################################
#################################################

RRHC:= function(G, k, t, lambda, mu)

local l, tempD, best, trials, notone;

notone:= Filtered(G, i -> Order(i) > 1);

l:= HC(G, RandomSubset(notone, k), k, t, lambda, mu);

tempD:= l[1];
best:= l[2];

trials:= 1;

while best > 0 do
	l:= HC(G, RandomSubset(notone, k), k, t, lambda, mu);
	if l[2] < best then
		tempD:= l[1];
		best:= l[2];
	fi;
	trials:= trials + 1;
	if trials mod 10 = 0 then
		Print(String(trials), "     ", String(best), "\n");
	fi;
od;

return tempD;

end;

#################################################
#################################################

RRHCkill:= function(G, k, t, lambda, mu, kill)

local l, tempD, best, trials, notone;

notone:= Filtered(G, i -> Order(i) > 1);

l:= HC(G, RandomSubset(notone, k), k, t, lambda, mu);

tempD:= l[1];
best:= l[2];

trials:= 1;

while (best > 0) and (trials < kill + 1) do
		l:= HC(G, RandomSubset(notone, k), k, t, lambda, mu);
		if l[2] < best then
			tempD:= l[1];
			best:= l[2];
		fi;
		trials:= trials + 1;
		if trials mod 10 = 0 then
			Print(String(trials), "     ", String(best), "\n");
		fi;
od;

if best = 0 then
	Print(tempD, "\n");
	return tempD;
else
	Print("None found", "\n");
	return [];
fi;

end;

#################################################
#################################################

RRHClist:= function(list, k, t, lambda, mu, kill)

local G, l, coll;

coll:= [];

for G in list do
	Print("***********************", "\n", IdGroup(G), "\n", "***********************", "\n");
	l:= RRHCkill(G, k, t, lambda, mu, kill);
	if not IsEmpty(l) then
		Add(coll, [G,l]);
	fi;
od;

return coll;

end;
