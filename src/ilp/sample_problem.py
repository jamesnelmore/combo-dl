from pulp import *

lp = LpProblem('sample_problem', LpMinimize)

x1 = LpVariable(name='percentChicken', lowBound=0, upBound=None, cat=LpInteger)
x2 = LpVariable(name='percentBeef', lowBound=0, upBound=None)

lp += .013 * x1 + .008 * x2, "Objective function"
lp += x1+x2 == 100
lp += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
lp += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
lp += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
lp += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

# lp.writeLP('test.lp')
lp.solve()
