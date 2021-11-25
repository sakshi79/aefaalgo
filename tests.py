from aefaalgo.aefa_optimize import aefa

N = 20
max_it = 50
FCheck = 1
Rpower = 1
tag = 0

func_num = 23
print(aefa().optimize(N, max_it, func_num))
