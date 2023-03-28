import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

for n in [100, 1000, 10000, 100000, 1000000]:
    for xp in (np, cp):
        x = xp.random.random(n).astype('f', copy=False)
        if xp is cp:
            x = cp.array(x)
        print(benchmark(xp.cumsum, (x,), n_repeat=10, name=f'cumsum-{n}-{xp.__name__}'))
