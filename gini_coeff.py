import pickle

import numpy as np
from joblib import Parallel, delayed

with open("../clustering/attempt_YW_0.bin", "rb") as file:
    Y, W = pickle.load(file)

N = 126900
K = 150
D = 6160

NUM_JOBS = 8

assert W.shape == (K, D)


def get_gini_coeff(c):
    w = W[c]
    sum_abs_diff = 0
    sum_elements = 0
    for i in w:
        sum_elements += i
        for j in w:
            sum_abs_diff += abs(i - j)
    return sum_abs_diff / (2.0 * D * sum_elements)


result = Parallel(n_jobs=NUM_JOBS)(delayed(get_gini_coeff)(c) for c in range(0, K))
gini = np.asarray(result)
np.savetxt("../gini.csv", gini, delimiter=',')
