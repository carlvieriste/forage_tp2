import os.path
import pickle
import numpy as np
import time
from scipy.spatial.distance import cdist


def unpickle_big_object(file_path):
    max_bytes = 2 ** 31 - 1

    print("Reading", file_path)

    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

    return pickle.loads(bytes_in)


docs = unpickle_big_object("repr.bin")
assert docs.shape == (129000, 6160)

N, D = docs.shape  # N documents, D dimensions
K = 10  # K clusters
W = np.ones((K, D))
Y = np.zeros(N, dtype=np.int16)

# 1. Set centers to random samples
means = docs[np.random.choice(N, K, replace=False), :]

for i in [1]:  # TODO change this
    print("Updating labels...")
    time0 = time.time()

    # 2. Update labels
    dist_matrix = cdist(docs, means, metric='wminkowski', w=W, p=2)
    Y = np.argmin(dist_matrix, axis=1)

    print("Done. Took", time.time() - time0)
    print("Updating means...")
    time0 = time.time()

    # 3. Update means
    for c in range(0, K):
        # noinspection PyTypeChecker
        means[c] = np.average(docs, axis=0, weights=(Y == c))

    print("Done. Took", time.time() - time0)
