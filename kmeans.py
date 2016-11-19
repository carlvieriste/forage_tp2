import os.path
import pickle
import numpy as np
import time
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed


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
K = 50  # K clusters
# W = np.ones((K, D))
# Y is the labels, with shape: (N, 1) - defined later
NUM_JOBS = 8
CHUNK_SIZE = int(np.ceil(N / NUM_JOBS))

for j, num_clusters in enumerate([150, 150, 200, 200, 200]):
    K = num_clusters

    # 1. Set centers to random samples
    means = docs[np.random.choice(N, K, replace=False), :]
    total_err = 0

    for i in range(0, 50):
        print("Iteration", i)
        print("Updating labels...")
        time0 = time.time()

        # 2. Update labels
        def get_updated_labels(offset):
            dist_matrix = cdist(docs[offset:offset + CHUNK_SIZE], means,
                                metric='sqeuclidean')  # metric='wminkowski', w=W, p=2
            i_min = np.argmin(dist_matrix, axis=1)
            err = dist_matrix[np.arange(len(dist_matrix)), i_min]
            return np.vstack((i_min, err))  # (2, n)


        result = Parallel(n_jobs=NUM_JOBS)(
            delayed(get_updated_labels)(i) for i in range(0, N, CHUNK_SIZE))
        matrix_result = np.concatenate(result, axis=1)  # (2, N) => lignes : label, err
        Y = matrix_result[0, :]
        total_err = np.sum(matrix_result[1, :])
        print("Err:", total_err)

        print("Done. Took", time.time() - time0)
        print("Updating means...")
        time0 = time.time()

        # 3. Update means
        def calc_mean(c):
            # noinspection PyTypeChecker
            docs_in_cluster = docs[np.where(Y == c)]
            return np.average(docs_in_cluster, axis=0)


        result = Parallel(n_jobs=NUM_JOBS)(delayed(calc_mean)(c) for c in range(0, K))
        means = np.asarray(result)

        print("Done. Took", time.time() - time0)

    # Completed this attempt
    with open("clustering/attempt" + str(j) + ".bin", "wb") as file:
        pickle.dump(Y, file, pickle.HIGHEST_PROTOCOL)

    with open("clustering/results.txt", "a") as file:
        file.write("Attempt " + str(j) + " ==== " + time.ctime() + "\n")
        file.write("Num. clusters: " + str(K) + "\n")
        file.write("Total dist from centers: " + str(total_err) + "\n")
        max_num = 0
        for c in range(0, K):
            # noinspection PyTypeChecker
            file.write("{:<4} {:<6} {:<.3f}%\n".format(c, np.sum(Y == c), np.sum(Y == c) / N))
            max_num = max(max_num, np.sum(means[c]))
        file.write("L1 norm of mean of largest cluster: " + str(max_num) + "\n")
        file.write("\n\n")
