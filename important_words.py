import os
import pickle
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist


# 1. Trier les clusters selon gini coefficient (ou impurity)
# 2. Sortir les 10 meilleurs
# 3. Calculer centroide * W
# 4. Trier pour avoir les 10 mots les plus significatifs par cluster


def unpickle_big_object(file_path):
    max_bytes = 2 ** 31 - 1

    print("Reading", file_path)

    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

    return pickle.loads(bytes_in)


# Load document representations
docs = unpickle_big_object("../repr.bin")
assert docs.shape == (129000, 6160)

docs = docs[0:126900, :]  # TODO on ignore environ 2000 documents a la fin

# Load word strings
words = ['ZERO']
with open("../words.txt", "r", encoding="ISO 8859-1") as file:
    for line in file.readlines():
        tokens = line.split(' ')
        words.append(tokens[1].strip())
print(len(words))

# Load keyword_id -> word_id relation
keywords = []
with open("../tfidf/keywords.txt", "r") as file:
    for line in file.readlines():
        keywords.append(int(line.strip()))
print(len(keywords))

# Load labels of a specific clustering attempt
with open("../clustering/attempt_YW_0.bin", "rb") as file:
    Y, W = pickle.load(file)

N = 126900
K = 150
D = 6160

NUM_JOBS = 8

assert Y.shape == (N,)
assert W.shape == (K, D)

cluster_sizes = np.zeros((K,))


# Compute means (and cluster sizes)
def calc_mean(c):
    # noinspection PyTypeChecker
    docs_in_cluster = docs[np.where(Y == c)]
    cluster_sizes[c] = docs_in_cluster.shape[0]
    return np.average(docs_in_cluster, axis=0)


print("Compute means")
result = Parallel(n_jobs=NUM_JOBS)(delayed(calc_mean)(c) for c in range(0, K))
means = np.asarray(result)

means_weighted = np.multiply(means, W)


def get_gini_coeff(c):
    print(c)
    m = means_weighted[c]
    sum_abs_diff = 0
    sum_elements = 0
    for i in m:
        sum_elements += i
        for j in m:
            sum_abs_diff += abs(i - j)
    return sum_abs_diff / (2.0 * D * sum_elements)


# print("Compute gini coeff")
# result = Parallel(n_jobs=NUM_JOBS)(delayed(get_gini_coeff)(c) for c in range(0, K))
# gini = np.asarray(result)
# np.savetxt("../means_gini_coeff.csv", gini, delimiter=',')
# gini = np.genfromtxt("../means_gini_coeff.csv", delimiter=",")

impurities = np.genfromtxt("../purity/purity_W_W_K_50.csv", delimiter=",")  # K=150

# Get the 100 purest clusters
best_clusters_idx = np.argsort(impurities)[0:100]
print(impurities[best_clusters_idx])
print(cluster_sizes[best_clusters_idx])

best_clusters_vectors = means_weighted[best_clusters_idx, :]

# Get the most significant words in each cluster
best_clusters_sorted = np.argsort(best_clusters_vectors, axis=1)[:, ::-1]

# Print 30 most significant words in the 100 purest clusters
for clus in best_clusters_sorted[:, 0:30]:
    significant_words = []
    for word in clus:
        word_id = keywords[word]
        word_str = words[word_id]
        significant_words.append(word_str)
    print(" ".join(significant_words))
