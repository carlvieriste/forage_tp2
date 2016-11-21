import pickle
import numpy as np


def gini(instances):
    assert len(instances.shape) == 1
    total = instances.size

    # Compute brute frequencies
    brute_freq = {}
    for instance in instances:
        brute_freq.setdefault(instance, 0)
        brute_freq[instance] += 1

    # Compute sum of squared fractions
    sum_fractions = 0
    for label in brute_freq:
        sum_fractions += (brute_freq[label] / total)**2

    return 1.0 - sum_fractions


N = 126900
# K = 100
# attempt = "W_K_" + str(K)

# docId -> NSF program
P = np.genfromtxt("../sweetSummerChild.txt", delimiter=" ")
P = P[0:N, :]

results = []
for K, attempt in [(50, "K_50"), (100, "K_100"), (150, "K_150"), (200, "K_200"), (50, "W_K_50"), (100, "W_K_100"), (150, "W_K_150"), (200, "W_K_200")]:
    with open("../clustering/best/best_" + attempt + ".bin", "rb") as file:
        Y = pickle.load(file)

    Y = Y[0:N]

    impurities = np.ones((200,))

    for c in range(0, K):
        print("label", c)
        cluster_indices = np.where(Y == c)
        program_instances_in_cluster = P[cluster_indices, 1]
        prog_list = program_instances_in_cluster.reshape((program_instances_in_cluster.size,))
        impurities[c] = gini(prog_list)
        print(impurities[c])

    results.append(impurities)
    #np.savetxt("../purity/purity_" + attempt + ".csv", impurities, delimiter=';')

results = np.vstack(results).transpose()
np.savetxt("../purity/purity.csv", results, delimiter=';')