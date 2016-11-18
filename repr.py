import pickle
import numpy as np


def pickle_big_object(o, filename):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(o, pickle.HIGHEST_PROTOCOL)
    with open(filename, "wb") as file:
        for idx in range(0, len(bytes_out), max_bytes):
            file.write(bytes_out[idx:idx + max_bytes])


print("Load stuff")

with open("tfidf/keywords.bin", "rb") as file:
    keywords = pickle.load(file)

with open("tfidf/tf.bin", "rb") as file:
    termFrequency = pickle.load(file)

# with open("docDics.bin", "rb") as file:
#    docDicks = pickle.load(file)

print("Make doc representations")

docDicks = {}

count = 0
for idTerm, term in termFrequency.items():
    for idDoc, tf in term.items():
        if idTerm not in keywords:
            continue

        if idDoc in docDicks:
            docDicks[idDoc][idTerm] = tf
        else:
            docDicks[idDoc] = {idTerm: tf}

    count += 1
    if count % 1000 == 0:
        print(count)

try:
    pickle_big_object(docDicks, "docDics.bin")
except OverflowError as e:
    print(e)
    pass

print("Convert repr to vectors")

docVectorsMat = np.zeros((129000, 6160), dtype=np.float32)
count = 0
for idDoc, docDic in docDicks.items():
    for idKw, idTerm in enumerate(keywords):
        if idTerm in docDic:
            docVectorsMat[idDoc, idKw] = docDic[idTerm]

    count += 1
    if count % 1000 == 0:
        print(count)

print("Output")

pickle_big_object(docVectorsMat, "repr.bin")
