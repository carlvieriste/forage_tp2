import pickle

print("Load stuff")

with open("tfidf/keywords.bin", "rb") as file:
    keywords = pickle.load(file)

with open("tfidf/tf.bin", "rb") as file:
    termFrequency = pickle.load(file)

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

print("Convert repr to vectors")

docVectors = {}
count = 0
for idDoc, docDic in docDicks.items():
    docVectors[idDoc] = []
    for idTerm in keywords:
        if idTerm in docDic:
            docVectors[idDoc].append(docDic[idTerm])
        else:
            docVectors[idDoc].append(0.0)

    count += 1
    if count % 1000 == 0:
        print(count)

print("Output")

max_bytes = 2**31 - 1
bytes_out = pickle.dumps(docVectors)
with open("repr.bin", "wb") as file:
    for idx in range(0, len(bytes_out), max_bytes):
        file.write(bytes_out[idx:idx+max_bytes])
