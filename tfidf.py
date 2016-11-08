import operator

print("Step 1: Load results of stemming")

stems = {}
indexReplacement = {}

with open('stemming/stems.txt') as file:
    for line in file.readlines():
        tokens = line.split(' ')
        idMot = int(tokens[0])
        mot = tokens[1].strip()
        stems[idMot] = mot

with open('stemming/indices.txt') as file:
    for line in file.readlines():
        tokens = line.split(' ')
        oldId = int(tokens[0])
        newId = int(tokens[1])
        indexReplacement[oldId] = newId

bruteTermFrequency = {}
nbWordsInDoc = {}

# Number of documents containing the word
inverseDocumentFrequency = {}  # Word -> Nb docs

print()

print("Step 2: Compute TF and IDF")

# Compute TF and IDF
for i in range(1, 4):
    print("File " + str(i))
    with open('docwords' + str(i) + '.txt') as file:
        for line in file.readlines():
            tokens = line.split(' ')
            idDoc = int(tokens[0])
            idMot = int(tokens[1])
            freqMot = int(tokens[2])

            if idMot in indexReplacement:
                idMot = indexReplacement[idMot]

            # Numbers of words per document
            if idDoc in nbWordsInDoc:
                nbWordsInDoc[idDoc] += freqMot
            else:
                nbWordsInDoc[idDoc] = freqMot

            # TF
            if idMot in bruteTermFrequency:
                if idDoc in bruteTermFrequency[idMot]:
                    bruteTermFrequency[idMot][idDoc] += freqMot
                else:
                    bruteTermFrequency[idMot][idDoc] = freqMot
            else:
                bruteTermFrequency[idMot] = {}
                bruteTermFrequency[idMot][idDoc] = freqMot

            # IDF
            if idMot in inverseDocumentFrequency:
                inverseDocumentFrequency[idMot] += 1
            else:
                inverseDocumentFrequency[idMot] = 1

print("Step 3: Compute Max TF-IDF")

sumTFIDF = {}
maxTFIDF = {}

# Compute TF-IDF
for idMot in bruteTermFrequency.keys():
    maxTFIDF[idMot] = 0
    sumTFIDF[idMot] = 0
    for idDoc in bruteTermFrequency[idMot]:
        tf = bruteTermFrequency[idMot][idDoc] / nbWordsInDoc[idDoc]
        tfidf = tf * inverseDocumentFrequency[idMot]
        maxTFIDF[idMot] = max(maxTFIDF[idMot], tfidf)
        sumTFIDF[idMot] += tfidf

# Print most important words
sortedMaxTFIDF = sorted(maxTFIDF.items(), key=operator.itemgetter(1), reverse=True)
sortedSumTFIDF = sorted(sumTFIDF.items(), key=operator.itemgetter(1), reverse=True)
for i in range(0, 6160):
    xMax = sortedMaxTFIDF[i]
    xSum = sortedSumTFIDF[i]
    print("{:>10.0f} {:<20} {:>10.0f} {}".format(xMax[1], stems[xMax[0]], xSum[1], stems[xSum[0]]))
