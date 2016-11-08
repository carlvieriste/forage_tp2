import operator
import pickle
import os.path
import math

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

if os.path.exists("tfidf/bruteTermFreq.lamereamax") \
        and os.path.exists("tfidf/inverseDocFreq.lamereamax") \
        and os.path.exists("tfidf/nbWordsInDoc.lamereamax"):

    print()
    print("Step 2: TF and IDF (load data)")

    with open("tfidf/bruteTermFreq.lamereamax", "rb") as file:
        bruteTermFrequency = pickle.load(file)

    with open("tfidf/inverseDocFreq.lamereamax", "rb") as file:
        nbDocsContainingWord = pickle.load(file)

    with open("tfidf/nbWordsInDoc.lamereamax", "rb") as file:
        nbWordsInDoc = pickle.load(file)

else:
    print()
    print("Step 2: Compute TF and IDF")

    bruteTermFrequency = {}
    nbWordsInDoc = {}

    # Number of documents containing the word
    nbDocsContainingWord = {}  # Word -> Nb docs

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

                # Brute term freq
                if idMot in bruteTermFrequency:
                    if idDoc in bruteTermFrequency[idMot]:
                        bruteTermFrequency[idMot][idDoc] += freqMot
                    else:
                        bruteTermFrequency[idMot][idDoc] = freqMot
                else:
                    bruteTermFrequency[idMot] = {}
                    bruteTermFrequency[idMot][idDoc] = freqMot

                # Num docs containing word
                if idMot in nbDocsContainingWord:
                    nbDocsContainingWord[idMot] += 1
                else:
                    nbDocsContainingWord[idMot] = 1

    print()
    print("Saving...")

    with open("tfidf/bruteTermFreq.lamereamax", "wb") as file:
        pickle.dump(bruteTermFrequency, file, pickle.HIGHEST_PROTOCOL)

    with open("tfidf/inverseDocFreq.lamereamax", "wb") as file:
        pickle.dump(nbDocsContainingWord, file, pickle.HIGHEST_PROTOCOL)

    with open("tfidf/nbWordsInDoc.lamereamax", "wb") as file:
        pickle.dump(nbWordsInDoc, file, pickle.HIGHEST_PROTOCOL)

print()
print("Step 3: Compute TF-IDF metrics")

sumTFIDF = {}
maxTFIDF = {}

# Compute TF-IDF
numDocs = len(nbWordsInDoc)
for idMot in bruteTermFrequency.keys():
    maxTFIDF[idMot] = 0
    sumTFIDF[idMot] = 0
    for idDoc in bruteTermFrequency[idMot]:
        tf = bruteTermFrequency[idMot][idDoc] / nbWordsInDoc[idDoc]
        idf = math.log(numDocs / nbDocsContainingWord[idMot])
        tfidf = tf * idf
        maxTFIDF[idMot] = max(maxTFIDF[idMot], tfidf)
        sumTFIDF[idMot] += tfidf ** 2

print()
print("Step 4: Output results (6160)")

with open("tfidf/out.txt", "w") as file:
    with open("tfidf/keywords.txt", "w") as keywordsFile:
        sortedMaxTFIDF = sorted(maxTFIDF.items(), key=operator.itemgetter(1), reverse=True)
        sortedSumTFIDF = sorted(sumTFIDF.items(), key=operator.itemgetter(1), reverse=True)

        file.write("{:>10} {:<20} {:>10} {}\n".format("Max TF-IDF", "Term", "Sum TF-IDF", "Term"))
        for i in range(0, 6160):
            xMax = sortedMaxTFIDF[i]
            xSum = sortedSumTFIDF[i]
            file.write("{:>10.3f} {:<20} {:>10.3f} {}\n".format(xMax[1], stems[xMax[0]], xSum[1], stems[xSum[0]]))
            keywordsFile.write("{}\n".format(xSum[0]))
