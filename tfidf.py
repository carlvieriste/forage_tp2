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
    print("Step 2: Compute gross metrics")

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

sqNormTFIDF = {}
termFrequency = {}

# Compute TF-IDF
numDocs = len(nbWordsInDoc)
for idMot in bruteTermFrequency.keys():
    sqNormTFIDF[idMot] = 0
    termFrequency[idMot] = {}
    for idDoc in bruteTermFrequency[idMot]:
        tf = bruteTermFrequency[idMot][idDoc] / nbWordsInDoc[idDoc]
        termFrequency[idMot][idDoc] = tf
        idf = math.log(numDocs / nbDocsContainingWord[idMot])
        tfidf = tf * idf
        sqNormTFIDF[idMot] += tfidf ** 2

print("Saving Term Frequency...")

with open("tfidf/tf.bin", "wb") as file:
    pickle.dump(termFrequency, file, pickle.HIGHEST_PROTOCOL)

print()
print("Step 4: Find keywords")

sortedSumTFIDF = sorted(sqNormTFIDF.items(), key=operator.itemgetter(1), reverse=True)
keywords = [x[0] for x in sortedSumTFIDF[0:6160]]

print("Saving keywords...")

with open("tfidf/keywords.bin", "wb") as file:
    pickle.dump(keywords, file, pickle.HIGHEST_PROTOCOL)

print()
print("Step 5: Output results (6160)")

with open("tfidf/out.txt", "w") as file:
    file.write("{:>12} {}\n".format("Norm TF-IDF", "Term"))
    for i in range(0, 6160):
        xSum = sortedSumTFIDF[i]
        file.write("{:>12.3f} {}\n".format(xSum[1], stems[xSum[0]]))