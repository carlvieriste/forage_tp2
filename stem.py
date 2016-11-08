from collections import OrderedDict

from code.porterStemmer import PorterStemmer

file = open('words.txt', 'r', encoding="ISO-8859-1")

p = PorterStemmer()

word2index = OrderedDict()
indexReplacement = {}

nb = 0
totalWords = 0
for line in file.readlines():
    tokens = line.split(' ')
    index = int(tokens[0])
    word = tokens[1].strip()

    stemmed_word = p.stem(word, 0, len(word) - 1)
    if word != stemmed_word:
        if nb < 20:
            print(word + ' -> ' + stemmed_word)
        nb += 1
    if stemmed_word in word2index:
        existingIndex = word2index[stemmed_word]
        indexReplacement[index] = existingIndex
        # word2index[stemmed_word] = index
    else:
        word2index[stemmed_word] = index
    totalWords += 1

file.close()

print()
print("Stems:       " + str(len(word2index)))
print("Total words: " + str(totalWords))
absRed = totalWords - len(word2index)
print("Reduction:   {} ({:.2%})".format(absRed, absRed / totalWords))


with open('stemming/stems.txt', 'w') as file:
    for word, index in word2index.items():
        file.write(str(index) + ' ' + word + '\n')


with open('stemming/indices.txt', 'w') as file:
    for oldId, newId in indexReplacement.items():
        file.write(str(oldId) + ' ' + str(newId) + '\n')
