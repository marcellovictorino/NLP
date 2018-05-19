from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, WordPunctTokenizer

# Wordnet is a lexical database for english language, developed by Princeton Univ.
# It provides the meanings of words, synonyms, antonyms, and more.

synonyms = wordnet.synsets('good')
# All synsets
print(synonyms)

# Splitting to get just the word - from the first identified element
print(synonyms[0].lemmas()[0].name())

# Definition
print(synonyms[0].definition())

# Examples in context
print(synonyms[0].examples())

synonyms =[]
antonyms =[]
for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
    if l.antonyms():
        antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)

# Compairing Semantic similarities

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

print('Similarity: {:.1f} %'.format(w1.wup_similarity(w2)*100))