# Develop a text classifier to perform Sentiment ANalysis:
# Identify Good or Bad sentiment. Works for any category, as long as binary outcome (good or bad)
# features will be used to train. Label will be the output/result

import nltk
import random
from nltk.corpus import movie_reviews

documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append( (list(movie_reviews.words(fileid)), category) )

random.shuffle(documents)

# List of all words. Normalizing all to lower case
all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
## The top 15 are useless words such as a, the, commas, periods, the...
# print(all_words.most_common(15))

word_features = list(all_words.keys())[:3000] # top 3000 most commons words

def find_features(document):
    words = set(document) # parses all words in the list, skipping duplicates
    features = {}
    for w in word_features:
        features[w] = (w in words) # if the Top 3000 most common in the Document, True. Otherwise, False
    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = []
for rev, category in documents:
    featuresets.append( (find_features(rev), category) )

###################
trainingRate = 0.85 # to be allocated for training purpose
delimiter = int(len(featuresets) * trainingRate)

training_set = featuresets[:delimiter]
testing_set = featuresets[delimiter:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(classifier, testing_set) *100))

classifier.show_most_informative_features(15)