# Develop a text classifier to perform Sentiment ANalysis:
# Identify Good or Bad sentiment. Works for any category, as long as binary outcome (good or bad)
# features will be used to train. Label will be the output/result

import nltk
import random
from nltk.corpus import movie_reviews, opinion_lexicon
import pickle # Pickle allows to save Python Objects to recall in the future

from nltk.classify.scikitlearn import SklearnClassifier # Wrapper to include Sklearn algorithm to NLTK
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

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
## The top 15 are useless words such as "a", "the", ",", "." ...
# print(all_words.most_common(15))

word_features = list(all_words.keys())[:3000] # top 3000 most commons words used on Movie Reviews

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

# classifier = nltk.NaiveBayesClassifier.train(training_set)

### Saving the classifier to recall in the future without having to retrain
# with open('naiveBayes.pickle','wb') as save_classifier:
#     pickle.dump(classifier, save_classifier)

### Loading a saved Classifier
with open('naiveBayes.pickle','rb') as classifier_load:
    classifier = pickle.load(classifier_load)

print('Original Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(classifier, testing_set) *100))
# classifier.show_most_informative_features(15)

##########
## Using sklearn algorithm with NLTK toolkit

# Multinomial NB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('Multinomial Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(MNB_classifier, testing_set) *100))

# Bernoulli NB
Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print('Bernoulli Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(Bernoulli_classifier, testing_set) *100))

# Logistic Regression
Logistic_classifier = SklearnClassifier(LogisticRegression())
Logistic_classifier.train(training_set)
print('Logistic Regression accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(Logistic_classifier, testing_set) *100))

# SGD classifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print('SGD accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(SGD_classifier, testing_set) *100))

# Linear SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(LinearSVC_classifier, testing_set) *100))

# NU SVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('Nu SVC accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(NuSVC_classifier, testing_set) *100))
#########################


voted_classifier = VoteClassifier(classifier, MNB_classifier, Bernoulli_classifier, Logistic_classifier, SGD_classifier, LinearSVC_classifier, NuSVC_classifier)
print('Votes_classifier accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(voted_classifier, testing_set) *100))

print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0])*100 ))
print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[1][0]), voted_classifier.confidence(testing_set[1][0])*100 ))
print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[2][0]), voted_classifier.confidence(testing_set[2][0])*100 ))
print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[3][0]), voted_classifier.confidence(testing_set[3][0])*100 ))
print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[4][0]), voted_classifier.confidence(testing_set[4][0])*100 ))
