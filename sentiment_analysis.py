# Sentiment analysis loading all previously saved classifiers

from nltk.tokenize import word_tokenize
import random
import pickle # Pickle allows to save Python Objects to recall in the future

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

with open('short_reviews/word_features5k.pickle','rb') as loading_pickle:
    word_features = pickle.load(loading_pickle)

def find_features(document):
    words = set(word_tokenize(document)) # parses all words in the specific review, skipping duplicates
    features = {}
    for w in word_features:
        features[w] = (w in words) # checks if the words in the specific review is in the Top 5000 most common words in the dataset. Returns True or False
    return features

### Loading classifiers

# Original Naive Bayes
with open('short_reviews/OriginalNaiveBayes_5k.pickle','rb') as loading_classifier:
    classifier = pickle.load(loading_classifier)

# classifier.show_most_informative_features(15)

# Multinomial NB
with open('short_reviews/MultinomialNaiveBayes_5k.pickle','rb') as loading_classifier:
    MNB_classifier = pickle.load(loading_classifier)

# Bernoulli NB
with open('short_reviews/BernoulliNaiveBayes_5k.pickle','rb') as loading_classifier:
    Bernoulli_classifier = pickle.load(loading_classifier)

# Logistic Regression
with open('short_reviews/LogisticRegression_5k.pickle','rb') as loading_classifier:
    Logistic_classifier = pickle.load(loading_classifier)

# SGD classifier - Stochastic Gradient Descent
with open('short_reviews/SGD_5k.pickle','rb') as loading_classifier:
    SGD_classifier = pickle.load(loading_classifier)

# Linear SVC
with open('short_reviews/LinearSVC_5k.pickle','rb') as loading_classifier:
    LinearSVC_classifier = pickle.load(loading_classifier)

# NU SVC
with open('short_reviews/NuSVC_5k.pickle','rb') as loading_classifier:
    NuSVC_classifier = pickle.load(loading_classifier)
#########################

voted_classifier = VoteClassifier(classifier, MNB_classifier, Bernoulli_classifier, Logistic_classifier, SGD_classifier, LinearSVC_classifier, NuSVC_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

print(sentiment('This was an amazing movie! I loved it so much. Great acting.'))

print(sentiment('This sucks. I hate the traffic in Sugar Land. Everyday is an awful standstill.'))