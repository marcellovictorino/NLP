# Develop a text classifier to perform Sentiment ANalysis:
# Identify Good or Bad sentiment. Works for any category, as long as binary outcome (good or bad)
# features will be used to train. Label will be the output/result

import nltk
from nltk.tokenize import word_tokenize
import random
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


short_pos = open('short_reviews/positive.txt', 'r').read()
short_neg = open('short_reviews/negative.txt', 'r').read()

documents = []
all_words = []

# Tags: J = adjective | R = adverb | V = verb
allowed_words = ['J', 'R', 'V']

for review in short_pos.split('\n'):
    documents.append( (review, 'pos') )
    words = word_tokenize(review)
    part_of_speech = nltk.pos_tag(words)
    for word in part_of_speech:
        if word[1][0] in allowed_words: # looking for Tag, first letter
            all_words.append(word[0].lower()) # only appends desired words

for review in short_neg.split('\n'):
    documents.append( (review, 'neg') )
    words = word_tokenize(review)
    part_of_speech = nltk.pos_tag(words)
    for word in part_of_speech:
        if word[1][0] in allowed_words: # looking for Tag, first letter
            all_words.append(word[0].lower()) # only appends desired words

with open('short_reviews/documents.pickle','wb') as saving_pickle:
    pickle.dump(documents, saving_pickle)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000] # top 5000 most commons words used on short revies dataset

with open('short_reviews/word_features5k.pickle','wb') as saving_pickle:
    pickle.dump(word_features, saving_pickle)


def find_features(document):
    words = set(word_tokenize(document)) # parses all words in the specific review, skipping duplicates
    features = {}
    for w in word_features:
        features[w] = (w in words) # checks if the words in the specific review is in the Top 5000 most common words in the dataset. Returns True or False
    return features


featuresets = []
for rev, category in documents:
    featuresets.append( (find_features(rev), category) )

with open('short_reviews/featureset_5k.pickle','wb') as saving_pickle:
    pickle.dump(featuresets, saving_pickle)

random.shuffle(featuresets)

###################
trainingRate = 0.85 # to be allocated for training purpose
delimiter = int(len(featuresets) * trainingRate)

training_set = featuresets[:delimiter]
testing_set = featuresets[delimiter:]

############################################################
### Verifying Bias in the Model: Negative vs. Positive
#
## Negative Only = Avg x% accuracy
# training_set = featuresets[:900] + featuresets[1000:1900]
# testing_set = featuresets[900:1000]

## Positive Only = Avg x% accuracy
# training_set = featuresets[:900] + featuresets[1000:1900]
# testing_set = featuresets[1900:]

#>> Conclusion: the model is biased towards Negative (since it tends to be more accurate)
#>>             This could be explained due the fact that Negative reviews tend to be "more negative" than the Positive reviews are positive.
#############################################################

classifier = nltk.NaiveBayesClassifier.train(training_set)
## Saving the classifier to recall in the future without having to retrain
with open('short_reviews/OriginalNaiveBayesV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(classifier, saving_classifier)

# ### Loading a saved Classifier
# with open('naiveBayes.pickle','rb') as classifier_load:
#     classifier = pickle.load(classifier_load)

print('Original Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(classifier, testing_set) *100))
# classifier.show_most_informative_features(15)

##########
## Using sklearn algorithm with NLTK toolkit

# Multinomial NB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('Multinomial Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(MNB_classifier, testing_set) *100))
with open('short_reviews/MultinomialNaiveBayesV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(MNB_classifier, saving_classifier)


# Bernoulli NB
Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print('Bernoulli Naive Bayes accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(Bernoulli_classifier, testing_set) *100))
with open('short_reviews/BernoulliNaiveBayesV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(Bernoulli_classifier, saving_classifier)

# Logistic Regression
Logistic_classifier = SklearnClassifier(LogisticRegression())
Logistic_classifier.train(training_set)
print('Logistic Regression accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(Logistic_classifier, testing_set) *100))
with open('short_reviews/LogisticRegressionV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(Logistic_classifier, saving_classifier)

# SGD classifier - Stochastic Gradient Descent
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print('SGD accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(SGD_classifier, testing_set) *100))
with open('short_reviews/SGDV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(SGD_classifier, saving_classifier)

# Linear SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(LinearSVC_classifier, testing_set) *100))
with open('short_reviews/LinearSVCV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(LinearSVC_classifier, saving_classifier)

# NU SVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('Nu SVC accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(NuSVC_classifier, testing_set) *100))
with open('short_reviews/NuSVCV2_5k.pickle','wb') as saving_classifier:
    pickle.dump(NuSVC_classifier, saving_classifier)
#########################

voted_classifier = VoteClassifier(classifier, MNB_classifier, Bernoulli_classifier, Logistic_classifier, SGD_classifier, LinearSVC_classifier, NuSVC_classifier)
print('Voted_classifier accuracy: {:.1f} %\n'.format(nltk.classify.accuracy(voted_classifier, testing_set) *100))

print('Classification: {}  |  Confidence: {:.1f} %'.format(voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0])*100 ))
