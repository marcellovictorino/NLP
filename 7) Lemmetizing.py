from nltk.stem import WordNetLemmatizer

# Similar to stemming, but the "simplified" text is an actual word.
# More useful since it helps keep the meaning of sentences

lemmatizer = WordNetLemmatizer()

# (word, desired value)
# the second parameter can vary between 'n', 'a', 'v' (Noun, adjective, verb)
print(lemmatizer.lemmatize("singing", 'v')) # returns sing

print(lemmatizer.lemmatize("better", pos='a')) # returns good

# Conclusion: overral more useful tham stemming, but requires correct definition of second paramater
