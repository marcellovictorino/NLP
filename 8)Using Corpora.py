from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# %appdata% / nltl_data / copora
# there you can find all available text material, like movie_review (po and neg)
# also wordnet, which will be used in the actual project

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

print(tok[5:20])