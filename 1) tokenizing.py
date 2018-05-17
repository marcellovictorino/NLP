from nltk import sent_tokenize, word_tokenize

# tokenizing: separate specific groups (word tokenizer, sentence tokenizer)
# corpora: body of text. Material from the same area. Ex: medical journals, presidential speeches, english language

# lexicon: words and their meanings (could be different based on context)
# investor-speak ... regular english-speak

textSample = 'Testing some text recognition. Does it really work? Where does it go to eat? How can you avoid it? Aqui, no Globo Reporter!'

print(word_tokenize(textSample))