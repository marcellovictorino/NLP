from nltk import sent_tokenize, word_tokenize

# tokenizing: separate specific groups (word tokenizer, sentence tokenizer)
# corpora: body of text. Material from the same area. Ex: medical journals, presidential speeches, english language

# lexicon: words and their meanings (could be different based on context)
# investor-speak ... regular english-speak

textSample = 'Testing some text recognition. Does it really work? Where does it go to eat? How can you avoid it? Aqui, no Globo Reporter!'

# print(word_tokenize(textSample))

###########################
# 2) Stop words = words to avoid/fillers (an, the, and, or, between...)
from nltk.corpus import stopwords

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

# filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for word in word_tokens:
    if word not in stop_words:
        filtered_sentence.append(word)

print(word_tokens)
print(filtered_sentence)

######################
# 3)Stemming = words having same meaning (write, writing, writen)
from nltk.stem import PorterStemmer
ps = PorterStemmer

new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)
for word in words:
    print(ps.stem(word))
