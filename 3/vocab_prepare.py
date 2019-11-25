from gensim.models import FastText
import nltk
import re
import string

import pickle

base = "/Volumes/新加卷/nlp/"

words = {}

def invalid_word(inputString):
    return bool(re.search(r'\d', inputString)) or not re.search('[a-zA-Z]', inputString)

def word_tokenize_dima(x):
	return filter(lambda w: not (invalid_word(w)), nltk.word_tokenize(x))


path = base + "abstracts_tokenized/pubmed19n0001.txt/part-00000"
lines = sum(1 for line in open(path))

with open(path) as infile:
	print("opened")

	i = 0
	for line in infile:

		if i % 1000 == 0:
			print(i/float(lines))

		token_words = word_tokenize_dima(line)
		for word in token_words:
			if word not in words:
				words[word] = 0 

			words[word] += 1

		i += 1


print(len(words))

sorted_x = sorted(words.items(), key= lambda kv: kv[1], reverse = True)[:1000]


with open(base + "vocab/top100.pickle", 'wb') as out:
	pickle.dump(sorted_x, out, protocol = pickle.HIGHEST_PROTOCOL)



