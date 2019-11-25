from gensim.models import FastText
import nltk
import re


base = "/Volumes/新加卷/nlp/"

words = {}

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def word_tokenize_dima(x):
	return filter(lambda w: not (hasNumbers(w)), nltk.word_tokenize(x))


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

sorted_x = sorted(words.items(), key= lambda kv: kv[1], reverse = True)

print(sorted_x[10])

top_100 = map(lambda x: x[0], sorted_x[:1000]) #list(filter(lambda k, v: v > sorted_x[10], words))

print(top_100)

with open(base + "vocab/top100.txt", 'w+') as out:
	out.write(" ".join(top_100))


