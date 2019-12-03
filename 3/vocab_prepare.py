# from gensim.models import FastText
import nltk
import re
import string

import pickle


parser = ArgumentParser()
parser.add_argument("-out_dir", dest="out_dir")
parser.add_argument("-in_dir", dest="in_dir")

args = parser.parse_args()

input_path = parser.in_dir
output_path = parser.out_dir

words = {}

def invalid_word(inputString):
    return bool(re.search(r'\d', inputString)) or not re.search('[a-zA-Z]', inputString)

def word_tokenize(x):
	return filter(lambda w: not (invalid_word(w)), nltk.word_tokenize(x))


# path = base + "abstracts_tokenized/all.txt"
lines = sum(1 for line in open(path))

for path in sorted(os.listdir(input_path)):
	if path.endswith(".txt"):
		with open(path) as infile:
			print("opened " + path)
			i = 0
			for line in infile:

				if i % 10000 == 0:
					# print(i/float(lines))
				token_words = word_tokenize(line)
				for word in token_words:
					if word not in words:
						words[word] = 0 

					words[word] += 1

				i += 1

print(len(words))

sorted_x = sorted(words.items(), key= lambda kv: kv[1], reverse = True)[:100000]

with open(output_path, 'wb') as out:
	pickle.dump(sorted_x, out, protocol = pickle.HIGHEST_PROTOCOL)



