# from gensim.models import FastText
import nltk
import re
import string
from argparse import ArgumentParser
import pickle, os


parser = ArgumentParser()
parser.add_argument("-out_dir", dest="out_dir")
parser.add_argument("-in_dir", dest="in_dir")

args = parser.parse_args()

input_path = args.in_dir
output_path = args.out_dir

words = {}

def invalid_word(inputString):
    return bool(re.search(r'\d', inputString)) or not re.search('[a-zA-Z]', inputString)

def word_tokenize(x):
	return filter(lambda w: not (invalid_word(w)), nltk.word_tokenize(x))

for path in sorted(os.listdir(input_path)):
	if path.endswith(".txt") and path != 'all.txt':
		print(path)
		with open(input_path + "/" + path + "/part-00000") as infile:
			for line in infile:
				token_words = word_tokenize(line)
				for word in token_words:
					if word not in words:
						words[word] = 0 
					words[word] += 1

sorted_x = sorted(words.items(), key= lambda kv: kv[1], reverse = True)[:100000]

with open(output_path + "/result.pickle", 'wb') as out:
	pickle.dump(sorted_x, out, protocol = pickle.HIGHEST_PROTOCOL)



