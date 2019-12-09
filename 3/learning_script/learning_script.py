from argparse import ArgumentParser
from gensim.models import FastText
from gensim.test.utils import get_tmpfile

import pickle
import logging

import os 

parser = ArgumentParser()
parser.add_argument("-d", dest="dict_path")
parser.add_argument("-out_dir", dest="out_dir")
parser.add_argument("-in_dir", dest="in_dir")
args = parser.parse_args()

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


modelTypes = ['skipgram', 'cbow']
dims = [300,600]
windows = [5,10]

base = "/Volumes/新加卷/nlp/"

vocab_dima = {}

in_dir = args.in_dir
dict_path = args.dict_path
out_dir = args.out_dir

with open(dict_path,"rb") as lines:
	vocab_dima = dict(pickle.load(lines))
# print(vocab_dima)

for path_dima in sorted(os.listdir(in_dir)):
	if path_dima.endswith(".txt") and path_dima != 'all.txt':
		full_path_dima = in_dir + "/" + path_dima
		with open(full_path_dima, "r") as file_dima:
			tatal_ex = 0
			tatal_words = 0
			for sentence in file_dima:
				for word in sentence.split(" "):
					tatal_words += 1
				tatal_ex += 1
			print(path_dima,tatal_ex,tatal_words)

			for mt in modelTypes:
				for dim in dims:
					for window in windows:
						name = str((mt,dim,window))

						print("started " + name + " on " + path_dima)

						fname = out_dir + "/" + name + '.model'

						model = None
						if os.path.exists(fname):
							model = FastText.load(fname)
						else:
							model = FastText(sg=1 if mt == "skipgram" else 0,size=dim, window=window, min_n=3,max_n=5, min_count=1)  # instantiate

							model.build_vocab_from_freq(vocab_dima)
							print("finished vocab")


						model.train(corpus_file = full_path_dima, total_examples=tatal_ex, total_words=tatal_words, epochs=model.epochs)  # train

						model.save(get_tmpfile(fname))


