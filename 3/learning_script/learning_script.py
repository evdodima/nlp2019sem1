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

vocab = {}

in_dir = args.in_dir
dict_path = args.dict_path
out_dir = args.out_dir

with open(dict_path,"rb") as lines:
	vocab = dict(pickle.load(lines))


for path in sorted(os.listdir(in_dir)):
	if path.endswith(".txt") and path != 'all.txt':
		full_path = in_dir + "/" + path + "/part-00000"
		with open(full_path, "r") as input_file:
			total_ex = 0
			total_words = 0
			for sentence in input_file:
				for word in sentence.split(" "):
					total_words += 1
				total_ex += 1
			print(path,total_ex,total_words)

			model_name = "fasttext_cbow_300_w10.model"
			model_path = out_dir + "/" + model_name
			model = None

			if os.path.exists(model_path):
				model = FastText.load(model_path)
			else:
				model = FastText(sg=0,size=300, window=10, min_n=3,max_n=5, min_count=1)  # instantiate

				model.build_vocab_from_freq(vocab)
				print("finished vocab")

			model.train(corpus_file = full_path, total_examples=total_ex, total_words=total_words, epochs=model.epochs)  # train

			model.save(get_tmpfile(model_path))


