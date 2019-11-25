from gensim.models import FastText
from gensim.test.utils import get_tmpfile

import pickle

modelTypes = ['skipgram', 'cbow']
dims = [300,600]
windows = [5,10]

base = "/Volumes/新加卷/nlp/"

vocab_dima = {}

with open(base + 'vocab/top100.pickle',"rb") as lines:
	vocab_dima = dict(pickle.load(lines))

print(vocab_dima)


with open(base + "abstracts_tokenized/pubmed19n0001.txt/part-00000", "r") as file_dima:
	tatal_ex = 0
	tatal_words = 0
	for sentence in file_dima:
		for word in sentence.split(" "):
			tatal_words += 1
		tatal_ex += 1
	print(tatal_ex,tatal_words)


	for mt in modelTypes:
		for dim in dims:
			for window in windows:
				name = str((mt,dim,window))
				print("starting to train " + name)

				model = FastText(sg=1 if mt == "skipgram" else 0,size=dim, window=window, min_n=3,max_n=5, min_count=1)  # instantiate
				model.build_vocab_from_freq(vocab_dima)
				print("finished vocab")

				model.train(corpus_file = base + "abstracts_tokenized/pubmed19n0001.txt/part-00000", total_examples=tatal_ex, total_words=tatal_words, epochs=model.epochs)  # train

				fname = get_tmpfile(base + "models/" + name + '.model')
				model.save(fname)

