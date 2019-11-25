import fasttext
import os

modelTypes = ['skipgram', 'cbow']
dims = [300,600]
windows = [5,10]

base = "/Volumes/新加卷/nlp/"

for mt in modelTypes:
	for dim in dims:
		for window in windows:
			name = str((mt,dim,window))
			print("starting to train " + name)
			model = fasttext.train_unsupervised(base+'abstracts_tokenized/all.txt', model=mt, dim=dim, ws=window, minn=3, maxn=5)
			print("saving " + name)
			model.save_model(base + "models/"+ name)
			print("finished " + name)

