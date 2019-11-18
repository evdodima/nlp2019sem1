import fasttext


path = './result.txt/part-00000'
modelTypes = ['skipgram', 'cbow']
dims = [300,600]
windows = [3,5]


for mt in modelTypes:
	for dim in dims:
		for window in windows:
			name = str((mt,dim,window))
			print(name)
			model = fasttext.train_unsupervised(path, model=mt, dim=dim, ws=window, minn=2, maxn=5)
			model.save_model("./models/"+ name)
			model.get_word_vector("human")


# model = fasttext.train_unsupervised(path, model='skipgram', dim=300, window=3)
# model = fasttext.train_unsupervised(path, model='skipgram', dim=300, window=5)

# model = fasttext.train_unsupervised(path, model='skipgram', dim=600, window=3, minn=2, maxn=5)
# model = fasttext.train_unsupervised(path, model='skipgram', dim=600, window=5, minn=2, maxn=5)


# model = fasttext.train_unsupervised(path, model='cbow', dim=300,)
# model = fasttext.train_unsupervised(path, model='cbow', dim=600,)
# model = fasttext.train_unsupervised(path, model='cbow', dim=300,)
# model = fasttext.train_unsupervised(path, model='cbow', dim=600,)
