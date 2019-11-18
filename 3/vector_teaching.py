import fasttext

model = fasttext.train_unsupervised('./result.txt/part-00000', model='skipgram')

print(model.words)
