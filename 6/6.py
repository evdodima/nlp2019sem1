from sklearn import svm
import numpy as np
import pandas as pd
import os

from sklearn.metrics import classification_report

input_path = "/Users/evdodima/education/pkls/"

for filename in sorted(os.listdir(input_path)):
	if filename.endswith("train.pkl"):
		corpusname = filename.split("_")[0]
		vectors_train = pd.read_pickle(input_path + filename)
		# print(vectors_train)

		clf = svm.SVC(kernel='linear', 
            class_weight='balanced', # penalize
            probability=True)
		clf.fit(list(vectors_train['vector']), list(vectors_train['label']))


		vectors_test = pd.read_pickle(input_path + corpusname + "_test.pkl")

		y_pred = clf.predict(list(vectors_test['vector']))
		y_test = list(vectors_test['label'])

		print(corpusname)
		print(classification_report(y_true=y_test,y_pred=y_pred))