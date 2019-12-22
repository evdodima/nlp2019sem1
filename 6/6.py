from sklearn import svm
import numpy as np
import pandas as pd
import os

from sklearn.utils import resample


from sklearn.metrics import classification_report

input_path = "/Users/evdodima/education/pkls/"

for filename in sorted(os.listdir(input_path)):
	if filename.endswith("train.pkl"):
		corpusname = filename.split("_")[0]
		vectors_train = pd.read_pickle(input_path + filename)
		# print(vectors_train)
		df_majority = vectors_train[vectors_train.label==0]
		df_minority = vectors_train[vectors_train.label==1]

		df_majority_down = resample(df_majority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_minority),    # to match majority class
                                 random_state=123) # reproducible results

		df_down = pd.concat([df_majority_down, df_minority])

		clf = svm.SVC()
		clf.fit(list(df_down['vector']), list(df_down['label']))


		vectors_test = pd.read_pickle(input_path + corpusname + "_test.pkl")

		y_pred = clf.predict(list(vectors_test['vector']))
		y_test = list(vectors_test['label'])

		print(corpusname)
		print(classification_report(y_true=y_test,y_pred=y_pred))