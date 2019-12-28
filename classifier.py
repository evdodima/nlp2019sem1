from sklearn import svm
import numpy as np
import pandas as pd
import os

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import classification_report

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import pickle


input_path = "/Users/evdodima/education/pkls/"

for filename in sorted(os.listdir(input_path)):
	if filename.endswith("train.pkl"):
		corpusname = filename.split("_")[0]
		vectors_train = pd.read_pickle(input_path + filename)
		# print(vectors_train)
		# df_majority = vectors_train[vectors_train.label==0]
		# df_minority = vectors_train[vectors_train.label==1]

		# df_minority_upsampled = resample(df_minority, 
  #                                replace=True,     # sample with replacement
  #                                n_samples=len(df_majority),    # to match majority class
  #                                random_state=123) # reproducible results

		# df_upsampled = pd.concat([df_majority, df_minority_upsampled])

		df = pd.DataFrame(vectors_train.vector.tolist(), columns=list(range(0, 100)))

		x_train, y_train = df, vectors_train["label"]
		clf = svm.SVC()
		clf.fit(x_train, y_train)


		vectors_test = pd.read_pickle(input_path + corpusname + "_test.pkl")

		x_test = pd.DataFrame(vectors_test.vector.tolist(), columns=list(range(0, 100)))
		y_pred = clf.predict(x_test)
		y_test = vectors_test['label']

		print(filename)
		print(classification_report(y_true=y_test,y_pred=y_pred))
		print("matthews_corrcoef: " + str(matthews_corrcoef(y_test,y_pred)))
		disp = plot_precision_recall_curve(clf, x_test, y_test, name=filename)
		# disp.plot(name=corpusname)
		# skplt.metrics.plot_precision_recall_curve(y_test, y_pred)
		filename = corpusname + "_model.pkl"
		pickle.dump(clf, open(filename, 'wb'))
plt.show()