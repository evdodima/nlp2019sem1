from sklearn import svm
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

vectors_train = pd.read_pickle('/Volumes/新加卷/nlp/vectors/out_train.pkl')

print(vectors_train)

clf = svm.SVC()
clf.fit(list(vectors_train['vector']), list(vectors_train['label']))


vectors_test = pd.read_pickle('/Volumes/新加卷/nlp/vectors/out_test.pkl')

predictions = clf.predict(list(vectors_test['vector']))

print(classification_report(list(vectors_test['label']), predictions))