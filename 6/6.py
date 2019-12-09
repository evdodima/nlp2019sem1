from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report

vectors_train = np.load('/Volumes/新加卷/nlp/vectors/out_train.npy', allow_pickle=True)
print(len(vectors_train[0][0]))

clf = svm.SVC()
clf.fit(vectors_train[0], vectors_train[1])


vectors_test = np.load('/Volumes/新加卷/nlp/vectors/out_test.npy', allow_pickle=True)

predictions = clf.predict(vectors_test[0])

print(classification_report(vectors_test[1], predictions))