from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


import pickle
import logging

import os 

import json
import numpy as np
import pandas as pd




input_path = "/Users/evdodima/education/input_for_5/"

input_data = []

for train_filename in sorted(os.listdir(input_path)):
	if train_filename.endswith(".txt"):
		with open(input_path + train_filename, 'r') as train_file:
			for json_line in train_file:
				element = json.loads(json_line)
				input_data.append((element["label"],element["middle_context"]))

print(input_data[:10])

vec_path = "/Users/evdodima/education/fasttext_pmc.vec"
wv = KeyedVectors.load_word2vec_format(vec_path, binary=False)


avg_vector = np.array([])
words_number = 0

vectors = []
labels = np.array([])

all_data = len(input_data)
i = 0
for context_tuple in input_data:
	progress = int((float(i) / all_data) * 1000)
	if progress % 10 == 0:
		print(progress)
	for word in context_tuple[1]:
		try:
			v = np.array(wv.get_vector(word))
			if len(avg_vector) == 0:
				avg_vector = v
			else:
				avg_vector += v
			words_number += 1
		except KeyError:
			continue
			# print(word +" no such word")
	i += 1

	avg_vector = avg_vector / words_number
	vectors += [avg_vector]
	labels = np.append(labels, context_tuple[0])
	
print(vectors[:10])

result = pd.DataFrame()
result["label"] = labels
result["vector"] = vectors

print(result)

output = "/Users/evdodima/education/train.pkl"
result.to_pickle(output)




# logging.basicConfig()
# logging.getLogger().setLevel(logging.INFO)

# for model_path in sorted(os.listdir("")):
# 	if model_path.endswith(".model"):
# 		model = FastText.load(fname)
