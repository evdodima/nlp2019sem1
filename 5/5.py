from gensim.models import FastText
from gensim.test.utils import get_tmpfile

import pickle
import logging

import os 

import json
import numpy as np
import pandas as pd




input_path = "/Users/rinat/Downloads/xu/aimed_relations_test.txt"

input_data = []


with open(input_path, 'r') as input_file:
	for json_line in input_file:
		element = json.loads(json_line)
		input_data.append((element["label"],element["middle_context"]))

print(input_data[:10])

fname = "/Volumes/新加卷/nlp/models/('cbow', 300, 5).model"
model = FastText.load(fname)


avg_vector = np.array([])
words_number = 0

vectors = []
labels = np.array([])

for context_tuple in input_data:
	for word in context_tuple[1]:
		if word in model.wv.vocab:
			if len(avg_vector) == 0:
				avg_vector = np.array(model.wv[word])
			else:
				avg_vector += np.array(model.wv[word])
			words_number += 1
	avg_vector = avg_vector / words_number
	vectors += [avg_vector]
	labels = np.append(labels, context_tuple[0])
	

print(vectors[:10])

result = pd.DataFrame()
result["label"] = labels
result["vector"] = vectors

print(result)

output = "/Volumes/新加卷/nlp/vectors/out_test.pkl"
result.to_pickle(output)




# logging.basicConfig()
# logging.getLogger().setLevel(logging.INFO)

# for model_path in sorted(os.listdir("")):
# 	if model_path.endswith(".model"):
# 		model = FastText.load(fname)
