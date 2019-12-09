from gensim.models import FastText
from gensim.test.utils import get_tmpfile

import pickle
import logging

import os 

import json
import numpy as np




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
for context_tuple in input_data:
	for word in context_tuple[1]:
		if word in model.wv.vocab:
			if len(avg_vector) == 0:
				avg_vector = np.array(model.wv[word])
			else:
				avg_vector += np.array(model.wv[word])
			words_number += 1
	avg_vector = avg_vector / words_number
	vectors.append((context_tuple[0],avg_vector))

print(vectors[:10])






# logging.basicConfig()
# logging.getLogger().setLevel(logging.INFO)

# for model_path in sorted(os.listdir("")):
# 	if model_path.endswith(".model"):
# 		model = FastText.load(fname)
