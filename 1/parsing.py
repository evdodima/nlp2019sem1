from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import os
from glob import glob
import pubmed_parser as pp
from pyspark.sql import Row

import nltk
from nltk.corpus import stopwords

conf = SparkConf().\
    setAppName('map').\
    setMaster('local[5]').\
    set('spark.yarn.appMasterEnv.PYSPARK_PYTHON', '~/anaconda3/bin/python').\
    set('spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON', '~/anaconda3/bin/python').\
    set('executor.memory', '8g').\
    set('spark.executor.memoryOverhead', '16g').\
    set('spark.sql.codegen', 'true').\
    set('spark.yarn.executor.memory', '16g').\
    set('yarn.scheduler.minimum-allocation-mb', '500m').\
    set('spark.dynamicAllocation.maxExecutors', '3').\
    set('spark.driver.maxResultSize', '0')

spark = SparkSession.builder.\
    appName("testing").\
    config(conf=conf).\
    getOrCreate()

# spark.sparkContext.addPyFile('pubmed_parser/dist/pubmed_parser-0.2.1-py3.7.egg') # building with Python 3.5

paths = ["/Volumes/新加卷/nlp/input/*.gz", "./data/*.xml.gz"] # full data, test data
data = spark.sparkContext.parallelize(glob(paths[1]), numSlices=500)

# task 1

def parse_abstracts(x):
    arr = []
    for publication_dict in pp.parse_medline_xml(x):
        if publication_dict['abstract'] != "":
            arr.append(Row(abstract=publication_dict['abstract']))
    return arr

abstracts = data.flatMap(parse_abstracts)

# task 2

def word_tokenize1(x):
    a = x.abstract
    return nltk.word_tokenize(a.lower())

words = abstracts.map(word_tokenize1)

stop_words=set(stopwords.words('english'))

def remove_stopwords(x):
    arr = list(filter(lambda w : w not in stop_words, x))
    words = ' '.join(arr)
    return words

result = words.map(remove_stopwords)

output_paths = ["/Volumes/新加卷/nlp/abstracts_tokenized/result.txt", 'result.txt'] # full data, test data
result.coalesce(1).saveAsTextFile(output_paths[1])



