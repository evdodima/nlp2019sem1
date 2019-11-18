import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import os
from glob import glob
import pubmed_parser as pp
from pyspark.sql import Row

import dill

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

spark.sparkContext.addPyFile('pubmed_parser/dist/pubmed_parser-0.2.1-py3.7.egg') # building with Python 3.5

df = spark.read.parquet('./raw_medline.parquet')

def word_tokenize1(x):
	a = x.abstract
	return nltk.word_tokenize(a.lower())

words = df.rdd.map(word_tokenize1)

stop_words=set(stopwords.words('english'))

def remove_stopwords(x):
	arr = list(filter(lambda w : w not in stop_words, x))
	words = ' '.join(arr)
	return words

stopW = words.map(remove_stopwords)

stopW.coalesce(1).saveAsTextFile('result.txt')


# with open('somefile.txt', 'rw+') as the_file:

#     the_file.write('Hello\n')
