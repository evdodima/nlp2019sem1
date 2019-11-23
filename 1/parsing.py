from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import os
from glob import glob
import pubmed_parser as pp
from pyspark.sql import Row

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

# /Volumes/新加卷/nlp/input/*.gz
# /Users/rinat/dev/nlp2019sem1/data/*.xml.gz

medline_files_rdd = spark.sparkContext.parallelize(glob('/Volumes/新加卷/nlp/input/*.gz'), numSlices=1000)

def process(x):
    arr = []
    for publication_dict in pp.parse_medline_xml(x):
        if publication_dict['abstract'] != "":
            arr.append(Row(abstract=publication_dict['abstract']))
    return arr

parse_results_rdd = medline_files_rdd.flatMap(process)

# df = parse_results_rdd.toDF()

# df = medline_df.filter("abstract != \"\"")

# print(df.take(10))

# save to parquet
parse_results_rdd.toDF().write.parquet('/Volumes/新加卷/nlp/raw_medline.parquet', mode='overwrite')



