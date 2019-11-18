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

medline_files_rdd = spark.sparkContext.parallelize(glob('data/*.gz'), numSlices=1000)

parse_results_rdd = medline_files_rdd.\
    flatMap(lambda x: [Row(file_name=os.path.basename(x), **publication_dict) 
                       for publication_dict in pp.parse_medline_xml(x)])

medline_df = parse_results_rdd.toDF()

df = medline_df[['abstract']].filter("abstract != \"\"")

# save to parquet
df.write.parquet('raw_medline.parquet', mode='overwrite')