import os
import pubmed_parser as pp
from pyspark.sql import Row

path_to_data = pp.list_xml_path('parsed/')
parsed_results = pubmed_oa_rdd.map(lambda x: Row(file_name=os.path.basename(x), **pp.parse_pubmed_xml(x)))
dataframe = parsed_results.toDF()
final_data = dataframe[['abstract']]
pubmed_oa_df_sel.write.parquet('abstracts.parquet')
