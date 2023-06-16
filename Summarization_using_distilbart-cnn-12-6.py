# Databricks notebook source
# DBTITLE 1,Extract the dataset using sh command
# MAGIC %sh
# MAGIC #To keep it simple, we'll download and extract the dataset using standard bash commands 
# MAGIC #Install 7zip to extract the file
# MAGIC apt-get install -y p7zip-full
# MAGIC
# MAGIC rm -rf /tmp/quant || true
# MAGIC mkdir -p /tmp/quant
# MAGIC cd /tmp/quant
# MAGIC #Download & extract the quant archive
# MAGIC curl -L https://archive.org/download/stackexchange/quant.stackexchange.com.7z -o quant.7z
# MAGIC 7z x quant.7z 
# MAGIC #Move the dataset to our main bucket
# MAGIC rm -rf /dbfs/dbdemos/product/llm/quant/raw || true
# MAGIC mkdir -p /dbfs/dbdemos/product/llm/quant/raw
# MAGIC cp -f Posts.xml /dbfs/dbdemos/product/llm/quant/raw

# COMMAND ----------

# DBTITLE 1,Our Q&A dataset is ready
# MAGIC %fs ls /dbdemos/product/llm/quant/raw

# COMMAND ----------

# DBTITLE 1,Review our raw Q&A dataset
quant_raw_path = "/dbdemos/product/llm/quant/raw"
print(f"loading raw xml dataset under {quant_raw_path}")
raw_quant = spark.read.format("xml").option("rowTag", "row").load(f"{quant_raw_path}/Posts.xml")
display(raw_quant)

# COMMAND ----------

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, udf, length, pandas_udf

#UDF to transform html content as text
@pandas_udf("string")
def html_to_text(html):
  return html.apply(lambda x: BeautifulSoup(x).get_text())

quant_df =(raw_quant
                  .filter("_Score >= 5") # keep only good answer/question
                  .filter(length("_Body") <= 1000) #remove too long questions
                  .withColumn("body", html_to_text("_Body")) #Convert html to text
                  .withColumnsRenamed({"_Id": "id", "_ParentId": "parent_id"})
                  .select("id", "body", "parent_id"))

display(quant_df)

# COMMAND ----------

# DBTITLE 1,Get 10 longest answers
docs_df = quant_df.withColumn('body_length', length(col('body')))\
                    .orderBy(col('body_length').desc()).limit(10)\
                    .select('body','body_length')
display(docs_df)

# COMMAND ----------

# DBTITLE 1,Summarizing the documents
from typing import Iterator
import pandas as pd 
from transformers import pipeline
import torch

#Make sure we clean the memory
try:
    torch.cuda.empty_cache()
    from numba import cuda
    cuda.get_current_device().reset()
except Exception as e:
    print(f"Couldn't clean the memory: {e}")

@pandas_udf("string")
def summarize(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Load the model for summarization
    torch.cuda.empty_cache()
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device="cuda:0")
    def summarize_txt(text):
      return summarizer(text)[0]['summary_text']

    for serie in iterator:
        # get a summary for each row
        yield serie.apply(summarize_txt)

docs_df = docs_df.repartition(1)\
                 .withColumn("summary", summarize("body"))\
                 .withColumn('summary_length', length(col('summary')))
display(docs_df)

# COMMAND ----------


