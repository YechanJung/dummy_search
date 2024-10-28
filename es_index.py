import nltk
import pyspark
from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import StringType
import os 
import json
from kafka import KafkaProducer
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType
nltk.download('stopwords')


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return ' '.join(filtered_tokens)

conf = pyspark.SparkConf()
spark = SparkSession.builder.appName("indexer").config(conf=conf).getOrCreate()

path = os.path.join(os.getcwd(), 'abo-listings', 'listings', 'metadata')
data_df = spark.read.json(f"{path}/listings_*.json.gz")

filtered_df = data_df.filter(col("item_name.language_tag") == 'en_US')

preprocess_udf = udf(preprocess_text, StringType())
preprocessed_df = filtered_df.withColumn("preprocessed_keywords", preprocess_udf(col("item_keywords"))) \
                             .withColumn("preprocessed_description", preprocess_udf(col("product_description")))

es_conf = {
    "es.nodes.discovery": "false",
    "es.nodes.data.only": "false",
    "es.net.http.auth.user": "elastic",
    "es.net.http.auth.pass": "password",
    "es.index.auto.create": "true",
    "es.nodes": "127.0.0.1",
    "es.port": "9200",
    "es.mapping.id": "ASIN",
}

preprocessed_df.write.mode("append") \
        .format('org.elasticsearch.spark.sql') \
        .options(**es_conf) \
        .save("abo-listings")

def on_success(metadata):
    print(f"Message produced with the offset: {metadata.offset}")

def on_error(error):
    print(f"An error occurred while publishing the message. {error}")

schema = StructType([
    StructField("ASIN", StringType(), True),
    StructField("item_keywords", StringType(), True),
    StructField("product_description", StringType(), True)
])

kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "abo-listings_update") \
    .option("startingOffsets", "latest") \
    .load()

json_stream = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", schema).alias("data")) \
    .select("data.*")

preprocessed_stream = json_stream \
    .withColumn("preprocessed_keywords", preprocess_udf(col("item_keywords"))) \
    .withColumn("preprocessed_description", preprocess_udf(col("product_description")))

es_update_conf = {
    "es.nodes.discovery": "false",
    "es.nodes.data.only": "false",
    "es.net.http.auth.user": "elastic",
    "es.net.http.auth.pass": "password",
    "es.nodes": "127.0.0.1",
    "es.port": "9200",
    "es.mapping.id": "ASIN",
    "es.write.operation": "update",
    "checkpointLocation": "/tmp/",
    "es.spark.sql.streaming.sink.log.enabled": "false",
}

query = preprocessed_stream.writeStream \
        .outputMode("update") \
        .foreachBatch(lambda df, epoch_id: df.write \
            .format("org.elasticsearch.spark.sql") \
            .options(**es_update_conf) \
            .save("abo-listings")) \
        .start()

query.awaitTermination()


