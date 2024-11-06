import numpy as np
import pyspark
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from elasticsearch import Elasticsearch


# {'interaction_id': '2851-13-6165', 'user_id': 'SW440P', 'item_id': '0PH12A', 'category_id': '7', 'interaction_type': 'views', 'title_length': '1', 'interaction_date': '2024-10-30 21:13:31', 'interaction_month': '2024-10'}
schema = StructType([
    StructField("interaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("item_id", StringType(), True),
    StructField("category_id", StringType(), True),
    StructField("interaction_type", StringType(), True),
    StructField("title_length", StringType(), True),
    StructField("interaction_date", StringType(), True),
    StructField("interaction_month", StringType(), True),
])

@udf(returnType=FloatType())
def month_sine(interaction_month):
    coef = np.random.uniform(0, 2 * np.pi) / 12
    month_of_interaction = datetime.strptime(interaction_month, "%Y-%m").month
    return float(np.sin(month_of_interaction * coef))

@udf(returnType=FloatType())
def month_cosine(interaction_month):     
    coef = np.random.uniform(0, 2 * np.pi) / 12
    month_of_interaction = datetime.strptime(interaction_month, "%Y-%m").month
    return float(np.cos(month_of_interaction * coef))

def write_to_multiple_sinks(df, _):
    df = df.cache()

    # featurize every 1 hour window
    keys = ["item_id", "category_id", "title_length", "interaction_date", "interaction_month"]
    item_feature = df.groupBy(*keys) \
                     .pivot("interaction_type", ["clicks", "views"]) \
                     .count() \
                     .na.fill(0)

    keys = ["user_id", "category_id", "title_length", "interaction_date", "interaction_month"]
    user_feature = df.groupBy(*keys) \
                     .pivot("interaction_type", ["clicks", "views"]) \
                     .count() \
                     .na.fill(0)
    
    interaction_feature = df.withColumn("sin_month", month_sine(col("interaction_month"))) \
                            .withColumn("cos_month", month_cosine(col("interaction_month"))) \
                            .drop(*["title_length", "interaction_type", "item_id", "user_id", "category_id"])

    # debug
    item_feature.write.format('console').save()
    user_feature.write.format('console').save()
    interaction_feature.write.format('console').save()
    
    es_sink_options = {
        "es.nodes": "localhost",
        "es.port": "9200",
        "es.net.http.auth.user": "elastic",
        "es.net.http.auth.pass": "password",
        "es.nodes.wan.only": "true"
    }

    item_feature.write \
                .mode("append") \
                .format("org.elasticsearch.spark.sql") \
                .options(**es_sink_options) \
                .option("es.mapping.id", "item_id") \
                .save("item_feature")
    
    user_feature.write \
                .mode("append") \
                .format("org.elasticsearch.spark.sql") \
                .options(**es_sink_options) \
                .option("es.mapping.id", "user_id") \
                .save("user_feature")

    interaction_feature.write \
                       .mode("append") \
                       .format("org.elasticsearch.spark.sql") \
                       .options(**es_sink_options) \
                       .option("es.mapping.id", "interaction_id") \
                       .save("interaction_feature")

def streaming_pipeline():
    spark = SparkSession.builder \
        .appName("GenerateFeature") \
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.15.0") \
        .getOrCreate()

    source_options = {
        "kafka.bootstrap.servers": "localhost:9092",
        "subscribe": "feedback_update",
        "startingOffsets": "latest",
    }

    df = spark.readStream \
        .format("kafka") \
        .options(**source_options) \
        .load() \
        .withColumn("json_data", from_json(col("value").cast("string"), schema)) \
        .withColumn("interaction_id", col("json_data.interaction_id")) \
        .withColumn("user_id", col("json_data.user_id")) \
        .withColumn("item_id", col("json_data.item_id")) \
        .withColumn("category_id", col("json_data.category_id")) \
        .withColumn("interaction_type", col("json_data.interaction_type")) \
        .withColumn("title_length", col("json_data.title_length")) \
        .withColumn("interaction_date", col("json_data.interaction_date")) \
        .withColumn("interaction_month", col("json_data.interaction_month")) \
        .drop(*["json_data", "value", "key", "topic", "partition", "offset", "timestamp", "timestampType"])

    df.writeStream.foreachBatch(write_to_multiple_sinks) \
      .start() \
      .awaitTermination()

    spark.stop()


if __name__ == "__main__":
    streaming_pipeline()
    