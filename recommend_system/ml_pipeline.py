# import luigi
# from luigi import Task
import pandas as pd
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from elasticsearch import Elasticsearch
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from model.two_tower import QueryTower, ItemTower, TwoTowerModel

es_client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "password"),
    verify_certs=False,
    ssl_show_warn=False
)

spark = SparkSession.builder \
    .appName("JoinFeature") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.15.0") \
    .getOrCreate()

es_options = {
    "es.nodes": "localhost",
    "es.port": "9200",
    "inferSchema": "true",
    "es.net.http.auth.user": "elastic",
    "es.net.http.auth.pass": "password",
    "es.nodes.wan.only": "true",
    "query": '{"query": {"match_all": {}}, "size": 1000, "sort": [{"_timestamp": {"order": "desc"}}]}'
}

batch_size = 2048
user_columns = ["user_id", "gender", "age", "country"]
user_feature_columns = ["user_id", "views", "clicks"]
item_columns = ["item_id", "category", "category_id", "title_length"]
item_feature_columns = ["item_id", "views", "clicks"]

def df_to_tensor(df):
    pandas_df = df.toPandas()
    tensor_dict = {col: torch.tensor(pandas_df[col].values) for col in pandas_df.columns}
    return tensor_dict

def prepare_two_tower_dataset():
    es_reader = spark.read \
        .format("org.elasticsearch.spark.sql") \
        .options(**es_options)

    user_df = es_reader.load("user")
    item_df = es_reader.load("item")
    interaction_df = es_reader.load("interaction")
    user_feature_df = es_reader.load("user_feature")
    item_feature_df = es_reader.load("item_feature")
    interaction_feature_df = es_reader.load("interaction_feature")

    selected_features = interaction_df \
        .join(user_df.select(user_columns), on="user_id") \
        .join(item_df.select(item_columns), on=["item_id", "category_id", "title_length"]) \
        .join(item_feature_df.select(item_feature_columns), on="item_id") \
        .join(interaction_feature_df.select(["interaction_id", "sin_month", "cos_month"]), on="interaction_id") \
        .alias("first") \
        .join(user_feature_df.select(user_feature_columns).alias("second"), on="user_id") \
        .selectExpr("first.*", "first.clicks + second.clicks as click_count", "first.views + second.views as view_count") \
        .drop(*["clicks", "views", "interaction_month"])

    selected_features = selected_features.withColumn("title_length", selected_features["title_length"].cast(DoubleType()))

    train_df, test_df = selected_features.randomSplit(weights=[0.9, 0.1], seed=100)

    train_data = df_to_tensor(train_df)
    test_data = df_to_tensor(test_df)

    return train_df, test_df, train_data, test_data

def get_candidates(item_model, item_data):
    item_model.eval()
    with torch.no_grad():
        item_embeddings = []
        for i in range(len(item_data['item_id'])):
            input_data = {k: v[i:i+1] for k, v in item_data.items()}
            embedding = item_model(input_data)
            item_embeddings.append(embedding)
        candidates = torch.cat(item_embeddings, dim=0)
    return candidates

def train_two_tower():
    mlflow.start_run()

    train_df, test_df, train_data, test_data = prepare_two_tower_dataset()

    user_id_list = train_df.select("user_id").distinct().rdd.map(lambda r: r['user_id']).collect()
    gender_list = train_df.select("gender").distinct().rdd.map(lambda r: r['gender']).collect()
    countries_list = train_df.select("country").distinct().rdd.map(lambda r: r['country']).collect()
    item_id_list = train_df.select("item_id").distinct().rdd.map(lambda r: r['item_id']).collect()
    category_list = train_df.select("category").distinct().rdd.map(lambda r: r['category']).collect()

    print(f"Number of users: {len(user_id_list)}")
    print(f"Number of items: {len(item_id_list)}")

    query_model = QueryTower(user_id_list, gender_list, countries_list)
    item_model = ItemTower(item_id_list, category_list)
    model = TwoTowerModel(query_model, item_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        batch_data = {k: v.to(device) for k, v in train_data.items()}
        user_embeddings, item_embeddings = model(batch_data)
        loss = model.compute_loss(user_embeddings, item_embeddings)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    mlflow.log_metric('loss', loss.item(), step=epoch)

    mlflow.pytorch.log_model(model, "two_tower_model")

    mlflow.end_run()

# class TrainTwoTower(Task):
#     def run(self):
#         train_two_tower()

# if __name__ == '__main__':
#     luigi.build([TrainTwoTower()])
def main():
    train_two_tower()

if __name__ == '__main__':
    main()