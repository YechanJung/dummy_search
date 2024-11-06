import json
# import tensorflow as tf
import torch
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from elasticsearch import Elasticsearch
import pandas as pd

app = FastAPI()

es_client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "password"),
    verify_certs=False,
    ssl_show_warn=False
)

def month_sine(interaction_month):
    coef = np.random.uniform(0, 2 * np.pi) / 12
    month_of_interaction = datetime.strptime(interaction_month, "%Y-%m").month
    return float(np.sin(month_of_interaction * coef))

def month_cosine(interaction_month):     
    coef = np.random.uniform(0, 2 * np.pi) / 12
    month_of_interaction = datetime.strptime(interaction_month, "%Y-%m").month
    return float(np.cos(month_of_interaction * coef))

class Model:
    def __init__(self):            
        self.model = torch.load("query_model")
        
    def predict(self, user, user_feature):
        inputs = {}
        interaction_month = user_feature.pop("interaction_month")       
        inputs["sin_month"] = [month_sine(interaction_month)]
        inputs["cos_month"] = [month_cosine(interaction_month)]
        inputs["user_id"] = [user['user_id']]
        inputs["age"] = [user['age']]
        inputs["gender"] = [user['gender']]
        inputs["country"] = [user['country']]
        inputs["view_count"] = [user_feature['views']]
        inputs["click_count"] = [user_feature['clicks']]
        df = pd.DataFrame(inputs)
        vector = torch.tensor(df)
        prediction = self.model(vector)
        return prediction

model = Model()

@app.get("/recommend/")
async def recommend(user_id):
    q = {
        'size': 1,
    	'query': {
    		'match': {
    			'user_id': user_id
    		}
    	}
    }
    result = es_client.search(index="user", body=q)
    user = result['hits']['hits'][0]['_source']

    q = {
        "size": 1, 
        "query": {
            "match": {
                'user_id': user_id,
            }
        }
    }
    result = es_client.search(index="user_feature", body=q)
    user_feature = result['hits']['hits'][0]['_source']
    
    # model prediction
    prediction = model.predict(user, user_feauture)
    return {"items": recommendations}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)