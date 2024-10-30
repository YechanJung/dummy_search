from pinecone import Pinecone
from decouple import config
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pinecone import Pinecone, ServerlessSpec
from create_index import CreateProductIndex
import time

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone_api_key = config('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index_name = "product"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

pc_client = pc.Index(index_name)
es_client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "password"),
    verify_certs=False,
    ssl_show_warn=False
)

index = 'product'

mapping = {
    "mappings": {
        "properties": {
            "item_id": {"type": "text"},
            "brand": {"type": "text"},
            "item_name": {"type": "text"},
            "item_keywords": {"type": "text"},
            "bullet_point": {"type": "text"},
            "product_description": {"type": "text"},
        }
    }
}

es_client.indices.create(index=index, body=mapping)
create_product_index = CreateProductIndex(index, es_client, pc_client, bulk, embedding_model)
create_product_index.create_es_index()
create_product_index.create_vector_index()




def text_search(query):
    q = {
        "query": {
            "query_string": {
                "query": query,
            }
        }
    }

    result = es_client.search(index=index, body=q)
    hits = result['hits']
    return [hit['_source'] for hit in hits['hits']]

def vector_search(query, topk=3):
    
    query_embedding = embedding_model.encode(query).tolist()

    results = pc_client.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=topk,
        include_values=False,
        include_metadata=True
    )
    
    sorted_matches = sorted(results['matches'], key=lambda x: x['score'], reverse=True)
    return sorted_matches




from FlagEmbedding import FlagLLMReranker
reranker = FlagLLMReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def get_product_str(product):
    name = product['item_name']
    bullet_points = product['bullet_point']
    descriptions = product['product_description']
    text = []
    if name:
        text.append("Product Name: %s" % name)
        if bullet_points:
            text.append("- bullet points: %s" % ','.join(bullet_points))
        if descriptions:
            text.append("- description: %s" % ','.join(descriptions))
    return '\n'.join(text)
def llm_reranker(query, docs):
    pairs = [(query, get_product_str(doc)) for doc in docs]
    scores = reranker.compute_score(pairs)
    scored_docs = zip(docs, scores)
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return sorted_docs

def fetch_doc(results):
    ids = []
    for result in results[0]:
        ids.append(result['id'])
    for result in results[1]:
        ids.append(result['item_id'])
    q = {
        "query": {
            "query_string": {
                "query": " OR ".join(ids),
                "default_field": "item_id",
            }
        }
    }
    result = es_client.search(index=index, body=q)
    hits = result['hits']
    return [hit['_source'] for hit in hits['hits']]

def hybrid_search2(query, topk=3):
    results = [vector_search(query, topk), text_search(query)]
    docs = fetch_doc(results)
    ranked_results = llm_reranker(query, docs)
    return ranked_results

start_time = time.time()
results = hybrid_search2("Find me a kitchen table")
print(f"Done in {time.time() - start_time} seconds")
print(results)