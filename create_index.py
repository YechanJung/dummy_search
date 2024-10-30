
import os
import os, gzip, json

def locale_filter(data, locale='en_US'):
    return [data['value'] for data in data if data.get('language_tag') == locale]

def get_product_descriptions(product, locale='en_US'):
    descriptions = locale_filter(product.get('product_description', []), locale)
    return descriptions

def get_product_bullet_points(product, locale='en_US'):
    bullet_points = locale_filter(product.get('bullet_point', []), locale)
    return bullet_points

def get_product_name(product, locale='en_US'):
    item_name = locale_filter(product.get('item_name', []), locale)
    return item_name[0] if item_name else ''

def file_iterator(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# convert to structured product document
def get_product_doc(product):
    name = get_product_name(product)
    bullet_points = get_product_bullet_points(product)
    descriptions = get_product_descriptions(product)
    doc = {}
    if name:
        text = []
        text.append("Product Name: %s" % name)
        if bullet_points:
            text.append("- bullet points: %s" % ','.join(bullet_points))
        if descriptions:
            text.append("- description: %s" % ','.join(descriptions))
        doc[name] = '\n'.join(text)
    return doc

def process_json_gz(file_path):
    text = []
    products = file_iterator(file_path)
    for product in products:
        doc = get_product_doc(product)
        if doc: text.append(doc)
    return text


def process_data():
    all_files = os.listdir("abo-listings/listings/metadata")

    list_data = []
    for file in all_files:
        file_path = 'abo-listings/listings/metadata/' + file
        data = process_json_gz(file_path)
        list_data.extend(data)
    print(f"created {len(list_data)} product documents")
    return list_data


must_have_keys = set(['item_id', 'brand', 'item_name', 'item_keywords', 'bullet_point'])
key_filter =  must_have_keys.union(set('product_description'))

def get_product_item(product, key_filter=key_filter, locale='en_US'):
    item = {}
    for key in key_filter:
        if key == 'item_id':
            item[key] = product[key]
        else:
            values = locale_filter(product.get(key, []), locale)
            if values:
                item[key] = ' '.join(values).strip()
    return item

def json_gz2product(file_path):
    data = []
    products = file_iterator(file_path)
    for product in products:
        # at least 'item_id', 'brand', 'item_name', 'item_keywords', 'bullet_point'
        item = get_product_item(product)
        if item.keys() == must_have_keys:
            data.append(item)
    return data

class CreateProductIndex():
    def __init__(self, index, es_client, pc_client, bulk, embedding_model):
        self.index = index
        self.es_client = es_client
        self.pc_client = pc_client
        self.bulk = bulk
        self.data = process_data()
        self.embeddings = embedding_model.encode(self.data)

        self.window_size = 100
        self.total_samples = 19900   
    def create_es_index(self):
        documents = []
        data_sample = data[:self.total_samples]
        for i in range(self.total_samples):
            data = data_sample[i]
            documents.append({
            '_index': self.index,
            '_id': i+1,
            '_source': {
                'item_id': data.get('item_id', ''),
                'brand': data.get('brand', ''),
                'item_name': data.get('item_name', ''),
                'item_keywords': data.get('item_keywords', ''),
                'bullet_point': data.get('bullet_point', ''),
                'product_description': data.get('product_description', ''),
            }
        })

        success, _ = self.bulk(self.es_client, documents)
        print(f"Successfully indexed {success} documents")
    def create_vector_index(self):
        vectors = []
        for d, e in zip(self.data, self.embeddings):
            vectors.append({
                "id": d['item_id'],
                "values": e,
                "metadata": {'item_name': d['item_name']},
            })
        vector_samples = vectors[:self.total_samples]
        for i in range(self.total_samples - self.window_size + 1):
            window = vector_samples[i: i + self.window_size]
            self.pc_client.upsert(
                vectors=window,
                namespace="ns1"
            )
