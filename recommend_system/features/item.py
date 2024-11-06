from mimesis import Generic
from mimesis.locales import Locale
import random
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# The function creates fake item data using the mimesis library.
# def generate_items(num_items, historical=False):
#     generic = Generic(locale=Locale.EN)
#     items = []
    
#     for _ in range(num_items):
#         if historical:
#             days_ago = random.randint(0, 730)  # random number of days up to two years
#             upload_date = datetime.now() - timedelta(days=days_ago)  # upload date
                        
#         else:
#             upload_date = datetime.now()
        
#         categories = {'clothing': 1, 'home & kitchen': 2, 'health & household': 3, 'appliances': 4, 'electronics': 5, 'books': 6, 'beauty': 7, 'pet': 8, 'garden': 9, 'baby': 10, 'toy': 11}
#         item_category = random.choice(list(categories.keys()))
#         item_title_length = random.randint(0, 50)

#         item = {
#             'item_id': generic.person.identifier(mask='#@@##@'),
#             'category_id': str(categories[item_category]),            
#             'category': item_category,
#             'title_length': str(item_title_length),
#             'listed_date': upload_date.strftime('%Y-%m-%d %H:%M:%S'),
#             'listed_month': upload_date.strftime('%Y-%m'),         
#         }

#         items.append(item)

#     return items


def get_items(num_items):
    
    return []