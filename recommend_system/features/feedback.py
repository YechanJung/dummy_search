from mimesis import Generic
from mimesis.locales import Locale
import random
from datetime import datetime, timedelta
import numpy as np


# Generate a list of dictionaries, each representing an interaction between a user and an item.
def generate_feedbacks(num_feedbacks, users, items):
    generic = Generic(locale=Locale.EN)
    feedbacks = []

    for _ in range(num_feedbacks):
        user = random.choice(users)
        item = random.choice(items)

        user_registration_date = datetime.strptime(user['registration_date'], '%Y-%m-%d %H:%M:%S')
        item_listed_date = datetime.strptime(item['listed_date'], '%Y-%m-%d %H:%M:%S')
        earliest_date = max(user_registration_date, item_listed_date)
        days_since_earliest = (datetime.now() - earliest_date).days
        random_days = random.randint(0, days_since_earliest)
        interaction_date = earliest_date + timedelta(days=random_days)
        interaction_types = ['views', 'clicks']
        weights = [0.2, 0.8]

        feedback = {
            'interaction_id': generic.person.identifier(mask='####-##-####'),
            'user_id': user['user_id'],
            'item_id': item['item_id'],
            'category_id': item['category_id'],
            'interaction_type': random.choices(interaction_types, weights=weights, k=1)[0],
            'title_length': item['title_length'],
            'interaction_date': interaction_date.strftime('%Y-%m-%d %H:%M:%S'),            
            'interaction_month': interaction_date.strftime('%Y-%m'),
        }

        feedbacks.append(feedback)

    return feedbacks

def build_feedback(item, user_id):
    interaction_date = datetime.now()
    feedback = {
        'interaction_id': generic.person.identifier(mask='####-##-####'),
        'user_id': user_id,
        'item_id': item['item_id'],
        'category_id': item['category_id'],
        'interaction_type': "click",
        'title_length': item['title_length'],
        'interaction_date': interaction_date.strftime('%Y-%m-%d %H:%M:%S'),            
        'interaction_month': interaction_date.strftime('%Y-%m'),
    }
    return feedback