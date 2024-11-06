from mimesis import Generic
from mimesis.locales import Locale
import random
from datetime import datetime, timedelta


# The function creates fake user data using the mimesis library.
def generate_users(num_users, historical=False):
    generic = Generic(locale=Locale.EN)
    users = []
    
    for _ in range(num_users):
        if historical:
            days_ago = random.randint(0, 730)  # random number of days up to two years
            registration_date = datetime.now() - timedelta(days=days_ago)  # date of registration
        else:
            registration_date = datetime.now()

        user = {
            'user_id': generic.person.identifier(mask='@@###@'),
            'gender': generic.person.gender(),
            'age': random.randint(12, 90),
            'country': 'United States',
            'registration_date': registration_date.strftime('%Y-%m-%d %H:%M:%S'),
            'registration_month': registration_date.strftime('%Y-%m'),
        }
        users.append(user)

    return users
