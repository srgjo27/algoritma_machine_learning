import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from connection import get_data_from_api

# Mengambil data dari API
profiles_rating = get_data_from_api('profiles-rating')
product_data = get_data_from_api('product-data')

# Mengambil data dari database menggunakan query
dt_profiles_rating_df = pd.DataFrame(profiles_rating)
dt_product_df = pd.DataFrame(product_data)

# Menggantikan nilai-nilai yang kosong dengan nilai 0
dt_profiles_rating_df['review_id'] = dt_profiles_rating_df['review_id'].fillna(0).astype(int)
dt_profiles_rating_df['product_id'] = dt_profiles_rating_df['product_id'].fillna(0).astype(int)
dt_profiles_rating_df['rating'] = dt_profiles_rating_df['rating'].fillna(0).astype(float)

# Menggantikan nilai-nilai yang kosong dengan nilai 0
dt_product_df['rating'] = dt_product_df['rating'].fillna(0).astype(float)

# Preprocessing data teks
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text