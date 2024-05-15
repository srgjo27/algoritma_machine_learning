import pandas as pd
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from connection import get_data_from_api
import redis

# Inisialisasi koneksi Redis
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)

def get_data_from_cache_or_api(endpoint):
    cached_data = redis_client.get(endpoint)
    if cached_data:
        return json.loads(cached_data.decode('utf-8'))
    else:
        # Jika tidak ada data di cache, ambil dari API dan simpan di cache
        api_data = get_data_from_api(endpoint)
        if api_data:
            redis_client.set(endpoint, json.dumps(api_data), ex=3600)  # Simpan data di cache selama 1 jam (3600 detik)
        return api_data

# Mengambil data dari API
profiles_rating = get_data_from_api('profiles-rating')
product_data = get_data_from_api('product-data')

# Mengambil data dari database menggunakan query
dt_profiles_rating_df = pd.DataFrame(profiles_rating)

# Mengambil data dari database menggunakan query
dt_product_df = pd.DataFrame(product_data)

# Menggantikan nilai-nilai yang kosong dengan nilai 0
dt_profiles_rating_df['review_id'] = dt_profiles_rating_df['review_id'].fillna(0).astype(int)
dt_profiles_rating_df['product_id'] = dt_profiles_rating_df['product_id'].fillna(0).astype(int)
dt_profiles_rating_df['rating'] = dt_profiles_rating_df['rating'].fillna(0).astype(float)

# Preprocessing data teks
def preprocess_text(text):
    text = str(text).lower() 
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

