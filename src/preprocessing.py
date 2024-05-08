import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from connection import get_data_from_api

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

### USER BASED FILLTERING ###
text_colums_user = ['gender','skin_type_face', 'hair_issue', 
                'skin_type_body', 'allergy_history', 'preferred_products', 
                'avoided_products', 'specific_needs']
for column in text_colums_user:
    dt_profiles_rating_df[column] = dt_profiles_rating_df[column].apply(preprocess_text)

# convert skin type, hair issue, skin type body to numeric value (int)
def convert_skin_type_face(skin_type): 
    skin_type_dict = {'normal': 0, 'kering': 1, 'minyak': 2, 'sensitif': 3, 'kombinasi': 4}
    return skin_type_dict.get(skin_type, 0)

def convert_hair_issue(hair_issue): 
    hair_issue_dict = {'normal': 1, 'ketombe': 1, 'kering': 2, 'minyak': 3, 'rontok': 4, 'cabang': 5}
    return hair_issue_dict.get(hair_issue, 0)

def convert_skin_type_body(skin_type): 
    skin_type_dict = {'normal': 0, 'kering': 1, 'minyak': 2, 'kombinasi': 3} 
    return skin_type_dict.get(skin_type, 0)

dt_profiles_rating_df["skin_type_face"] = dt_profiles_rating_df["skin_type_face"].apply(convert_skin_type_face) 
dt_profiles_rating_df["hair_issue"] = dt_profiles_rating_df["hair_issue"].apply(convert_hair_issue) 
dt_profiles_rating_df["skin_type_body"] = dt_profiles_rating_df["skin_type_body"].apply(convert_skin_type_body)

# Precompute user vectors
user_vectors = dt_profiles_rating_df.groupby('user_id')[['skin_type_face', 'hair_issue', 'skin_type_body']].mean().round(2)
user_vectors.reset_index(inplace=True)
user_vectors = user_vectors[user_vectors['user_id'].isin(dt_profiles_rating_df['user_id'].unique())]
user_ids = user_vectors['user_id']
user_vectors = user_vectors.drop('user_id', axis=1)
user_similarities = cosine_similarity(user_vectors)
user_similarities = pd.DataFrame(user_similarities, index=user_ids, columns=user_ids).round(2)

# Get unique items
items = dt_profiles_rating_df['product_id'].unique()
unique_user_ids = dt_profiles_rating_df['user_id'].unique()

def get_recommendations_for_new_user(user_id):
    if user_profile_similar_to_existing(user_id):
        return recommend_based_on_profile(user_id)
    else:
        return recommend_default_products()

def user_profile_similar_to_existing(user_id):
    existing_user_profiles = dt_profiles_rating_df[dt_profiles_rating_df['user_id'] != user_id]
    new_user_profile = dt_profiles_rating_df[dt_profiles_rating_df['user_id'] == user_id].iloc[0]
    similar_users = existing_user_profiles[existing_user_profiles['gender'] == new_user_profile['gender']]
    return not similar_users.empty

def recommend_based_on_profile(user_id):
    user_profile = dt_profiles_rating_df[dt_profiles_rating_df['user_id'] == user_id].iloc[0]
    similar_users = dt_profiles_rating_df[dt_profiles_rating_df['gender'] == user_profile['gender']]
    similar_users_ratings = similar_users.groupby('product_id')['rating'].mean().sort_values(ascending=False)
    return similar_users_ratings.index.tolist()

def recommend_default_products():
    popular_products = dt_profiles_rating_df.groupby('product_id')['rating'].count().sort_values(ascending=False)
    return popular_products.index.tolist()


### CONTENT BASED FILTERING ###
# Preprocessing kolom teks
text_colums_content = ['gender', 'skin_type_face', 'hair_issue', 'skin_type_body']
for column in text_colums_content:
    dt_product_df[column] = dt_product_df[column].apply(preprocess_text)

# Lakukan pengelompokan berdasarkan category_id
grouped_data = dt_product_df.groupby('category_id')

# Fungsi untuk menghitung TF-IDF dan similaritas kosinus serta memberikan rekomendasi
def calculate_similarity(group, user_id):
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Ambil atribut untuk perhitungan (subcategory_id dan skin_type)
    attributes = group[['subcategory_id', 'skin_type_face', 'hair_issue', 'skin_type_body']].astype(str).apply(lambda x: ' '.join(x), axis=1)

    # Hitung TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(attributes)

    # Ambil data produk yang sudah dirating oleh user
    rated_products = set(group[group['user_id'] == user_id]['id'])

    # Inisialisasi list untuk menyimpan rekomendasi produk
    recommendations = []

    # Lakukan iterasi melalui setiap produk sebagai query
    for i, query_index in enumerate(range(len(group))):
        # Ambil query dan lakukan reshape
        query = tfidf_matrix[query_index]

        # Hitung similaritas kosinus antara query dan semua produk
        cosine_similarities = cosine_similarity(query, tfidf_matrix).flatten()

        # Urutkan indeks produk berdasarkan similaritas kosinus
        similar_indices = cosine_similarities.argsort()[::-1]

        # Tambahkan produk yang belum dirating oleh user ke dalam list recommendations
        for idx in similar_indices:
            if group.iloc[idx]['id'] not in rated_products:
                recommendations.append(group.iloc[idx]['id'])
                if len(recommendations) == 3:  # Hanya ambil 3 rekomendasi teratas
                    break

        if len(recommendations) == 3:  # Hanya ambil 3 rekomendasi teratas
            break

    # Menghapus duplikat dan mengembalikan rekomendasi
    return recommendations