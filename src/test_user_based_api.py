from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from connection import get_data_from_api

app = Flask(__name__)

# Mengambil data dari API
profiles_rating = get_data_from_api('profiles-rating')

# Mengambil data dari database menggunakan query
dt_profiles_rating_df = pd.DataFrame(profiles_rating)

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

text_columns = ['gender','skin_type_face', 'hair_issue', 
                'skin_type_body', 'allergy_history', 'preferred_products', 
                'avoided_products', 'specific_needs']
for column in text_columns:
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

@app.route('http://localhost:5000/user-based/<int:user_id>')
def get_recommendations(user_id):
    predictions = {}
    similarity_sum = user_similarities.loc[user_id].sum()
    
    if similarity_sum > 0:
        for item in items:
            other_user_ratings = dt_profiles_rating_df[dt_profiles_rating_df['product_id'] == item]
            rating_sum = 0
            for other_user_id in other_user_ratings['user_id']:
                if other_user_id != user_id:
                    rating = other_user_ratings[other_user_ratings['user_id'] == other_user_id]['rating'].values[0]
                    similarity = user_similarities.loc[user_id, other_user_id]
                    rating_sum += rating * similarity
            predictions[item] = rating_sum / similarity_sum
            
    recommendations = sorted(predictions, key=predictions.get, reverse=True)[:10]
    # Convert integer recommendations to strings before returning as JSON
    recommendations = [str(item) for item in recommendations]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)