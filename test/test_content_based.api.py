from flask import Flask, request, jsonify
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from connection import get_data_from_api

app = Flask(__name__)

product_data = get_data_from_api('product-data')

# Mengambil data dari database menggunakan query
dt_product_df = pd.DataFrame(product_data)

# Preprocessing data teks
def preprocess_text(text):
    # Case folding
    text = str(text).lower() 
    # Punctuational removal
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenizing
    tokens = word_tokenize(text)
    # Stop words removal
    stop_words = set(stopwords.words('indonesian'))  # Menggunakan stop words bahasa Indonesia
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Menggabungkan kembali token menjadi kalimat
    text = ' '.join(tokens)
    return text

# Preprocessing kolom teks
text_columns = ['gender', 'skin_type_face', 'hair_issue', 'skin_type_body']
for column in text_columns:
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

@app.route('/content-based/<int:user_id>', methods=['GET'])
def content_based_recommendation(user_id):
    # Mengelompokkan data berdasarkan kategori
    grouped_data = dt_product_df.groupby('category_id')

    # Inisialisasi list untuk menyimpan rekomendasi produk
    recommendations = []

    # Iterasi melalui setiap kelompok
    for category_id, group in grouped_data:
        recommendations.extend(calculate_similarity(group, user_id))

    # Menghapus duplikat dan mengubah int64 ke integer
    recommendations = list(set(recommendations))

    # Mengembalikan rekomendasi sebagai list
    return jsonify([int(id) for id in recommendations])

if __name__ == '__main__':
    app.run(debug=True)