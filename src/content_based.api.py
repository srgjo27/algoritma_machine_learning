from flask import Flask, request, jsonify
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import inspect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from connection import create_db_connection

app = Flask(__name__)

# Membuat koneksi ke database
engine = create_db_connection()

# Query untuk mengambil data dari tabel products, product_reviews, dan profiles
q_product = """
    SELECT p.id, p.category_id, p.subcategory_id, p.name,pf.gender, pf.skin_type_face, pf.hair_issue, pf.skin_type_body
    FROM products p
    LEFT JOIN product_reviews pr ON p.id = pr.product_id
    LEFT JOIN profiles pf ON pr.user_id = pf.user_id
"""

# Mengambil data dari database menggunakan query
dt_product_df = pd.read_sql(q_product, engine)

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
def calculate_similarity(group):
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Ambil atribut untuk perhitungan (subcategory_id dan skin_type)
    attributes = group[['subcategory_id', 'skin_type_face', 'hair_issue', 'skin_type_body']].astype(str).apply(lambda x: ' '.join(x), axis=1)

    # Hitung TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(attributes)

    # Tampilkan nilai bobot dari hasil perhitungan TF-IDF
    print("TF-IDF weights:")
    words = tfidf_vectorizer.get_feature_names_out()
    for i, doc in enumerate(tfidf_matrix.toarray()):
        print(f"Product {group.iloc[i]['id']} : {group.iloc[i]['name']}:")
        for j, word in enumerate(words):
            print(f"{word}: {doc[j]:.2f}")
        print()

    # Lakukan iterasi melalui setiap produk sebagai query
    for i, query_index in enumerate(range(len(group))):
        # Ambil query dan lakukan reshape
        query = tfidf_matrix[query_index]

        # Hitung similaritas kosinus antara query dan semua produk
        cosine_similarities = cosine_similarity(query, tfidf_matrix).flatten()

        # Urutkan indeks produk berdasarkan similaritas kosinus
        similar_indices = cosine_similarities.argsort()[::-1]

        # Tampilkan hasil similaritas kosinus untuk setiap produk
        print(f"Query Product: {group.iloc[query_index]['id']} : {group.iloc[query_index]['name']}")
        top_product_names = []
        for j in similar_indices[:10]:  # Ambil 10 hasil teratas
            if j != query_index:
                top_product_names.append(group.iloc[j]['name'])
                print(f"Similarity with Product {group.iloc[j]['id']} - {group.iloc[j]['name']}: {cosine_similarities[j]:.2f}")

        # Cari nilai bobot paling tertinggi-terendah
        max_weight_index = tfidf_matrix[query_index].toarray().argmax()
        min_weight_index = tfidf_matrix[query_index].toarray().argmin()
        max_weight_product = group.iloc[max_weight_index]['id']
        min_weight_product = group.iloc[min_weight_index]['id']
        print(f"Highest weight product: {max_weight_product} : {group.iloc[max_weight_index]['name']}, Weight: {tfidf_matrix[query_index].toarray().max():.2f}")
        print(f"Lowest weight product: {min_weight_product} : {group.iloc[min_weight_index]['name']}, Weight: {tfidf_matrix[query_index].toarray().min():.2f}")

        # Cari hasil tingkat kemiripan yang mendekati 1
        for j, similarity in enumerate(cosine_similarities):
            if similarity > 0.95 and j != query_index:  # Ubah threshold sesuai kebutuhan
                print(f"High similarity with Product {group.iloc[j]['id']} : {group.iloc[j]['name']}: {similarity:.2f}")
                break  # Keluar dari loop setelah menemukan tingkat kemiripan yang mendekati 1
        print()

        # Tampilkan daftar nama 10 produk teratas
        print(30*"=")
        print("Top 10 Recommended Products:")
        for name in top_product_names:
            print(name)
        print()

# Iterasi melalui setiap kelompok
for category_id, group in grouped_data:
    print(f"Category ID: {category_id}")
    # Hitung similaritas untuk setiap kelompok
    calculate_similarity(group)