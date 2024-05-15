from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import dt_product_df, preprocess_text 

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