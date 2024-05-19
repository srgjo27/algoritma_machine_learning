from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

    # Jika user belum melakukan rating, kembalikan array kosong
    if not rated_products:
        return []
    
    # Ambil jumlah total rating dan rata-rata rating
    total_ratings = group['rating'].sum()
    average_rating = group['rating'].mean()

    # Inisialisasi list untuk menyimpan rekomendasi produk
    recommendations = []

    # Ambil indeks produk yang dirating oleh user
    rated_indices = [idx for idx, product_id in enumerate(group['id']) if product_id in rated_products]

     # Lakukan iterasi melalui setiap produk yang dirating oleh user
    for query_index in rated_indices:
        # Ambil query dan lakukan reshape
        query = tfidf_matrix[query_index]

        # Hitung similaritas kosinus antara query dan semua produk
        cosine_similarities = cosine_similarity(query, tfidf_matrix).flatten()

        # Urutkan indeks produk berdasarkan similaritas kosinus
        similar_indices = cosine_similarities.argsort()[::-1]

        # Tambahkan produk yang belum dirating oleh user ke dalam list recommendations
        for idx in similar_indices:
            if group.iloc[idx]['id'] not in rated_products:
                # Hitung bobot rekomendasi berdasarkan nilai similaritas, bobot TF-IDF, jumlah rating, dan rata-rata rating
                recommendation_weight = (0.6 * tfidf_matrix[idx, :].sum() + 
                                         0.3 * cosine_similarities[idx] + 
                                         0.05 * (group.iloc[idx]['rating'] / total_ratings) + 
                                         0.05 * (group.iloc[idx]['rating'] / average_rating))
                recommendations.append((group.iloc[idx]['id'], recommendation_weight))
                if len(recommendations) >= 4:  # Hanya ambil 3 atau 4 rekomendasi teratas
                    break

        if len(recommendations) >= 4:  # Hanya ambil 3 atau 4 rekomendasi teratas
            break

    # Urutkan rekomendasi berdasarkan nilai similaritas tertinggi
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Mengambil hanya id produk dari rekomendasi
    recommended_product_ids = [rec[0] for rec in recommendations]

    # Mengembalikan rekomendasi
    return recommended_product_ids