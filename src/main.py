from flask import Flask, jsonify
from user_based import items, user_similarities, dt_profiles_rating_df
from content_based import grouped_data, calculate_similarity
import threading
import time
import importlib
import preprocessing, user_based, content_based

app = Flask(__name__)

def reload_data():
    while True:
        time.sleep(60)
        importlib.reload(preprocessing)
        importlib.reload(user_based)
        importlib.reload(content_based)

        global dt_profiles_rating_df, user_similarities, grouped_data

        dt_profiles_rating_df = preprocessing.dt_profiles_rating_df
        user_similarities = user_based.user_similarities
        grouped_data = content_based.grouped_data

        print("Reloaded data...")

# Fungsi polling akan dijalankan di thread terpisah
polling_thread = threading.Thread(target=reload_data)
polling_thread.daemon = True
polling_thread.start()

@app.route('/user-based/<int:user_id>', methods=['GET'])
def get_user_based_recommendations(user_id):
    # Inisialisasi dictionary kosong untuk menyimpan prediksi untuk setiap item
    predictions = {}

    try:
        # Menjumlahkan skor kemiripan dari user yang diberikan dengan semua user lain
        similarity_sum = user_similarities.loc[user_id].sum()
    except KeyError:
        # Jika user_id tidak ditemukan dalam dataframe user_similarities, kembalikan error 404
        return jsonify({"error": f"User ID {user_id} tidak ditemukan dalam user_similarities"}), 404
    
    if similarity_sum > 0:
        # Jika user memiliki kemiripan dengan user lain
        for item in items:
            # Mendapatkan rating yang diberikan kepada item oleh user lain
            other_user_ratings = dt_profiles_rating_df[dt_profiles_rating_df['product_id'] == item]
            rating_sum = 0
            weight_sum = 0
            for other_user_id in other_user_ratings['user_id']:
                if other_user_id != user_id:
                    # Mendapatkan rating yang diberikan oleh user lain
                    rating = other_user_ratings[other_user_ratings['user_id'] == other_user_id]['rating'].values[0]
                    # Mendapatkan skor kemiripan antara user yang diberikan dan user lain
                    similarity = user_similarities.loc[user_id, other_user_id]
                    # Mengalikan rating dengan kemiripan dan menambahkannya ke rating_sum
                    rating_sum += rating * similarity
                    # Menambahkan skor kemiripan ke weight_sum
                    weight_sum += similarity
            if weight_sum > 0:
                # Menghitung rata-rata rating berbobot untuk item
                predictions[item] = rating_sum / weight_sum
            
    # Mengurutkan item berdasarkan prediksi rating secara menurun dan mendapatkan 16 rekomendasi teratas
    recommendations = sorted(predictions, key=predictions.get, reverse=True)[:16]
    # Mengonversi rekomendasi yang berupa integer menjadi string sebelum mengembalikannya sebagai JSON
    recommendations = [str(item) for item in recommendations]
    return jsonify(recommendations)


@app.route('/content-based/<int:user_id>', methods=['GET'])
def get_content_based_recommendations(user_id):
    # Inisialisasi list untuk menyimpan rekomendasi produk
    recommendations = []

    # Iterasi melalui setiap kelompok
    for _, group in grouped_data:
        recommendations.extend(calculate_similarity(group, user_id))

    # Menghapus duplikat dan mengubah int64 ke integer
    recommendations = list(set(recommendations))

    # Mengembalikan rekomendasi sebagai list
    return jsonify([int(id) for id in recommendations])

if __name__ == '__main__':
    app.run(debug=True)