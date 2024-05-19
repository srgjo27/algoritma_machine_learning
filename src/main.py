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
        # Memuat ulang data dari API setiap 60 detik
        time.sleep(10)
        importlib.reload(preprocessing)
        importlib.reload(user_based)
        importlib.reload(content_based)

        global dt_profiles_rating_df, user_similarities, grouped_data
        dt_profiles_rating_df = preprocessing.dt_profiles_rating_df
        user_similarities = user_based.user_similarities
        grouped_data = content_based.grouped_data

        print("Reload...")

# Fungsi polling akan dijalankan di thread terpisah
polling_thread = threading.Thread(target=reload_data)
polling_thread.daemon = True
polling_thread.start()

@app.route('/user-based/<int:user_id>', methods=['GET'])
def get_user_based_recommendations(user_id):
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