from flask import Flask, request, jsonify
from preprocessing import dt_profiles_rating_df, dt_product_df, user_similarities, items, calculate_similarity

app = Flask(__name__)

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
