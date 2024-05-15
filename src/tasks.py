from celery import Celery
from preprocessing import dt_product_df
from content_based import calculate_similarity

app = Celery('tasks', broker='redis://127.0.0.1:6379/0')

@app.task
def calculate_recommendations(user_id):
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
    return [int(id) for id in recommendations]