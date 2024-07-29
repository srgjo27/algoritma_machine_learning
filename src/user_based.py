import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import dt_profiles_rating_df, preprocess_text

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
    hair_issue_dict = {'normal': 0, 'ketombe': 1, 'kering': 2, 'minyak': 3, 'rontok': 4, 'cabang': 5}
    return hair_issue_dict.get(hair_issue, 0)

def convert_skin_type_body(skin_type): 
    skin_type_dict = {'normal': 0, 'kering': 1, 'minyak': 2, 'kombinasi': 3} 
    return skin_type_dict.get(skin_type, 0)

dt_profiles_rating_df["skin_type_face"] = dt_profiles_rating_df["skin_type_face"].apply(convert_skin_type_face) 
dt_profiles_rating_df["hair_issue"] = dt_profiles_rating_df["hair_issue"].apply(convert_hair_issue) 
dt_profiles_rating_df["skin_type_body"] = dt_profiles_rating_df["skin_type_body"].apply(convert_skin_type_body)

# Precompute user vectors
# Kelompokkan data berdasarkan 'user_id' dan hitung nilai rata-rata dari 'jenis_kulit_wajah', 'masalah_rambut', dan 'jenis_kulit_tubuh' untuk setiap pengguna
user_vectors = dt_profiles_rating_df.groupby('user_id')[['skin_type_face', 'hair_issue', 'skin_type_body']].mean().round(2)
# Setel ulang indeks sehingga 'user_id' menjadi kolom biasa di dalam dataframe
user_vectors.reset_index(inplace=True)
# Menyaring user_id yang tidak ada di dt_profiles_rating_df dan hanya menyimpan vektor pengguna yang relevan
user_vectors = user_vectors[user_vectors['user_id'].isin(dt_profiles_rating_df['user_id'].unique())]
# Simpan user_ids secara terpisah untuk digunakan nanti
user_ids = user_vectors['user_id']
# Hapus kolom 'user_id' karena kolom ini sekarang disimpan di dalam user_ids dan tidak lagi diperlukan untuk perhitungan kemiripan
user_vectors = user_vectors.drop('user_id', axis=1)

user_similarities = cosine_similarity(user_vectors)
user_similarities = pd.DataFrame(user_similarities, index=user_ids, columns=user_ids).round(2)

# Get unique items
items = dt_profiles_rating_df['product_id'].unique()