from sqlalchemy import create_engine

def create_db_connection():
    # Masukkan informasi koneksi database Anda
    db_host = 'localhost'
    db_user = 'root'
    db_password = ''
    db_name = 'deti1931_antonitulang'

    # Buat URL koneksi
    db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}'

    # Buat engine untuk koneksi
    engine = create_engine(db_url)

    # Kembalikan objek koneksi
    return engine