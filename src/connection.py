import requests

def get_data_from_api(endpoint):
    # Masukkan URL endpoint API Laravel Anda
    base_url = 'http://localhost:8000/api'
    url = f'{base_url}/{endpoint}'

    # Kirim permintaan GET ke API
    response = requests.get(url)

    # Pastikan permintaan berhasil
    if response.status_code == 200:
        # Ubah respons menjadi JSON dan kembalikan
        return response.json()
    else:
        print(f'Error: received status code {response.status_code}')
        return None