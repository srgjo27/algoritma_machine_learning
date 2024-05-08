import requests

def get_data_from_api(endpoint):
    # Masukkan URL endpoint API Laravel Anda
    # base_url = 'https://ud-anthony.vpnstores.net/api' # gunakan api yang diambil dari hosting
    base_url = 'http://127.0.0.1:8000/api' # gunakan dalam local untuk melakukan test
    url = f'{base_url}/{endpoint}'

    # Kirim permintaan GET ke API
    response = requests.get(url, verify=False) # aktifkan selama tahap development
    # response = requests.get(url) # aktifkan jika sudah dalam tahap produksi
    response.raise_for_status()

    # Pastikan permintaan berhasil
    if response.status_code == 200:
        # Ubah respons menjadi JSON dan kembalikan
        return response.json()
    else:
        print(f'Error: received status code {response.status_code}')
        return None