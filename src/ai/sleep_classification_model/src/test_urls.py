import requests
import json

TOKEN = '30672-tzKq562EPM9uzs8aKH5K'
FILE = 'polysomnography/edfs/mesa-sleep-0001.edf'

urls = [
    f'https://sleepdata.org/api/v1/datasets/mesa/files/{FILE}',
    f'https://sleepdata.org/datasets/mesa/files/{FILE}',
    f'https://sleepdata.org/api/v1/datasets/mesa/download/{FILE}',
    f'https://sleepdata.org/datasets/mesa/download/{FILE}',
    f'https://sleepdata.org/api/v1/datasets/mesa/files/download.json?path={FILE}',
]

for url in urls:
    try:
        r = requests.get(url, headers={'Authorization': f'token {TOKEN}'}, stream=True, timeout=10, allow_redirects=False)
        print(f"URL: {url}")
        print(f"  Status: {r.status_code}")
        if r.status_code in [301, 302, 303, 307, 308]:
            print(f"  Location: {r.headers.get('Location')}")
        print(f"  Content-Type: {r.headers.get('Content-Type')}")
    except Exception as e:
        print(f"URL: {url} failed with {e}")
