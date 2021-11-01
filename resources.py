#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#


import sys
import os
import requests

from qtutil import make_dirs_if_not_exists

resource_list = {
    'ffhq_styleganv2': 'resources/ffhq.pkl',
    'dlib_align_data': 'resources/shape_predictor_68_face_landmarks.dat',
    'styleclip_afro': 'resources/styleclip/afro.pt',
    'styleclip_bobcut':'resources/styleclip/bobcut.pt',
    'styleclip_curly_hair': 'resources/styleclip/curly_hair.pt',
    'styleclip_bowlcut': 'resources/styleclip/bowlcut.pt',
    'styleclip_mohawk': 'resources/styleclip/mohawk.pt',
}

resource_sources = {
    'ffhq_styleganv2'     : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/ffhq.pkl',
    'dlib_align_data'     : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/shape_predictor_68_face_landmarks.dat',
    'styleclip_afro'      : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/styleclip/afro.pt',
    'styleclip_bobcut'    : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/styleclip/bobcut.pt',
    'styleclip_curly_hair': 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/styleclip/curly_hair.pt',
    'styleclip_bowlcut'   : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/styleclip/bowlcut.pt',
    'styleclip_mohawk'    : 'https://dcgi.fel.cvut.cz/~futscdav/chunkmogrify/styleclip/mohawk.pt',
}

def download_url(url, store_at):
    dir = os.path.dirname(store_at)
    make_dirs_if_not_exists(dir)

    # open in binary mode
    with open(store_at, "wb") as file:
        # get request
        response = requests.get(url, stream=True)
        total_size = response.headers.get('content-length')
        downloaded = 0
        total_size = int(total_size)
        for data in response.iter_content(chunk_size=8192):
            downloaded += len(data)
            file.write(data)
            done = int(50 * downloaded / total_size)
            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {100*downloaded/total_size:.02f}%")   
            sys.stdout.flush()
        print()

def check_and_download_all():
    for resource, storage in resource_list.items():
        if not os.path.exists(storage):
            print(f"{resource} not found at {storage}, downloading..")
            download_url(resource_sources[resource], storage)
