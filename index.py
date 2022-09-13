from FaceExtractor import FaceExtractor
from BackgroundRemover import BackgroundRemover
from Cartoonifier import Cartoonifier
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import requests
import io
import sys
import numpy as np
import cv2

app = Flask(__name__)

def arrToImg(arr):
    img = Image.fromarray(arr.astype('uint8'))
    in_memory_file = io.BytesIO()
    img.save(in_memory_file, "PNG")
    in_memory_file.seek(0)
    return in_memory_file

@app.route('/')
def hello():
    return 'FaceCartoonify'

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    face_extractor = FaceExtractor()
    bg_remover = BackgroundRemover()
    
    source = request.files.get('source')

    in_memory_file = io.BytesIO()
    source.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    decoded = bg_remover.process(decoded)
    decoded = face_extractor.process(decoded)
    
    files = {
        'file_type': (None, 'image'),
        'source': (secure_filename(source.filename), arrToImg(decoded), source.content_type)
    }
    res = requests.post(
        'https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/predict', files=files)
    
    data = np.fromstring(res.content, np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    decoded = bg_remover.process(decoded, 0.3)
    ad = np.concatenate([decoded, np.full((decoded.shape[0], decoded.shape[1], 1), 255, dtype=np.uint8)], axis=-1)
    white = np.all(decoded == [255,255,255], axis=-1)
    ad[white, -1] = 0
    final_img = arrToImg(ad)
    return send_file(final_img, mimetype=source.content_type)
