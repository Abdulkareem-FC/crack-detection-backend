from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import os
import gdown

app = FastAPI(title='Crack Detection API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CRACK_CLASS_INDEX = 0
MODEL_PATH = 'model/crack_model.h5'
GDRIVE_FILE_ID = '1-brcUIgBd5mfWxXlSZhAl1794kF2nGSd'

if not os.path.exists(MODEL_PATH):
    print('Downloading model from Google Drive...')
    os.makedirs('model', exist_ok=True)
    url = f'https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    print('Model downloaded!')

print('Loading AI model...')
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded successfully!')

CRACK_INFO = {
    'Hairline': {
        'causes': [
            'Normal building settlement during early years',
            'Minor thermal expansion and contraction',
            'Surface shrinkage of plaster or render',
        ],
        'solutions': [
            'Clean the crack and apply flexible filler or sealant',
            'Repaint the surface after filling',
            'Monitor over 3 months — if it does not grow, no further action needed',
        ],
        'urgency': 'Low — cosmetic issue only'
    },
    'Diagonal': {
        'causes': [
            'Differential settlement of the foundation',
            'Shear stress from structural loading',
            'Soil movement or subsidence beneath the building',
        ],
        'solutions': [
            'Consult a structural engineer for full assessment',
            'Investigate foundation conditions with a soil survey',
            'Install crack monitors to track movement over time',
            'Do not attempt DIY repair — professional intervention required',
        ],
        'urgency': 'High — possible structural issue'
    },
    'Horizontal': {
        'causes': [
            'Lateral soil or water pressure on walls',
            'Overloading of floor slabs or beams',
            'Foundation failure or wall tie corrosion',
        ],
        'solutions': [
            'Immediate structural inspection is essential',
            'Relieve lateral pressure sources if possible',
            'Wall may require reinforcement, underpinning or rebuilding',
            'Do not occupy the building until assessed',
        ],
        'urgency': 'Critical — structural
