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
            'Monitor over 3 months - if it does not grow, no further action needed',
        ],
        'urgency': 'Low - cosmetic issue only'
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
            'Do not attempt DIY repair - professional intervention required',
        ],
        'urgency': 'High - possible structural issue'
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
        'urgency': 'Critical - structural integrity at risk'
    },
    'Vertical': {
        'causes': [
            'Uniform foundation settlement',
            'Thermal movement in long walls without expansion joints',
            'Drying shrinkage of concrete or mortar',
        ],
        'solutions': [
            'Fill with flexible sealant to prevent moisture ingress',
            'Install expansion joints if thermal movement is the cause',
            'Monitor regularly - consult engineer if widening continues',
        ],
        'urgency': 'Medium - monitor closely'
    },
    'Stair-step': {
        'causes': [
            'Differential settlement along mortar joints in brick or block walls',
            'Foundation movement on one side of the wall',
            'Poor mortar mix or bond failure over time',
        ],
        'solutions': [
            'Repoint the affected mortar joints with suitable mix',
            'Investigate and stabilise the foundation if settlement is ongoing',
            'Consult a structural engineer if pattern is widening',
        ],
        'urgency': 'Medium to High - monitor and assess'
    },
}

def detect_crack_type(image_bytes: bytes) -> str:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    avg_width = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        perim   = cv2.arcLength(largest, True)
        avg_width = area / perim if perim > 0 else 0

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20, minLineLength=20, maxLineGap=15
    )

    if lines is None:
        return 'Hairline' if avg_width < 3 else 'Vertical'

    angles = []
    line_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if x2 - x1 == 0:
            ang = 90.0
        else:
            ang = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
        angles.append(ang)
        line_lengths.append(length)

    weighted_angle = np.average(angles, weights=line_lengths)
    h_lines = [a for a in angles if a < 25]
    v_lines = [a for a in angles if a > 65]
    mixed   = len(h_lines) > 2 and len(v_lines) > 2
    short_lines = [l for l in line_lengths if l < 35]
    short_ratio = len(short_lines) / len(line_lengths)

    if mixed and short_ratio > 0.5:
        return 'Stair-step'
    elif weighted_angle < 25:
        return 'Horizontal'
    elif weighted_angle > 65:
        return 'Hairline' if avg_width < 2.5 else 'Vertical'
    else:
        return 'Hairline' if avg_width < 2.0 else 'Diagonal'

def get_severity(confidence: float) -> str:
    if confidence >= 85: return 'High'
    elif confidence >= 60: return 'Medium'
    else: return 'Low'

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_output = model.predict(img_array, verbose=0)[0][0]

    if CRACK_CLASS_INDEX == 0:
        crack_prob = float(1.0 - raw_output)
    else:
        crack_prob = float(raw_output)

    has_crack  = crack_prob > 0.5
    confidence = round(crack_prob * 100, 1) if has_crack else round((1 - crack_prob) * 100, 1)
    severity   = get_severity(confidence) if has_crack else 'None'

    if not has_crack:
        return {
            'has_crack'     : False,
            'confidence'    : confidence,
            'severity'      : 'None',
            'crack_type'    : None,
            'causes'        : [],
            'solutions'     : [],
            'urgency'       : None,
            'recommendation': 'No cracks detected. Continue routine monitoring every 6 months.',
        }

    crack_type = detect_crack_type(contents)
    info       = CRACK_INFO[crack_type]

    return {
        'has_crack'     : True,
        'confidence'    : confidence,
        'severity'      : severity,
        'crack_type'    : crack_type,
        'causes'        : info['causes'],
        'solutions'     : info['solutions'],
        'urgency'       : info['urgency'],
        'recommendation': info['urgency'],
    }

@app.get('/')
def health():
    return {'status': 'Crack Detection API is running', 'model': 'MTDove DLM v1.0'}
