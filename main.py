from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io

app = FastAPI(title='Crack Detection API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CRACK_CLASS_INDEX = 0

print('Loading AI model...')
model = tf.keras.models.load_model('model/crack_model.h5')
print('Model loaded successfully!')

# ── CRACK TYPE DEFINITIONS ────────────────────────────────
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
        'urgency': 'Critical — structural integrity at risk'
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
            'Monitor regularly — consult engineer if widening continues',
        ],
        'urgency': 'Medium — monitor closely'
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
        'urgency': 'Medium to High — monitor and assess'
    },
}

# ── CRACK TYPE DETECTION USING OPENCV ────────────────────
def detect_crack_type(image_bytes: bytes) -> str:
    # Convert bytes to OpenCV image
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection to find crack lines
    edges  = cv2.Canny(gray, 50, 150)

    # Find contours of crack regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 'Hairline'

    # Use the largest contour (main crack)
    largest = max(contours, key=cv2.contourArea)

    # Fit a bounding rectangle to measure orientation
    if len(largest) >= 5:
        (x, y), (w, h), angle = cv2.fitEllipse(largest)
    else:
        rect  = cv2.boundingRect(largest)
        x, y, w, h = rect
        angle = 0

    # Fit a line to get the dominant angle
    rows, cols = gray.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=10
    )

    if lines is None:
        return 'Hairline'

    # Calculate average angle of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            angles.append(90.0)
        else:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            angles.append(ang)

    avg_angle = np.mean(angles)

    # Check for stair-step pattern (many short segments at mixed angles)
    short_lines = [l for l in lines if
                   abs(l[0][2] - l[0][0]) + abs(l[0][3] - l[0][1]) < 40]
    stair_ratio = len(short_lines) / len(lines) if lines is not None else 0

    # Classify based on dominant angle
    if stair_ratio > 0.6:
        return 'Stair-step'
    elif avg_angle < 20 or avg_angle > 160:
        return 'Horizontal'
    elif 70 <= avg_angle <= 110:
        return 'Vertical'
    elif 20 <= avg_angle < 70 or 110 < avg_angle <= 160:
        return 'Diagonal'
    else:
        return 'Hairline'

# ── SEVERITY ─────────────────────────────────────────────
def get_severity(confidence: float) -> str:
    if confidence >= 85: return 'High'
    elif confidence >= 60: return 'Medium'
    else: return 'Low'

# ── MAIN PREDICTION ENDPOINT ─────────────────────────────
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # ── VGG16: Detect if crack exists ──
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

    # ── OpenCV: Detect crack type ──
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

# ── HEALTH CHECK ─────────────────────────────────────────
@app.get('/')
def health():
    return {'status': 'Crack Detection API is running', 'model': 'MTDove DLM v1.0'}
