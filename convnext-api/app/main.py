# resnet-api/app/main.py
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# 1. Import the ConvNeXt model and its related functions
from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)

# GPU 메모리 설정 (이전과 동일)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = FastAPI()

# CORS 설정 (Vue 앱 주소들을 포함)
origins = [
    #"http://localhost:5173",
    #"http://172.168.10.29:5173", # Vue 앱을 실행하는 컴퓨터 주소
    #"http://172.168.10.125:5173",# 다른 테스트용 컴퓨터 주소
    #"http://172.168.10.*:5173",# 다른 테스트용 컴퓨터 주소
    #"http://192.168.35.*:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # <- 모든 주소에서 request가 들어올 수 있음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the ConvNeXtTiny model
#    Like before, we explicitly define the input_shape for color images.
model = ConvNeXtTiny(weights="imagenet", input_shape=(224, 224, 3))

@app.get("/")
def read_root():
    return {"status": "ResNet50 API is running"}

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(BytesIO(contents)).convert("RGB")
    pil_img = pil_img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Use the specific preprocess_input and decode_predictions for ConvNeXt
    processed_img = preprocess_input(img_batch)
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    result = [
        {"label": label, "description": description, "probability": f"{prob:.2%}"}
        for label, description, prob in decoded_predictions
    ]
    return {"filename": file.filename, "predictions": result}