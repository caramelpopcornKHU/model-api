# yolo-api/app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
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
    "http://localhost:5173",
    "http://172.168.10.29:5173", # Vue 앱을 실행하는 컴퓨터 주소
    "http://172.168.10.125:5173",# 다른 테스트용 컴퓨터 주소
    "http://172.168.10.*:5173",# 다른 테스트용 컴퓨터 주소
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1. 모델 로드
#    'yolov8n.pt'는 가장 작고 빠른 나노 버전의 YOLOv8 모델입니다.
#    처음 실행 시 자동으로 모델 파일을 다운로드합니다.
model = YOLO('yolov8n.pt')

@app.get("/")
def read_root():
    return {"status": "YOLOv8 Object Detection API is running"}

@app.post("/detect-objects", responses={200: {"content": {"image/jpeg": {}}}})
async def classify_image(file: UploadFile = File(...)):
    """
    이미지를 받아 객체를 탐지하고, 결과 바운딩 박스를 그린 이미지를 반환합니다.
    """
    contents = await file.read()
    pil_img = Image.open(BytesIO(contents)).convert("RGB")
    
    # 2. 모델로 객체 탐지 수행
    results = model(pil_img)

    # 3. 결과 처리 및 이미지에 바운딩 박스 그리기
    #    결과는 리스트 형태로 반환되며, 보통 이미지가 하나이므로 첫 번째 요소를 사용합니다.
    result = results[0]
    draw = ImageDraw.Draw(pil_img)

    for box in result.boxes:
        # box.xyxy[0]는 [x1, y1, x2, y2] 형태의 텐서(Tensor)입니다.
        coords = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        label = f"{model.names[class_id]}: {confidence:.2f}"

        # 바운딩 박스 그리기
        draw.rectangle(coords, outline="red", width=2)
        # 라벨 텍스트 그리기
        draw.text((coords[0], coords[1]), label, fill="red")

    # 4. 결과 이미지를 JPEG 형식으로 메모리 버퍼에 저장
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    
    logging.info(f"Detected {len(result.boxes)} objects.")
    
    return Response(content=img_bytes, media_type="image/jpeg")