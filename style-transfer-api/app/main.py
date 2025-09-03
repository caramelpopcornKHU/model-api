# style-transfer-api/app/main.py

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from io import BytesIO
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- 튜토리얼 코드 시작 (API에 맞게 일부 수정) ---

# 1. 이미지 전처리 및 후처리 함수들
def load_img(image_bytes):
    max_dim = 512
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# 2. VGG 모델 및 손실 함수 정의
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# 3. 모델 전역으로 로드 (서버 시작 시 한 번만 실행)
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
extractor = StyleContentModel(style_layers, content_layers)

# --- 튜토리얼 코드 끝 ---


# 4. FastAPI 엔드포인트 정의
@app.post("/style-transfer", responses={200: {"content": {"image/jpeg": {}}}})
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...)
):
    logging.info("Starting style transfer process...")
    
    # 입력 이미지 읽기 및 전처리
    content_image_bytes = await content_image.read()
    style_image_bytes = await style_image.read()

    content_image_tensor = load_img(content_image_bytes)
    style_image_tensor = load_img(style_image_bytes)
    
    # 스타일과 콘텐츠 타겟 값 추출
    style_targets = extractor(style_image_tensor)['style']
    content_targets = extractor(content_image_tensor)['content']

    # 최적화를 시작할 이미지 변수 생성 (콘텐츠 이미지로 시작)
    image = tf.Variable(content_image_tensor)

    # 옵티마이저 생성
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # 손실 가중치 설정
    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30

    # 최적화 함수 정의
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)
        loss = style_loss + content_loss
        return loss

    # 최적화 루프 실행
    steps = 100 # 반복 횟수 (늘릴수록 품질이 좋아지지만 오래 걸림)
    for n in range(steps):
        train_step(image)
        logging.info(f"Step {n+1}/{steps}")

    # 최종 결과 이미지를 반환
    result_pil_image = tensor_to_image(image)
    buffer = BytesIO()
    result_pil_image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    
    logging.info("Style transfer complete.")
    return Response(content=img_bytes, media_type="image/jpeg")