# resnet-api/Dockerfile

# GPU 버전의 TensorFlow 이미지를 베이스로 사용
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# 컨테이너 내부는 8000번 포트를 사용 (나중에 밖에서 8001번과 연결)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]