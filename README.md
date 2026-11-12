# model-api

> [학습] FastAPI로 이미지 ML 모델을 API로 서빙해 본 실험입니다.

이미지를 올리면 모델이 결과를 돌려주는 것이 신기해서 만들어 본 학습용 저장소입니다. 서비스 구조는 폴더 이름 기준으로 아래와 같습니다.

| 폴더 | 내용 |
|---|---|
| `convnext-api/` | ConvNeXt 모델 API |
| `style-transfer-api/` | 스타일 변환 모델 API |
| `yolo-api/` | YOLO 모델 API |

- 스택: Python, FastAPI, Docker (폴더별 Dockerfile)
- 상태: 실험용입니다. 배포·운영은 하지 않았습니다.
