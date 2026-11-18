# 폐배터리 금속 회수율 최적화 — 데모 이미지
#
# 파이프라인을 빌드 때 돌려 정제본·모델·씬 데이터를 이미지에 굽는다.
#   - 기동이 즉시다 (컨테이너가 올라오면 바로 쓸 수 있다)
#   - 빌드 로그 자체가 전처리→학습→추천이 이어진다는 증거가 된다
#   - 모델을 컨테이너 안에서 학습하므로 scikit-learn 버전 불일치로 인한
#     joblib 언피클 문제가 생기지 않는다
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# lightgbm 은 OpenMP 런타임을 요구한다. slim 이미지에는 없다.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 의존성을 먼저 깔아 레이어 캐시를 살린다 (소스만 바뀌면 재설치하지 않는다)
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

# 모듈 1 정제 → 모듈 2 학습. 3D 화면(docs/modeling/index.html)은 단일 파일이라 빌드할 것이 없다.
# 원본 .xlsx 는 레포에 들어 있으므로 외부 다운로드가 없다.
RUN python -m module1_preprocessing.cleaning \
 && python -m module2_prediction.train

EXPOSE 8501 8000

# compose 가 서비스별로 덮어쓴다. 단독 실행 시 기본은 데모 화면.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
