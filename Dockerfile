FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY static/ ./static/
COPY .env.example ./.env

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/images /app/data/labels /app/data/mismatched

# 포트 노출
EXPOSE 8150

# 환경변수 설정
ENV PYTHONUNBUFFERED=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8150/health')" || exit 1

# 애플리케이션 실행
CMD ["python", "-m", "src.main"]
