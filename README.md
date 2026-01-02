# Anti-Cat Data Collection Server v2.0

Frigate + Gemini 기반 고양이 탐지 데이터 수집 서버입니다.
YOLO NAS 모델 학습을 위한 데이터셋을 자동으로 수집하고 라벨링합니다.

## 주요 기능

### �️ SQLite 데이터베이스
- 이벤트 처리 여부를 SQLite DB로 관리
- 인덱싱을 통한 빠른 조회 (event_id, status, created_at, final_label)
- 트랜잭션 지원으로 데이터 무결성 보장
- 대량의 이벤트도 효율적으로 처리

### 🔄 자동 이벤트 폴링
- MQTT 대신 주기적으로 Frigate API를 통해 이벤트 목록을 가져옵니다
- 최대 3일간의 이벤트를 조회하여 처리합니다
- SQLite DB로 중복 처리를 방지합니다
- 새로운 이벤트가 감지되면 자동으로 스냅샷을 저장합니다

### 🤖 듀얼 검증 시스템
- **Frigate**: 실시간 객체 탐지 결과
- **Gemini**: AI 기반 이미지 분석으로 2차 검증
- 두 시스템의 결과가 일치하면 자동으로 라벨 부여
- 불일치 시 수동 라벨링 대기열에 추가

### 📊 웹 대시보드
- 실시간 통계 확인 (분류별, 상태별)
- 분류 완료/미분류 이벤트 목록 조회
- 수동 라벨링 인터페이스
- YOLO 데이터셋 내보내기

### 🏷️ 라벨 종류
- `person`: 사람
- `cat`: 고양이
- `others`: 기타 (개, 새, 차량 등)

## 📁 프로젝트 구조

```
anti-cat/
├── src/                    # 소스 코드
│   ├── api/               # Frigate API, 대시보드 API
│   ├── services/          # Gemini, 이벤트 처리, 데이터셋 관리
│   ├── utils/             # 설정 및 유틸리티
│   └── main.py            # FastAPI 서버
├── static/                 # 웹 대시보드 프론트엔드
├── data/                   # 수집된 데이터
└── tests/                  # 테스트 코드
```

## 설치 및 실행

### Docker Compose 사용 (권장)

```bash
# 1. 저장소 클론
git clone <repository-url>
cd anti-cat

# 2. 환경변수 설정
cp .env.example .env
nano .env  # 또는 vim .env
# FRIGATE_URL과 GEMINI_API_KEY를 설정하세요

# 3. Docker Compose로 실행
docker-compose up -d

# 4. 로그 확인
docker-compose logs -f

# 5. 중지
docker-compose down
```

### Docker 단독 사용

```bash
# 이미지 빌드
docker build -t anti-cat:latest .

# 컨테이너 실행
docker run -d \
  --name anti-cat \
  -p 8150:8150 \
  -v $(pwd)/data:/app/data \
  -e FRIGATE_URL=http://your-frigate-ip:5000 \
  -e GEMINI_API_KEY=your-key \
  anti-cat:latest

# 로그 확인
docker logs -f anti-cat

# 중지 및 삭제
docker stop anti-cat
docker rm anti-cat
```

### 로컬 Python 실행

```bash
# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
nano .env

# 서버 실행
python -m src.main
```

### 4. 대시보드 접속

브라우저에서 `http://localhost:8150` 접속

## API 엔드포인트

### 통계 및 상태
- `GET /api/stats` - 전체 통계
- `GET /api/status` - 이벤트 처리 상태
- `GET /api/health` - 헬스 체크

### 이벤트 조회
- `GET /api/events` - 전체 이벤트 목록
- `GET /api/events/classified` - 분류 완료된 이벤트
- `GET /api/events/mismatched` - 수동 라벨링 필요한 이벤트
- `GET /api/events/pending` - Gemini 분석 대기 중인 이벤트
- `GET /api/events/{event_id}` - 특정 이벤트 상세

### 이벤트 관리
- `POST /api/events/{event_id}/label` - 수동 라벨 지정
- `POST /api/events/{event_id}/retry` - Gemini 분석 재시도
- `DELETE /api/events/{event_id}` - 이벤트 삭제

### 이미지
- `GET /api/images/{event_id}` - 이벤트 이미지 조회

### 데이터셋
- `POST /api/export/yolo` - YOLO NAS 학습용 데이터셋 내보내기

### 수동 처리
- `POST /api/process/{event_id}` - 특정 이벤트 수동 처리
- `POST /api/process/trigger` - 이벤트 폴링 수동 트리거

## 데이터 구조

```
data/
├── images/           # 스냅샷 이미지
│   └── {event_id}_{timestamp}.jpg
├── labels/           # YOLO 형식 라벨
│   └── {event_id}_{timestamp}.txt
├── mismatched/       # 분류 불일치 데이터
├── events.db         # SQLite 데이터베이스 (이벤트 관리)
├── metadata.json     # 레거시 메타데이터 (호환성)
├── stats.json        # 레거시 통계 (호환성)
└── yolo_export/      # 내보낸 YOLO 데이터셋
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml  # YOLO 설정 파일
```

## YOLO 라벨 형식

```
# <class_id> <x_center> <y_center> <width> <height>
0 0.5 0.5 0.3 0.4
```

- class_id: 0=person, 1=cat, 2=others
- 좌표: 0-1 사이의 정규화된 값

## 워크플로우

1. **이벤트 폴링**: 주기적으로 Frigate에서 새 이벤트 조회
2. **이미지 저장**: 이벤트 스냅샷을 `data/images/`에 저장
3. **Gemini 분석**: 이미지를 Gemini에 전송하여 객체 분류
4. **결과 비교**:
   - Frigate == Gemini: `classified` 상태, 자동 라벨 생성
   - Frigate != Gemini: `mismatched` 상태, 수동 라벨링 대기
5. **수동 라벨링**: 웹 대시보드에서 불일치 이벤트 라벨 지정
6. **데이터셋 내보내기**: YOLO NAS 학습용 포맷으로 내보내기

## 이벤트 상태

| 상태 | 설명 |
|------|------|
| `pending` | Gemini 분석 대기 중 |
| `classified` | 분류 완료 (Frigate & Gemini 일치) |
| `mismatched` | 분류 불일치 (수동 라벨링 필요) |
| `gemini_error` | Gemini API 에러 (자동 재시도 대상) |
| `manual_labeled` | 수동 라벨링 완료 |

## 문제 해결

### Gemini API 에러가 자주 발생함
- `GEMINI_MAX_RETRIES` 값을 늘립니다
- `GEMINI_RETRY_INTERVAL_SECONDS` 값을 늘려 재시도 간격을 늘립니다

### 이벤트가 수집되지 않음
- Frigate URL이 올바른지 확인합니다
- Frigate에 이벤트가 있는지 확인합니다
- 서버 로그를 확인합니다

### 이미지 로드 실패
- 이미지 파일이 `data/images/` 디렉토리에 있는지 확인합니다
- 파일 권한을 확인합니다

## 라이선스

MIT License
