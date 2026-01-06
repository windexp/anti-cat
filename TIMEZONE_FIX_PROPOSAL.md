# Timezone 문제 해결 방안

## 현재 문제
- 모든 datetime이 timezone-naive (시간대 정보 없음)
- Docker TZ는 시스템 시간만 변경, Python datetime은 여전히 naive
- Frigate API는 UTC 기준, 우리 코드는 로컬 기준 → 불일치

## 해결 방법

### 1. 전역 timezone 설정 추가 (추천)

`src/main.py` 또는 `src/utils/config.py`에 추가:

```python
import os
from zoneinfo import ZoneInfo

# 타임존 설정
TZ = ZoneInfo(os.getenv('TZ', 'UTC'))

# datetime 헬퍼 함수
def now_aware():
    """현재 시간 (timezone-aware)"""
    from datetime import datetime
    return datetime.now(TZ)

def to_aware(dt):
    """naive datetime을 aware로 변환"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ)
    return dt

def to_utc_timestamp(dt):
    """datetime을 UTC Unix timestamp로 변환"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.timestamp()
```

### 2. 코드 수정 필요 부분

#### event_processor.py
```python
# 변경 전
now = datetime.now()
after_timestamp = (datetime.now() - timedelta(days=3)).timestamp()

# 변경 후
from src.utils.config import now_aware, to_utc_timestamp
now = now_aware()
after_timestamp = to_utc_timestamp(now_aware() - timedelta(days=3))
```

#### database.py
```python
# 변경 전
'updated_at': datetime.now().isoformat()

# 변경 후
from src.utils.config import now_aware
'updated_at': now_aware().isoformat()  # 자동으로 +09:00 포함
```

#### gemini_service.py
```python
# 변경 전
last_dt = datetime.fromisoformat(last_str)
if datetime.now() - last_dt < timedelta(days=30):

# 변경 후
from src.utils.config import now_aware, to_aware
last_dt = to_aware(datetime.fromisoformat(last_str))
if now_aware() - last_dt < timedelta(days=30):
```

### 3. 마이그레이션 (선택사항)

기존 DB의 naive datetime 문자열들을 aware로 변환:

```python
async def migrate_add_timezone():
    """기존 DB의 datetime 문자열에 timezone 추가"""
    # 기존: "2026-01-02T05:30:00"
    # 변경: "2026-01-02T05:30:00+09:00"
    pass
```

### 4. 최소 변경 방안 (임시)

지금 당장 문제가 없다면:
- 모든 datetime을 KST로 해석한다고 **명시적으로 주석** 작성
- Frigate timestamp 변환 시 주의사항 문서화
- 나중에 timezone-aware로 전환

## 테스트 방법

```python
# 1. timezone-aware 확인
from src.utils.config import now_aware
dt = now_aware()
print(dt)  # 2026-01-02 05:30:00+09:00 (O)
print(dt.tzinfo)  # Asia/Seoul

# 2. Unix timestamp 확인
from datetime import datetime
from zoneinfo import ZoneInfo

dt_kst = datetime(2026, 1, 2, 5, 0, tzinfo=ZoneInfo('Asia/Seoul'))
dt_utc = datetime(2026, 1, 1, 20, 0, tzinfo=ZoneInfo('UTC'))
print(dt_kst.timestamp() == dt_utc.timestamp())  # True

# 3. Frigate API 호출 테스트
# KST 2026-01-02 05:00 = UTC 2026-01-01 20:00 = timestamp 1767292800
```

## 우선순위

1. **긴급**: Frigate API timestamp 계산 (event_processor.py L145)
2. **중요**: 스케줄링 (event_processor.py L75)
3. **보통**: DB 저장 시간, 파일명
4. **낮음**: 기존 데이터 마이그레이션
