"""FastAPI 메인 애플리케이션"""
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from src.utils.config import settings
from src.services.event_processor import EventProcessor
from src.api.dashboard_api import router as dashboard_router

# 로그 디렉토리 생성
log_dir = settings.data_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "anti-cat.log"

# 루트 로거 설정 (파일 + 콘솔)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 포맷터
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 파일 핸들러 (rotating, 10MB, 최대 5개 백업)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 기존 핸들러 제거 후 새로 추가
root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# 모든 주요 모듈의 로거 레벨 설정
logging.getLogger('src').setLevel(logging.INFO)
logging.getLogger('src.api').setLevel(logging.INFO)
logging.getLogger('src.services').setLevel(logging.INFO)
logging.getLogger('uvicorn').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# 전역 event_processor 인스턴스
event_processor: Optional[EventProcessor] = None


def get_event_processor() -> EventProcessor:
    """event_processor 인스턴스를 반환 (의존성 주입용)"""
    if event_processor is None:
        raise RuntimeError("EventProcessor가 초기화되지 않았습니다")
    return event_processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    global event_processor
    
    # 시작 시
    logger.info("=== Anti-Cat 서버 시작 ===")
    logger.info(f"Frigate URL: {settings.frigate_url}")
    logger.info(f"데이터 저장 경로: {settings.data_dir}")
    logger.info(f"폴링 간격: {settings.polling_interval_seconds}초")
    logger.info(f"Gemini 재시도 간격: {settings.gemini_retry_interval_seconds}초")
    
    # 이벤트 프로세서 인스턴스 생성 및 시작
    event_processor = EventProcessor()
    await event_processor.start()
    
    yield
    
    # 종료 시
    if event_processor:
        await event_processor.stop()
    logger.info("=== Anti-Cat 서버 종료 ===")


app = FastAPI(
    title="Anti-Cat Data Collection Server",
    description="Frigate + Gemini 기반 고양이 탐지 데이터 수집 서버 (폴링 모드)",
    version="2.0.0",
    lifespan=lifespan
)

# 대시보드 API 라우터 추가
app.include_router(dashboard_router)

# 정적 파일 서빙 (대시보드 프론트엔드)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """대시보드 메인 페이지"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "running",
        "service": "Anti-Cat Data Collection Server",
        "version": "2.0.0",
        "message": "대시보드 파일이 없습니다. /api/stats에서 API를 사용하세요."
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    is_running = event_processor._is_running if event_processor else False
    
    # 기존 gemini_service 인스턴스 재사용
    if event_processor and event_processor.gemini_service:
        gemini_health = event_processor.gemini_service.health_check()
    else:
        gemini_health = {"status": "unavailable", "error": "service not initialized"}
    
    return {
        "status": "healthy",
        "event_processor_running": is_running,
        "frigate_url": settings.frigate_url,
        "data_dir": str(settings.data_dir),
        "gemini": gemini_health
    }


@app.get("/routes")
async def list_routes():
    """등록된 모든 라우트 확인 (디버깅용)"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": routes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=True
    )
