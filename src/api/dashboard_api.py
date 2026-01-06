"""웹 대시보드 API 라우터"""
import logging
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.services.dataset_manager import DatasetManager, EventStatus
from src.utils.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api", tags=["dashboard"])

# 전역 DatasetManager 인스턴스
dataset_manager = DatasetManager()


# Request/Response 모델
class ManualLabelRequest(BaseModel):
    """수동 라벨링 요청"""
    label: Optional[str] = None  # person, cat, background


class EventResponse(BaseModel):
    """이벤트 응답"""
    event_id: str
    status: str
    frigate_label: str
    gemini_label: Optional[str] = None
    final_label: Optional[str] = None
    confidence: Optional[str] = None
    image_url: str
    created_at: str
    error_message: Optional[str] = None


class StatsResponse(BaseModel):
    """통계 응답"""
    total_images: int
    by_class: dict
    by_status: dict
    synthetic_count: int = 0


class ModelSelectionSetRequest(BaseModel):
    """family별 대표 모델 수동 설정 요청"""
    family: str
    selected_model_name: str


def _extract_frigate_confidence(event: dict) -> Optional[float]:
    frigate_data = event.get('frigate_data')
    if not isinstance(frigate_data, dict):
        return None

    data = frigate_data.get('data')
    if not isinstance(data, dict):
        return None

    score = data.get('score')
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _get_event_sort_key(event: dict) -> str:
    """이벤트 정렬 키 (event_id의 타임스탬프 기준)"""
    # Synthetic인 경우 source_event_id 사용 (원본 시간)
    # 일반 이벤트는 event_id 사용
    return event.get('source_event_id') or event.get('event_id', '')


def _decorate_event_for_dashboard(event: dict) -> dict:
    event_copy = event.copy()
    event_copy['image_url'] = f"/api/images/{event['event_id']}"
    
    # Synthetic 플래그 추가
    event_copy['is_synthetic'] = event.get('is_synthetic', 0)
    event_copy['source_event_id'] = event.get('source_event_id')

    gemini_result = event.get('gemini_result')
    if isinstance(gemini_result, dict):
        event_copy['gemini_label'] = gemini_result.get('primary_class')
        # Gemini confidence는 서비스에서 'high'/'medium'/'low' 문자열
        event_copy['gemini_confidence'] = gemini_result.get('confidence')
        # 기존 필드명 호환 (프론트/외부 코드가 confidence를 참고할 수 있음)
        event_copy['confidence'] = gemini_result.get('confidence')
        
        # Gemini box 정보 추가 (normalized 0~1000 scale: [ymin, xmin, ymax, xmax])
        detected_objects = gemini_result.get('detected_objects', [])
        if detected_objects:
            event_copy['gemini_boxes'] = [obj.get('bbox_normalized') for obj in detected_objects if 'bbox_normalized' in obj]
        
        # logger.info(f"Event {event['event_id']}: detected_objects={len(detected_objects)}, gemini_boxes={event_copy.get('gemini_boxes')}")

    event_copy['frigate_confidence'] = _extract_frigate_confidence(event)
    
    # Frigate box 정보 추가 (0~1 scale: [x_center, y_center, width, height])
    frigate_data = event.get('frigate_data')
    if isinstance(frigate_data, dict):
        camera = frigate_data.get('camera')
        event_copy['camera'] = camera

        # if box is None and 'box' in frigate_data:
        #     box = frigate_data['box']
        # if box and len(box) == 4:
        #     event_copy['frigate_box'] = box  # [x_center, y_center, width, height]
        #     event_copy['frigate_region'] = region  # [x, y, width, height]
    
    return event_copy


# ===== 통계 API =====

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """전체 통계 조회"""
    return await dataset_manager.get_stats()


@router.get("/status")
async def get_processing_status():
    """이벤트 처리 상태 조회"""
    from src.main import get_event_processor
    processor = get_event_processor()
    return await processor.get_processing_status()


# ===== 이벤트 목록 API =====

@router.get("/events")
async def get_events(
    status: Optional[str] = Query(None, description="상태 필터 (classified, mismatched, pending, gemini_error, manual_labeled)"),
    label: Optional[List[str]] = Query(None, description="라벨 필터 (복수 선택 가능: label=person&label=cat 등)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    이벤트 목록 조회
    
    - status: 상태별 필터링
    - label: 라벨별 필터링 (status가 없을 때만 적용됨)
    - limit: 반환할 최대 개수
    - offset: 시작 위치
    """
    if status:
        events = await dataset_manager.get_events_by_status(status)
        # status 필터링 후 label 필터링 적용 (메모리 상에서)
        if label:
            events = [e for e in events if e.get('final_label') in label]
    else:
        # All events 탭: Synthetic 포함, Label 필터링 적용
        events = await dataset_manager.db.get_all_events(limit=1000, include_synthetic=True)
        if label:
            events = [e for e in events if e.get('final_label') in label]
    
    # 최신순 정렬 (원본 시간 기준)
    events.sort(key=_get_event_sort_key, reverse=True)
    
    # 페이지네이션
    paginated = events[offset:offset + limit]
    
    # 이미지 URL 추가
    result = []
    for event in paginated:
        result.append(_decorate_event_for_dashboard(event))
    
    return {
        "total": len(events),
        "limit": limit,
        "offset": offset,
        "events": result
    }


@router.get("/events/classified")
async def get_classified_events(
    label: Optional[List[str]] = Query(None, description="라벨 필터 (복수 선택 가능)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """분류 완료된 이벤트 목록"""
    events = await dataset_manager.get_classified_events()
    
    # 라벨 필터링
    if label:
        events = [e for e in events if e.get('final_label') in label]
        
    events.sort(key=_get_event_sort_key, reverse=True)
    
    paginated = events[offset:offset + limit]
    
    result = []
    for event in paginated:
        result.append(_decorate_event_for_dashboard(event))
    
    return {
        "total": len(events),
        "events": result
    }


@router.get("/events/mismatched")
async def get_mismatched_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """수동 라벨링이 필요한 이벤트 목록"""
    events = await dataset_manager.get_mismatched_events()
    events.sort(key=_get_event_sort_key, reverse=True)
    
    paginated = events[offset:offset + limit]
    
    result = []
    for event in paginated:
        result.append(_decorate_event_for_dashboard(event))
    
    return {
        "total": len(events),
        "events": result
    }


@router.get("/events/pending")
async def get_pending_events():
    """Gemini 분석 대기 중인 이벤트 목록"""
    events = await dataset_manager.get_pending_gemini_events()
    
    result = []
    for event in events:
        result.append(_decorate_event_for_dashboard(event))
    
    return {
        "total": len(events),
        "events": result
    }


@router.get("/events/synthetic")
async def get_synthetic_events(
    label: Optional[List[str]] = Query(None, description="라벨 필터 (복수 선택 가능)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Synthetic 이벤트 목록 (수동 라벨링 대기)"""
    all_events = await dataset_manager.db.get_all_events(limit=1000)
    
    # is_synthetic=1 필터링
    synthetic_events = [e for e in all_events if e.get('is_synthetic') == 1]
    
    # 라벨 필터링
    if label:
        synthetic_events = [e for e in synthetic_events if e.get('final_label') in label]
        
    synthetic_events.sort(key=_get_event_sort_key, reverse=True)
    
    paginated = synthetic_events[offset:offset + limit]
    
    result = []
    for event in paginated:
        result.append(_decorate_event_for_dashboard(event))
    
    return {
        "total": len(synthetic_events),
        "events": result
    }


# ===== 개별 이벤트 API =====

@router.get("/events/{event_id}")
async def get_event(event_id: str):
    """특정 이벤트 상세 정보"""
    event = await dataset_manager.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    
    return _decorate_event_for_dashboard(event)


@router.post("/events/{event_id}/label")
async def label_event(event_id: str, request: ManualLabelRequest):
    """수동으로 이벤트 라벨 지정"""
    if request.label not in ["person", "cat", "background", None]:
        raise HTTPException(
            status_code=400, 
            detail="라벨은 person, cat, background 중 하나여야 합니다"
        )
    
    success = await dataset_manager.manual_label_event(event_id, request.label)
    
    if not success:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    
    return {
        "status": "success",
        "event_id": event_id,
        "label": request.label
    }


@router.delete("/events/{event_id}")
async def delete_event(event_id: str):
    """이벤트 삭제"""
    success = await dataset_manager.delete_event(event_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    
    return {
        "status": "success",
        "event_id": event_id
    }


@router.post("/events/{event_id}/retry")
async def retry_gemini_analysis(event_id: str):
    """Gemini 분석 재시도"""
    event = await dataset_manager.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    
    image_path = Path(event.get('image_path', ''))
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다")
    
    # Gemini 분석 재시도
    from src.main import get_event_processor
    processor = get_event_processor()
    await processor._analyze_with_gemini(event_id, image_path)
    
    # 업데이트된 이벤트 반환
    updated_event = await dataset_manager.get_event(event_id)
    
    return {
        "status": "success",
        "event": updated_event
    }


# ===== 이미지 API =====

@router.get("/images/{event_id}")
async def get_image(event_id: str):
    """이벤트 이미지 조회"""
    event = await dataset_manager.get_event(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")
    
    image_path_str = event.get('image_path')
    
    if not image_path_str:
        raise HTTPException(status_code=404, detail="스냅샷이 아직 다운로드되지 않았습니다")
    
    image_path = Path(image_path_str)
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다")
    
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        filename=image_path.name
    )


# ===== 수동 이벤트 처리 API =====


@router.post("/models/refresh")
async def refresh_gemini_models():
    """Gemini 모델 레지스트리/대표 모델 선택을 수동으로 갱신"""
    from src.main import get_event_processor

    processor = get_event_processor()
    if not processor.gemini_service:
        raise HTTPException(status_code=503, detail="Gemini 서비스가 초기화되지 않았습니다")

    try:
        refreshed = await processor.gemini_service.refresh_models_if_needed(
            refresh_interval_days=30,
            force=True,
        )
        last_refresh_at = None
        selected_models = processor.gemini_service.models

        if processor.gemini_service.db:
            last_refresh_at = await processor.gemini_service.db.get_kv("gemini_models_last_refresh_at")
            selected_models = await processor.gemini_service.db.get_selected_gemini_models() or selected_models

        return {
            "status": "success",
            "refreshed": bool(refreshed),
            "last_refresh_at": last_refresh_at,
            "selected_models": selected_models,
        }
    except Exception as e:
        logger.error(f"모델 갱신 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/selection")
async def get_gemini_model_selection():
    """DB에 저장된 Gemini 모델 레지스트리/선택 정보를 반환"""
    await dataset_manager.initialize()

    selections = await dataset_manager.db.get_gemini_model_selection_map()
    last_refresh_at = await dataset_manager.db.get_kv("gemini_models_last_refresh_at")
    models = await dataset_manager.db.list_gemini_models()

    families = {}
    for m in models:
        family = m.get('family')
        if not family:
            continue

        model_name = m.get('model_name') or ''
        env_name = model_name.split('models/', 1)[1] if model_name.startswith('models/') else model_name
        
        # Gemini 2.0 모델 제외 (더 이상 지원되지 않음)
        if env_name.startswith('gemini-2.0'):
            continue

        entry = {
            "model_name": env_name,
            "multimodal_status": m.get('multimodal_status'),
            "is_preview": bool(m.get('is_preview')),
            "is_exp": bool(m.get('is_exp')),
            "last_seen_at": m.get('last_seen_at'),
            "last_probed_at": m.get('last_probed_at'),
            "last_probe_error": m.get('last_probe_error'),
        }
        families.setdefault(family, []).append(entry)

    # 각 family는 이름순으로 정렬
    family_list = []
    for fam in sorted(families.keys(), key=lambda x: (len(x), x)):
        options = sorted(families[fam], key=lambda r: r['model_name'])
        family_list.append({
            "family": fam,
            "selected_model_name": selections.get(fam),
            "options": options,
        })

    return {
        "status": "success",
        "last_refresh_at": last_refresh_at,
        "families": family_list,
    }


@router.post("/models/selection")
async def set_gemini_model_selection(request: ModelSelectionSetRequest):
    """family별 대표 모델을 수동으로 변경"""
    await dataset_manager.initialize()

    family = (request.family or '').strip()
    selected = (request.selected_model_name or '').strip()
    if not family or not selected:
        raise HTTPException(status_code=400, detail="family와 selected_model_name은 필수입니다")

    # 해당 family에 모델이 존재하는지 확인 (models/ 접두사 유무 모두 허용)
    all_models = await dataset_manager.db.list_gemini_models()
    allowed = set()
    for m in all_models:
        if (m.get('family') or '').strip() != family:
            continue
        mn = m.get('model_name') or ''
        env_name = mn.split('models/', 1)[1] if mn.startswith('models/') else mn
        
        # Gemini 2.0 모델 제외 (더 이상 지원되지 않음)
        if env_name.startswith('gemini-2.0'):
            continue
            
        allowed.add(env_name)

    if selected not in allowed:
        raise HTTPException(status_code=400, detail=f"선택 불가: family={family}에 {selected} 모델이 없습니다")

    ok = await dataset_manager.db.upsert_gemini_model_selection(
        family=family,
        selected_model_name=selected,
        selection_reason="manual override",
    )
    if not ok:
        raise HTTPException(status_code=500, detail="DB 저장 실패")

    # 런타임 적용 (EventProcessor의 GeminiService 모델 리스트 갱신)
    from src.main import get_event_processor
    processor = get_event_processor()
    if processor.gemini_service:
        await processor.gemini_service.load_selected_models_from_db()
        # 현재 키 기준으로 모델 인덱스 리셋
        processor.gemini_service._init_client(processor.gemini_service.current_key_index, model_index=0)

    return {
        "status": "success",
        "family": family,
        "selected_model_name": selected,
        "selected_models": processor.gemini_service.models if processor.gemini_service else [],
    }
# ===== 수동 이벤트 처리 트리거 API =====
@router.post("/process/trigger")
async def trigger_polling():
    """이벤트 폴링 수동 트리거 (백그라운드 실행)"""
    logger.info("=== trigger_polling 함수 호출됨 ===")
    try:
        from src.main import get_event_processor
        processor = get_event_processor()
        logger.info("EventProcessor 가져오기 성공")
        
        # 백그라운드로 실행
        result = processor.trigger_background_processing()
        logger.info(f"백그라운드 처리 시작: {result}")
        
        return result
    except Exception as e:
        logger.error(f"trigger_polling 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/{event_id}")
async def process_event(event_id: str):
    """특정 이벤트 수동 처리"""
    from src.main import get_event_processor
    processor = get_event_processor()
    result = await processor.process_single_event(event_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# ===== 데이터셋 내보내기 API =====

@router.post("/export/yolo")
async def export_yolo_dataset():
    """YOLO NAS 학습용 데이터셋 내보내기"""
    logger.info("=== 데이터셋 내보내기 시작 ===")
    try:
        output_dir = await dataset_manager.export_yolo_dataset()
        logger.info(f"데이터셋 내보내기 성공: {output_dir}")
        
        # 통계 계산
        classified = await dataset_manager.get_classified_events()
        train_count = int(len(classified) * 0.8)
        val_count = len(classified) - train_count
        
        logger.info(f"Train: {train_count}, Val: {val_count}")
        
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "train_images": train_count,
            "val_images": val_count,
            "config_file": str(output_dir / "dataset.yaml")
        }
        
    except Exception as e:
        logger.error(f"데이터셋 내보내기 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate/bound-boxes")
async def recalculate_bound_boxes(limit: int = Query(100, ge=1, le=1000)):
    """
    기존 이벤트들의 bound_box 재계산 (마이그레이션용)
    
    Args:
        limit: 한 번에 처리할 이벤트 수 (기본: 100)
    """
    logger.info(f"=== bound_box 재계산 시작 (limit={limit}) ===")
    try:
        result = await dataset_manager.recalculate_bound_boxes(limit)
        logger.info(f"bound_box 재계산 완료: {result}")
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        logger.error(f"bound_box 재계산 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regenerate-labels")
async def regenerate_labels():
    """모든 분류된 이벤트의 라벨 파일 재생성"""
    logger.info("=== 라벨 재생성 시작 ===")
    try:
        result = await dataset_manager.regenerate_all_labels()
        logger.info(f"라벨 재생성 완료: {result}")
        
        return {
            "status": "success",
            "success": result["success"],
            "skipped": result["skipped"],
            "failed": result["failed"]
        }
        
    except Exception as e:
        logger.error(f"라벨 재생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===== 로그 API =====

@router.get("/logs")
async def get_logs(lines: int = Query(200, ge=10, le=1000)):
    """로그 파일 조회 (최근 N줄)"""
    try:
        log_file = settings.data_dir / "logs" / "anti-cat.log"
        
        if not log_file.exists():
            return {"logs": [], "message": "로그 파일이 없습니다"}
        
        # 파일에서 마지막 N줄 읽기
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "showing": len(recent_lines)
        }
        
    except Exception as e:
        logger.error(f"로그 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 헬스 체크 API =====

@router.get("/health")
async def health_check():
    """API 헬스 체크"""
    from src.main import get_event_processor
    
    try:
        processor = get_event_processor()
        is_running = processor._is_running
        # 기존 gemini_service 인스턴스 재사용
        gemini_status = processor.gemini_service.health_check()
    except RuntimeError:
        is_running = False
        gemini_status = {"status": "unavailable", "error": "processor not initialized"}
    
    return {
        "status": "healthy",
        "gemini": gemini_status,
        "data_dir": str(settings.data_dir),
        "event_processor_running": is_running
    }


# ===== Export Review API =====

@router.get("/export/list")
async def list_exported_files():
    """Export된 파일 목록 반환"""
    export_dir = settings.data_dir / "yolo_export"
    if not export_dir.exists():
        return {"train": [], "val": []}
    
    result = {"train": [], "val": []}
    
    for split in ["train", "val"]:
        img_dir = export_dir / split / "images"
        if img_dir.exists():
            # 파일명만 리스트로 반환
            result[split] = [f.name for f in img_dir.glob("*.jpg")]
                
    return result

@router.get("/export/image/{split}/{filename}")
async def get_exported_image(split: str, filename: str):
    """Export된 이미지 반환"""
    if split not in ["train", "val"]:
        raise HTTPException(status_code=400, detail="Invalid split")
        
    file_path = settings.data_dir / "yolo_export" / split / "images" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)

@router.get("/export/label/{split}/{filename}")
async def get_exported_label(split: str, filename: str):
    """Export된 라벨 반환"""
    if split not in ["train", "val"]:
        raise HTTPException(status_code=400, detail="Invalid split")
        
    # 이미지 파일명에서 .txt로 변경
    txt_filename = Path(filename).stem + ".txt"
    file_path = settings.data_dir / "yolo_export" / split / "labels" / txt_filename
    
    from fastapi.responses import PlainTextResponse
    
    if not file_path.exists():
        # 라벨 파일이 없는 경우 (background 등) 빈 내용 반환
        return PlainTextResponse("")
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    return PlainTextResponse(content)

@router.get("/export/summary")
async def get_export_summary():
    """Export 요약 정보 반환"""
    summary_path = settings.data_dir / "yolo_export" / "summary.md"
    from fastapi.responses import PlainTextResponse
    
    if not summary_path.exists():
        return PlainTextResponse("No summary available.")
        
    with open(summary_path, 'r') as f:
        content = f.read()
        
    return PlainTextResponse(content)


@router.get("/dataset/review")
async def get_dataset_review(
    split: Optional[str] = Query(None, description="train or val"),
    label: Optional[List[str]] = Query(None, description="Filter by class: person, cat, background"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Dataset Review: YOLO export 폴더의 train/val 데이터 조회"""
    yolo_export_dir = settings.data_dir / "yolo_export"
    
    if not yolo_export_dir.exists():
        return {"items": [], "total": 0, "stats": {}}
    
    # 데이터 수집
    all_items = []
    
    splits_to_check = [split] if split in ['train', 'val'] else ['train', 'val']
    
    for split_name in splits_to_check:
        split_dir = yolo_export_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists():
            continue
            
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            # 라벨 파일 확인
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # 라벨 파싱
            classes_in_image = []
            bboxes = []
            
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # class_id를 class name으로 변환 (0: person, 1: cat)
                                class_name = 'person' if class_id == 0 else 'cat'
                                classes_in_image.append(class_name)
                                bboxes.append({
                                    'class': class_name,
                                    'bbox': [x_center, y_center, width, height]
                                })
                except Exception as e:
                    logger.error(f"Error reading label file {label_file}: {e}")
            
            # 클래스가 없으면 background
            if not classes_in_image:
                classes_in_image = ['background']
            
            # 필터링
            if label:
                # 요청된 라벨 중 하나라도 포함하는지 확인
                if not any(cls in classes_in_image for cls in label):
                    continue
            
            all_items.append({
                'event_id': img_file.stem,
                'split': split_name,
                'classes': list(set(classes_in_image)),  # 중복 제거
                'bboxes': bboxes,
                'image_path': str(img_file),
                'label_path': str(label_file) if label_file.exists() else None
            })
    
    # 통계 계산
    stats = {
        'total': len(all_items),
        'by_split': {},
        'by_class': {'person': 0, 'cat': 0, 'background': 0}
    }
    
    for item in all_items:
        # Split 통계
        split_name = item['split']
        stats['by_split'][split_name] = stats['by_split'].get(split_name, 0) + 1
        
        # Class 통계
        for cls in item['classes']:
            if cls in stats['by_class']:
                stats['by_class'][cls] += 1
    
    # 정렬 및 페이징
    all_items.sort(key=lambda x: x['event_id'], reverse=True)
    total = len(all_items)
    paginated_items = all_items[offset:offset + limit]
    
    return {
        "items": paginated_items,
        "total": total,
        "stats": stats
    }


@router.get("/dataset/images/{event_id}")
async def get_dataset_image(event_id: str):
    """Dataset Review용 이미지 반환"""
    yolo_export_dir = settings.data_dir / "yolo_export"
    
    # train과 val 모두 확인
    for split in ['train', 'val']:
        images_dir = yolo_export_dir / split / "images"
        
        # 다양한 확장자 시도
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = images_dir / f"{event_id}{ext}"
            if img_path.exists():
                return FileResponse(img_path)
    
    raise HTTPException(status_code=404, detail="Image not found")
