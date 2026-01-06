"""이벤트 폴링 및 처리 서비스"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image

from src.utils.config import settings
from src.api.frigate_api import FrigateAPI
from src.services.gemini_service import GeminiService
from src.services.dataset_manager import DatasetManager, EventStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EventProcessor:
    """Frigate 이벤트 폴링 및 처리 서비스"""
    
    def __init__(self):
        self.frigate_api = FrigateAPI()
        self.dataset_manager = DatasetManager()
        self.gemini_service = None  # 나중에 DB와 함께 초기화
        
        self._polling_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._background_task: Optional[asyncio.Task] = None  # 수동 트리거된 백그라운드 태스크
        self._background_running = False  # 백그라운드 처리 실행 중 플래그
    
    async def start(self):
        """폴링 서비스 시작"""
        if self._is_running:
            logger.warning("이벤트 프로세서가 이미 실행 중입니다")
            return
        
        # 데이터베이스 초기화
        await self.dataset_manager.initialize()
        
        # GeminiService를 DB와 함께 초기화
        self.gemini_service = GeminiService(db=self.dataset_manager.db)

        # DB에 저장된 대표 모델이 있으면 우선 로드 (없으면 .env 모델 사용)
        await self.gemini_service.load_selected_models_from_db()
        
        self._is_running = True
        
        # 매일 05시 실행 태스크 시작
        self._polling_task = asyncio.create_task(self._daily_schedule_loop())
        
        logger.info("이벤트 프로세서 시작됨")
        logger.info(f"  - 매일 05시 자동 실행")
        api_key_count = len(self.gemini_service.api_keys) if self.gemini_service else len(settings.gemini_api_key_list)
        model_list = self.gemini_service.models if self.gemini_service else settings.gemini_model_list
        logger.info(f"  - Gemini API 키: {api_key_count}개")
        logger.info(f"  - Gemini 모델(현재 사용): {len(model_list)}개 ({', '.join(model_list)})")
        logger.info(f"  - 1일 최대 처리(대략): 약 {api_key_count * len(model_list) * 20}개")
    
    async def stop(self):
        """폴링 서비스 중지"""
        self._is_running = False
        
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("이벤트 프로세서 중지됨")
    
    async def _daily_schedule_loop(self):
        """매일 05시에 실행되는 스케줄 루프"""
        while self._is_running:
            try:
                now = datetime.now()
                target_time = now.replace(hour=5, minute=0, second=0, microsecond=0)
                
                # 이미 05시가 지났으면 다음날 05시로 설정
                if now >= target_time:
                    target_time += timedelta(days=1)
                
                wait_seconds = (target_time - now).total_seconds()
                
                logger.info(f"다음 실행 예정: {target_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds/3600:.1f}시간 후)")
                
                # 다음 05시까지 대기
                await asyncio.sleep(wait_seconds)
                
                # 05시가 되면 이벤트 처리 실행
                if self._is_running:
                    logger.info("=== 일일 이벤트 처리 시작 ===")
                    
                    # Gemini 클라이언트 첫 번째 키, 첫 번째 모델로 초기화
                    if self.gemini_service:
                        self.gemini_service._init_client(0, model_index=0)

                        # 30일 간격으로만 모델 레지스트리 갱신 + 대표 모델 selection 업데이트
                        try:
                            await self.gemini_service.refresh_models_if_needed(refresh_interval_days=30)
                        except Exception as e:
                            logger.warning(f"월간 Gemini 모델 갱신 실패(계속 진행): {e}")
                    
                    await self._fetch_and_process_events()
                    logger.info("=== 일일 이벤트 처리 완료 ===")
                    
            except Exception as e:
                logger.error(f"스케줄 루프 오류: {e}")
                # 오류 발생 시 1시간 후 재시도
                await asyncio.sleep(3600)
    
    def trigger_background_processing(self) -> Dict[str, Any]:
        """백그라운드에서 이벤트 처리를 시작 (논블로킹)
        
        Returns:
            처리 시작 상태 정보
        """
        if self._background_running:
            return {
                "status": "already_running",
                "message": "이미 백그라운드 처리가 실행 중입니다"
            }
        
        # 백그라운드 태스크 시작
        self._background_task = asyncio.create_task(self._background_fetch_and_process())
        return {
            "status": "started",
            "message": "백그라운드에서 이벤트 처리를 시작했습니다"
        }
    
    async def _background_fetch_and_process(self):
        """백그라운드에서 실행되는 이벤트 처리 래퍼"""
        try:
            self._background_running = True
            await self._fetch_and_process_events()
        except Exception as e:
            logger.error(f"백그라운드 이벤트 처리 중 오류: {e}", exc_info=True)
        finally:
            self._background_running = False
    
    async def _fetch_and_process_events(self):
        """
        Frigate 이벤트 처리 (3단계)
        1. EVENT UPDATE: Frigate에서 이벤트 목록 가져와서 DB에 없는 이벤트 추가
        2. SNAPSHOT DOWNLOAD: DB에 있는 이벤트 중 스냅샷 다운로드 시도
        3. GEMINI ANALYSIS: 스냅샷이 있고 판정 안된 이벤트 Gemini 분석
        """
        logger.info("=" * 60)
        logger.info("이벤트 처리 시작")
        logger.info("=" * 60)
        
        # === STEP 1: EVENT UPDATE ===
        logger.info("\n[1/3] Frigate 이벤트 목록 업데이트")
        await self._step1_update_events()
        
        # === STEP 2: SNAPSHOT DOWNLOAD ===
        logger.info("\n[2/3] 스냅샷 다운로드")
        await self._step2_download_snapshots()
        
        # === STEP 3: GEMINI ANALYSIS ===
        logger.info("\n[3/3] Gemini 분석")
        success = await self._step3_gemini_analysis()
        
        if not success:
            logger.warning("Gemini API 할당량 소진으로 처리 중단")
        
        logger.info("=" * 60)
        logger.info("이벤트 처리 완료")
        logger.info("=" * 60)
    
    async def _step1_update_events(self):
        """Step 1: Frigate에서 이벤트 목록 가져와서 DB 업데이트"""
        try:
            # 최대 3일 전부터의 이벤트 조회
            after_timestamp = (datetime.now() - timedelta(days=settings.event_max_age_days)).timestamp()
            
            logger.info(f"Frigate API 호출 - URL: {self.frigate_api.base_url}")
            
            # 이벤트 목록 조회
            events = self.frigate_api.get_event_list(
                limit=500,
                after=after_timestamp
            )
            
            if events is None:
                logger.warning("Frigate 이벤트 목록 조회 실패")
                return
            
            logger.info(f"조회된 이벤트: {len(events)}개")
            
            # DB에 없는 이벤트만 추가
            new_count = 0
            for event in events:
                event_id = event.get('id')
                if not await self.dataset_manager.is_event_processed(event_id):
                    # 이벤트 등록 (스냅샷 다운로드 전 상태)
                    await self.dataset_manager.register_event_without_snapshot(
                        event_id=event_id,
                        frigate_data=event
                    )
                    new_count += 1
            
            logger.info(f"새로 추가된 이벤트: {new_count}개")
            
        except Exception as e:
            logger.error(f"이벤트 목록 업데이트 중 오류: {e}", exc_info=True)
    
    async def _step2_download_snapshots(self):
        """Step 2: 스냅샷 다운로드"""
        try:
            # 스냅샷이 없는 이벤트 목록 조회
            events = await self.dataset_manager.db.get_events_without_snapshot(limit=500)
            
            if not events:
                logger.info("다운로드할 스냅샷이 없습니다")
                return
            
            logger.info(f"다운로드 대상: {len(events)}개")
            
            success_count = 0
            fail_count = 0
            
            for event in events:
                event_id = event['event_id']
                
                try:
                    # 스냅샷 다운로드
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{event_id}_{timestamp}.jpg"
                    image_path = settings.images_dir / image_filename
                    
                    image_data = self.frigate_api.get_event_snapshot(event_id, image_path)
                    
                    if image_data:
                        # 다운로드 성공 - bound_box 계산
                        bound_box = None
                        is_bound_box = 0
                        
                        try:
                            # 이미지 크기 확인
                            with Image.open(image_path) as img:
                                frame_size = (img.width, img.height)
                            
                            # Frigate box 추출
                            frigate_data = event.get('frigate_data')
                            if isinstance(frigate_data, dict):
                                data_obj = frigate_data.get('data', {})
                                if isinstance(data_obj, dict):
                                    box = data_obj.get('box')
                                    camera = frigate_data.get('camera', 'unknown')
                                    
                                    if box and len(box) == 4:
                                        # YOLO box 변환
                                        yolo_box = self.dataset_manager._get_yolo_bound_box(
                                            box, frame_size, camera, is_cropped=True
                                        )
                                        
                                        if yolo_box:
                                            bound_box = json.dumps(yolo_box)
                                            is_bound_box = 1
                                            logger.info(f"[{event_id}] bound_box 계산 성공: {yolo_box}")
                                        else:
                                            logger.warning(f"[{event_id}] bound_box 계산 실패")
                        except Exception as e:
                            logger.error(f"[{event_id}] bound_box 계산 중 오류: {e}")
                        
                        await self.dataset_manager.db.update_event(
                            event_id,
                            {
                                'snapshot_downloaded': 1,
                                'image_path': str(image_path),
                                'bound_box': bound_box,
                                'is_bound_box': is_bound_box,
                                'updated_at': datetime.now().isoformat()
                            }
                        )
                        success_count += 1
                    else:
                        # 다운로드 실패 (Frigate에 데이터 없음)
                        await self.dataset_manager.db.update_event(
                            event_id,
                            {
                                'snapshot_error': 'Frigate에 스냅샷 데이터 없음',
                                'updated_at': datetime.now().isoformat()
                            }
                        )
                        fail_count += 1
                        logger.warning(f"[{event_id}] 스냅샷 다운로드 실패")
                    
                    # API 부하 방지
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"[{event_id}] 스냅샷 다운로드 중 오류: {e}")
                    await self.dataset_manager.db.update_event(
                        event_id,
                        {
                            'snapshot_error': str(e),
                            'updated_at': datetime.now().isoformat()
                        }
                    )
                    fail_count += 1
            
            logger.info(f"스냅샷 다운로드 완료 - 성공: {success_count}, 실패: {fail_count}")
            
        except Exception as e:
            logger.error(f"스냅샷 다운로드 처리 중 오류: {e}", exc_info=True)
    
    async def _step3_gemini_analysis(self) -> bool:
        """
        Step 3: Gemini 분석
        
        Returns:
            True: 정상 완료 또는 더 이상 처리할 이벤트 없음
            False: API 할당량 소진으로 중단
        """
        try:
            # Gemini 분석이 필요한 이벤트 목록
            events = await self.dataset_manager.db.get_events_for_gemini(limit=500)
            
            if not events:
                logger.info("Gemini 분석이 필요한 이벤트가 없습니다")
                return True
            
            logger.info(f"분석 대상: {len(events)}개")
            
            processed_count = 0
            
            for event in events:
                event_id = event['event_id']
                
                # Synthetic 이벤트는 Gemini 분석 스킵 (수동 라벨링만)
                if event.get('is_synthetic'):
                    logger.info(f"[{event_id}] Synthetic 이벤트 - Gemini 분석 스킵")
                    continue
                
                image_path = Path(event['image_path'])
                
                if not image_path.exists():
                    logger.warning(f"[{event_id}] 이미지 파일 없음: {image_path}")
                    continue
                
                logger.info(f"[{event_id}] Gemini 분석 시작...")
                
                try:
                    # Gemini 분석
                    gemini_result = await self.gemini_service.analyze_image(image_path)
                    
                    if gemini_result:
                        await self.dataset_manager.update_event_with_gemini_result(
                            event_id=event_id,
                            gemini_result=gemini_result
                        )
                        logger.info(f"[{event_id}] 분석 완료: {gemini_result.get('primary_class')}")
                        processed_count += 1
                    else:
                        await self.dataset_manager.update_event_with_gemini_result(
                            event_id=event_id,
                            gemini_result=None,
                            error_message="Gemini API 응답 없음"
                        )
                        logger.warning(f"[{event_id}] Gemini 분석 실패")
                    
                    # API 부하 방지
                    await asyncio.sleep(settings.event_process_delay)
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # 429 할당량 초과 에러 체크
                    is_quota_error = '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower()
                    
                    if is_quota_error:
                        logger.error(f"[{event_id}] Gemini API 할당량 초과!")
                        await self.dataset_manager.db.update_event(
                            event_id,
                            {
                                'error_message': 'API 할당량 초과 (내일 재처리)',
                                'updated_at': datetime.now().isoformat()
                            }
                        )
                        # 할당량 초과 시 즉시 중단
                        logger.info(f"처리 완료: {processed_count}개, 남은 이벤트: {len(events) - processed_count}개")
                        return False
                    else:
                        await self.dataset_manager.update_event_with_gemini_result(
                            event_id=event_id,
                            gemini_result=None,
                            error_message=error_msg
                        )
                        logger.error(f"[{event_id}] Gemini 분석 오류: {error_msg}")
            
            logger.info(f"Gemini 분석 완료: {processed_count}개")
            return True
            
        except Exception as e:
            logger.error(f"Gemini 분석 처리 중 오류: {e}", exc_info=True)
            return False
    
    async def process_single_event(self, event_id: str) -> Dict[str, Any]:
        """단일 이벤트 수동 처리 - 3단계 처리를 해당 이벤트에만 적용"""
        # 이벤트 정보 가져오기
        event_data = self.frigate_api.get_event_details(event_id)
        
        if not event_data:
            return {"error": "이벤트를 찾을 수 없습니다"}
        
        # 이미 처리된 이벤트인지 확인
        if await self.dataset_manager.is_event_processed(event_id):
            existing = await self.dataset_manager.get_event(event_id)
            return {
                "status": "already_exists",
                "event": existing
            }
        
        # Step 1: 이벤트 등록
        await self.dataset_manager.register_event_without_snapshot(
            event_id=event_id,
            frigate_data=event_data
        )
        
        # Step 2: 스냅샷 다운로드
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{event_id}_{timestamp}.jpg"
        image_path = settings.images_dir / image_filename
        
        image_data = self.frigate_api.get_event_snapshot(event_id, image_path)
        
        if not image_data:
            await self.dataset_manager.db.update_event(
                event_id,
                {
                    'snapshot_error': 'Frigate에 스냅샷 데이터 없음',
                    'updated_at': datetime.now().isoformat()
                }
            )
            return {"error": "스냅샷 다운로드 실패"}
        
        await self.dataset_manager.db.update_event(
            event_id,
            {
                'snapshot_downloaded': 1,
                'image_path': str(image_path),
                'updated_at': datetime.now().isoformat()
            }
        )
        
        # Step 3: Gemini 분석
        try:
            gemini_result = await self.gemini_service.analyze_image(image_path)
            
            if gemini_result:
                await self.dataset_manager.update_event_with_gemini_result(
                    event_id=event_id,
                    gemini_result=gemini_result
                )
        except Exception as e:
            logger.error(f"[{event_id}] Gemini 분석 오류: {e}")
        
        return {
            "status": "processed",
            "event": await self.dataset_manager.get_event(event_id)
        }
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """현재 처리 상태 반환"""
        stats = await self.dataset_manager.get_stats()
        pending = len(await self.dataset_manager.get_pending_gemini_events())
        mismatched = len(await self.dataset_manager.get_mismatched_events())
        classified = len(await self.dataset_manager.get_classified_events())
        
        return {
            "is_running": self._is_running,
            "background_processing": self._background_running,
            "stats": stats,
            "pending_gemini": pending,
            "mismatched": mismatched,
            "classified": classified,
            "polling_interval": settings.polling_interval_seconds,
            "gemini_retry_interval": settings.gemini_retry_interval_seconds
        }
