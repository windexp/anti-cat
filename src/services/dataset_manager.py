"""데이터셋 저장 및 라벨링 관리 모듈"""
import json
import logging
import shutil
import random
from collections import defaultdict
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from enum import Enum
from src.utils.config import settings
from src.services.database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventStatus(str, Enum):
    """이벤트 처리 상태"""
    PENDING = "pending"           # Gemini 분석 대기중
    CLASSIFIED = "classified"     # 분류 완료 (Frigate & Gemini 일치)
    MISMATCHED = "mismatched"     # 분류 불일치 (수동 라벨링 필요)
    GEMINI_ERROR = "gemini_error" # Gemini API 에러 (재시도 필요)
    MANUAL_LABELED = "manual_labeled"  # 수동 라벨링 완료


class DatasetManager:
    """YOLO 형식 데이터셋 관리 (SQLite 기반)"""
    
    # 클래스 ID 매핑 (null은 매핑 없음)
    CLASS_MAPPING = {
        "person": 0,
        "cat": 1
    }
    
    # NOTE: background는 '학습용 negative sample' 의미로 라벨 파일은 빈 파일로 생성한다.
    #       과거 데이터/호환을 위해 None도 입력으로는 허용한다.
    CLASS_NAMES = ["person", "cat", "background", None]  # None(legacy) -> background
    
    def __init__(self):
        self.images_dir = settings.images_dir
        self.labels_dir = settings.labels_dir
        self.mismatched_dir = settings.mismatched_dir
        self.synthetic_dir = settings.data_dir / "synthetic"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.synthetic_probability = 0.3  # 30% 확률
    
    async def initialize(self):
        """데이터베이스 초기화"""
        await self.db.initialize()
    
    async def is_event_processed(self, event_id: str) -> bool:
        """이벤트가 이미 처리되었는지 확인"""
        return await self.db.event_exists(event_id)
    
    async def get_event_status(self, event_id: str) -> Optional[str]:
        """이벤트의 현재 상태 반환"""
        event = await self.db.get_event(event_id)
        return event.get('status') if event else None
    
    async def register_event_without_snapshot(
        self,
        event_id: str,
        frigate_data: Dict[str, Any]
    ) -> str:
        """
        새 이벤트 등록 (스냅샷 다운로드 전)
        
        Args:
            event_id: Frigate 이벤트 ID
            frigate_data: Frigate 이벤트 데이터
            
        Returns:
            등록된 이벤트 ID
        """
        timestamp = datetime.now().isoformat()
        
        event_data = {
            "event_id": event_id,
            "status": EventStatus.PENDING.value,
            "frigate_label": frigate_data.get('label', 'unknown'),
            "frigate_data": frigate_data,
            "image_path": None,
            "snapshot_downloaded": 0,
            "snapshot_error": None,
            "gemini_result": None,
            "final_label": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "error_message": None
        }
        
        await self.db.insert_event(event_data)
        return event_id
    
    async def update_event_with_gemini_result(
        self,
        event_id: str,
        gemini_result: Optional[Dict[str, Any]],
        error_message: Optional[str] = None
    ) -> str:
        """
        Gemini 분석 결과로 이벤트 업데이트
        
        Args:
            event_id: Frigate 이벤트 ID
            gemini_result: Gemini 분석 결과 (None이면 에러)
            error_message: 에러 메시지 (있는 경우)
            
        Returns:
            새로운 상태
        """
        event = await self.db.get_event(event_id)
        if not event:
            logger.error(f"이벤트를 찾을 수 없음: {event_id}")
            return None
        
        updates = {"updated_at": datetime.now().isoformat()}
        
        if gemini_result is None:
            # Gemini 에러
            updates['status'] = EventStatus.GEMINI_ERROR.value
            updates['error_message'] = error_message
        else:
            updates['gemini_result'] = gemini_result
            gemini_label = self._normalize_label(gemini_result.get('primary_class', 'unknown'))
            frigate_label = self._normalize_label(event['frigate_label'])
            
            if frigate_label == gemini_label:
                # Frigate와 Gemini 일치
                updates['status'] = EventStatus.CLASSIFIED.value
                updates['final_label'] = gemini_label
                await self._save_yolo_label(event_id, gemini_label, event)
                
                # Synthetic 데이터 생성 시도 (10% 확률)
                if not event.get('is_synthetic'):  # 원본 이벤트만
                    try:
                        await self.create_synthetic_sample(event)
                    except Exception as e:
                        logger.error(f"[{event_id}] Synthetic 생성 오류: {e}")
            else:
                # 불일치 - 수동 라벨링 필요
                updates['status'] = EventStatus.MISMATCHED.value
                updates['error_message'] = f"Frigate: {frigate_label}, Gemini: {gemini_label}"
        
        await self.db.update_event(event_id, updates)
        
        logger.info(f"이벤트 업데이트: {event_id} -> {updates['status']}")
        return updates['status']
    
    async def manual_label_event(self, event_id: str, label: Optional[str]) -> bool:
        """
        수동으로 이벤트 라벨 지정
        
        Args:
            event_id: 이벤트 ID
            label: 라벨 (person, cat, None)
            
        Returns:
            성공 여부
        """
        event = await self.db.get_event(event_id)
        if not event:
            logger.error(f"이벤트를 찾을 수 없음: {event_id}")
            return False
        
        # legacy: null/None 입력은 background로 저장
        if label is None:
            label = "background"

        if label not in self.CLASS_NAMES:
            logger.error(f"잘못된 라벨: {label}")
            return False
        
        # YOLO 라벨 파일 생성
        await self._save_yolo_label(event_id, label, event)
        
        # DB 업데이트
        updates = {
            'final_label': label,
            'status': EventStatus.MANUAL_LABELED.value
        }
        
        await self.db.update_event(event_id, updates)
        
        # Synthetic 데이터 생성 시도 (10% 확률)
        if not event.get('is_synthetic') and label in ("person", "cat"):  # 원본 이벤트만, background 제외
            try:
                await self.create_synthetic_sample(event)
            except Exception as e:
                logger.error(f"[{event_id}] Synthetic 생성 오류: {e}")
        
        logger.info(f"수동 라벨링 완료: {event_id} -> {label}")
        return True
    
    async def _save_yolo_label(self, event_id: str, label: Optional[str], event: Dict):
        """YOLO 형식 라벨 파일 저장"""
        # 이미지 파일명 추출
        image_path = Path(event['image_path'])
        label_filename = image_path.stem + ".txt"
        label_path = self.labels_dir / label_filename
        
        # background(또는 legacy None)면 빈 파일 생성
        if label is None or label == "background":
            with open(label_path, 'w') as f:
                pass  # 빈 파일
            logger.info(f"YOLO 라벨 저장 (background): {label_path}")
            return
        
        # is_bound_box가 false면 라벨 생성 안 함
        if not event.get('is_bound_box'):
            logger.warning(f"[{event_id}] bound_box가 없어 라벨 생성 생략")
            return
        
        # bound_box 사용
        bound_box_data = event.get('bound_box')
        if not bound_box_data:
            logger.error(f"[{event_id}] bound_box 데이터 없음")
            return
        
        try:
            # bound_box가 이미 list면 그대로, string이면 파싱
            if isinstance(bound_box_data, str):
                bound_box = json.loads(bound_box_data)
            else:
                bound_box = bound_box_data
                
            class_id = self.CLASS_MAPPING.get(label)
            
            if class_id is None:
                logger.error(f"[{event_id}] 잘못된 라벨: {label}")
                return
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {bound_box[0]:.6f} {bound_box[1]:.6f} {bound_box[2]:.6f} {bound_box[3]:.6f}\n")
            
            logger.info(f"YOLO 라벨 저장: {label_path}")
        except Exception as e:
            logger.error(f"[{event_id}] 라벨 저장 실패: {e}")
    
    def _extract_bbox(self, frigate_data: Dict) -> Optional[Dict]:
        """Frigate 데이터에서 YOLO 형식 바운딩 박스 추출 (레거시)"""
        # 레거시 메서드: 이제 _get_yolo_bound_box 사용 권장
        box = None
        
        # 먼저 data.box 확인 (실제 위치)
        if 'data' in frigate_data and isinstance(frigate_data['data'], dict):
            box = frigate_data['data'].get('box')
        
        # 없으면 최상위 box 확인 (이전 버전 호환)
        if box is None and 'box' in frigate_data:
            box = frigate_data['box']
        
        if box is None or len(box) != 4:
            return None
        
        # Frigate의 box 형식: [x_center, y_center, width, height] (이미 정규화됨)
        x_center, y_center, width, height = box
        
        return {
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        }
    
    def _get_yolo_bound_box(self, box: list, frame: tuple, camera: str, is_cropped: bool = True) -> Optional[tuple]:
        """
        Crop된 스냅샷에서 YOLO 형식 좌표 반환.
        
        Args:
            box: [x, y, w, h] - Event.data["box"] (0~1)
            frame: (width, height) - detect 크기 (스냅샷 실제 크기)
            camera: 카메라 이름
            is_cropped: crop 여부
        
        Returns:
            (x_center, y_center, width, height) - 0~1, 실패 시 None
        """
        if not box or len(box) != 4:
            logger.error(f"Invalid box format: {box}")
            return None
        
        if not is_cropped:
            # crop 안 된 경우, 원본 box 그대로 사용
            x_center = box[0] + box[2] / 2
            y_center = box[1] + box[3] / 2
            return (x_center, y_center, box[2], box[3])
        
        # Camera별 원본 프레임 크기
        if camera == "main_entrance":
            fw, fh = 640, 360
        elif camera == "backyard":
            fw, fh = 640, 360
        elif camera == "garden":
            fw, fh = 960, 540
        else:
            logger.error(f"Unknown camera: {camera}")
            return None
        
        # 0~1 scale → 픽셀 좌표
        x_min = int(box[0] * fw)
        y_min = int(box[1] * fh)
        w = int(box[2] * fw)
        h = int(box[3] * fh)
        
        # Crop 영역 계산 (Frigate: box_size=300, multiplier=1.1)
        size = max(300, int(1.1 * max(w, h)))
        size = (size // 4) * 4
        
        # 실제 스냅샷 크기와 비교
        x_size, y_size = frame
        if not (x_size == y_size == size):
            logger.error(f"Size mismatch: expected {size}x{size}, got {x_size}x{y_size}")
            return None
        
        cx, cy = x_min + w // 2, y_min + h // 2
        
        # Crop 시작 위치
        x1 = cx - size // 2
        y1 = cy - size // 2
        
        # Offset 보정: crop 영역이 프레임을 벗어나지 않도록
        if x1 < 0:
            x1 = 0
        elif x1 > fw - size:
            x1 = fw - size
        
        if y1 < 0:
            y1 = 0
        elif y1 > fh - size:
            y1 = fh - size
        
        # YOLO 정규화 좌표 (0~1)
        x_center = (x_min - x1 + w / 2) / size
        y_center = (y_min - y1 + h / 2) / size
        width = w / size
        height = h / size
        
        return (x_center, y_center, width, height)
    
    def _normalize_label(self, label: str) -> Optional[str]:
        """Frigate 라벨을 표준 클래스로 정규화"""
        label_lower = label.lower()
        if label_lower == "person":
            return "person"
        elif label_lower == "cat":
            return "cat"
        else:
            return None  # others → null
    
    async def get_stats(self) -> Dict:
        """통계 반환"""
        return await self.db.get_stats()
    
    async def get_events_by_status(self, status: str) -> List[Dict]:
        """특정 상태의 이벤트 목록 반환"""
        return await self.db.get_events_by_status(status)
    
    async def get_pending_gemini_events(self) -> List[Dict]:
        """Gemini 분석이 필요한 이벤트 목록"""
        return await self.db.get_pending_gemini_events()
    
    async def get_mismatched_events(self) -> List[Dict]:
        """수동 라벨링이 필요한 이벤트 목록"""
        return await self.get_events_by_status(EventStatus.MISMATCHED.value)
    
    async def get_classified_events(self) -> List[Dict]:
        """분류 완료된 이벤트 목록"""
        classified = await self.get_events_by_status(EventStatus.CLASSIFIED.value)
        manual = await self.get_events_by_status(EventStatus.MANUAL_LABELED.value)
        return classified + manual
    
    async def get_event(self, event_id: str) -> Optional[Dict]:
        """특정 이벤트 정보 반환"""
        return await self.db.get_event(event_id)
    
    async def delete_event(self, event_id: str) -> bool:
        """이벤트 삭제"""
        event = await self.db.get_event(event_id)
        if not event:
            return False
        
        # 이미지 파일 삭제
        image_path = Path(event.get('image_path', ''))
        if image_path.exists():
            image_path.unlink()
        
        # 라벨 파일 삭제
        label_path = self.labels_dir / (image_path.stem + ".txt")
        if label_path.exists():
            label_path.unlink()
        
        await self.db.delete_event(event_id)
        
        logger.info(f"이벤트 삭제: {event_id}")
        return True
    
    async def export_yolo_dataset(self, output_dir: Optional[Path] = None) -> Path:
        """
        YOLO NAS 학습용 데이터셋 내보내기
        
        Export 로직:
        - person 50%, cat 50% (더 적은 개수 기준)
        - null: 총 데이터의 10% 이내 (있는 만큼만)
        - 라벨링(라벨 파일 생성)이 안 된 이벤트는 Export 대상에서 제외
        - 선별 우선순위: manual_labeled > garden 카메라 > 나머지
        - is_bound_box=False인 이벤트는 제외
        
        Args:
            output_dir: 출력 디렉토리 (기본: data/yolo_export)
            
        Returns:
            데이터셋 경로
        """
        if output_dir is None:
            output_dir = settings.data_dir / "yolo_export"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # train/val 디렉토리 구조 생성
        train_images = output_dir / "train" / "images"
        train_labels = output_dir / "train" / "labels"
        val_images = output_dir / "val" / "images"
        val_labels = output_dir / "val" / "labels"
        
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Export 대상: 분류 완료된 이벤트만 (classified + manual_labeled)
        classified_events = await self.get_classified_events()

        # 중복 제거
        all_events = {e['event_id']: e for e in classified_events}

        def has_yolo_label_file(event: Dict[str, Any]) -> bool:
            image_path = event.get('image_path')
            if not image_path:
                return False
            src_img = Path(image_path)
            label_path = self.labels_dir / (src_img.stem + ".txt")
            return label_path.exists()

        # is_bound_box=False 및 라벨링 안 된 이벤트 필터링
        # - get_classified_events()만 사용하므로 mismatched/pending/gemini_error는 원칙적으로 제외되지만,
        #   데이터 정합성 문제를 대비해 안전장치를 둔다.
        # - background(negative sample)는 bound_box가 없어도(0이어도) 허용한다.
        valid_events = []
        for e in all_events.values():
            # 1. Status check
            if e.get('status') == EventStatus.MISMATCHED.value:
                continue
            
            # 2. Label file check
            if not has_yolo_label_file(e):
                continue
            
            # 3. Bounding box check
            label = e.get('final_label')
            if label in ['person', 'cat']:
                # Positive samples must have a bounding box
                if e.get('is_bound_box', 0) != 1:
                    continue
            
            valid_events.append(e)
        
        logger.info(f"Export 대상: 전체 {len(all_events)}개 중 유효한 이벤트 {len(valid_events)}개")
        
        # 클래스별로 분류
        person_events = [e for e in valid_events if e.get('final_label') == 'person']
        cat_events = [e for e in valid_events if e.get('final_label') == 'cat']
        background_events = [e for e in valid_events if e.get('final_label') in (None, 'background')]

        logger.info(
            f"클래스별 개수: person={len(person_events)}, cat={len(cat_events)}, background={len(background_events)}"
        )
        
        # person 50%, cat 50%로 더 적은 개수 기준
        base_count = min(len(person_events), len(cat_events))
        
        # background는 총 데이터의 10% 이내 (있는 만큼만)
        background_count = min(int(base_count * 2 * 0.1), len(background_events))
        
        logger.info(f"Export 목표: person={base_count}, cat={base_count}, background={background_count}")
        
        def get_confidence(event):
            """이벤트의 Frigate confidence 추출"""
            try:
                fd = event.get('frigate_data', {})
                # database.py에서 이미 dict로 파싱되었을 것임
                if isinstance(fd, str):
                    fd = json.loads(fd)
                return float(fd.get('data', {}).get('score', 0.0))
            except:
                return 0.0

        def select_events_weighted(events, target_count):
            """
            가중치 랜덤 선별 로직
            Returns: List of (event, reason, score)
            """
            # Tier 0: Manual Labeled
            manual_events = [e for e in events if e.get('status') == EventStatus.MANUAL_LABELED.value]
            
            # Tier 1 & 2: Others
            other_events = [e for e in events if e.get('status') != EventStatus.MANUAL_LABELED.value]
            
            # Manual은 무조건 포함
            selected_with_info = []
            for e in manual_events:
                selected_with_info.append({
                    'event': e,
                    'reason': 'Manual Labeled (Tier 0)',
                    'score': 999.0  # Highest priority
                })
            
            # 부족분 계산
            needed = target_count - len(selected_with_info)
            
            if needed > 0 and other_events:
                # 점수 계산 및 정렬
                scored_events = []
                for e in other_events:
                    conf = get_confidence(e)
                    
                    camera = ''
                    fd = e.get('frigate_data')
                    if isinstance(fd, dict):
                        camera = fd.get('camera', '')
                    
                    is_garden = (camera == 'garden')
                    
                    # 점수 공식: (1.0 - Confidence) + Garden Bonus + Random Noise
                    # Confidence가 낮을수록(어려운 데이터) 점수가 높음
                    score = (1.0 - conf) + (0.5 if is_garden else 0) + random.uniform(0, 0.3)
                    scored_events.append((score, e))
                
                # 점수 높은 순 정렬
                scored_events.sort(key=lambda x: x[0], reverse=True)
                
                # 상위 needed 개수만큼 선택
                for score, e in scored_events[:needed]:
                    selected_with_info.append({
                        'event': e,
                        'reason': 'Weighted Selection (Tier 1/2)',
                        'score': score
                    })
            
            return selected_with_info

        def select_background_random(events, target_count):
            """
            Background 선별 로직
            Returns: List of (event, reason, score)
            """
            # 전체 셔플
            candidates = list(events)
            random.shuffle(candidates)
            
            selected_with_info = []
            for e in candidates[:target_count]:
                selected_with_info.append({
                    'event': e,
                    'reason': 'Random Background',
                    'score': 0.0
                })
            
            return selected_with_info

        # 각 클래스별 선별 로직 적용
        sampled_person_info = select_events_weighted(person_events, base_count)
        sampled_cat_info = select_events_weighted(cat_events, base_count)
        sampled_background_info = select_background_random(background_events, background_count)
        
        logger.info(
            f"실제 Export: person={len(sampled_person_info)}, cat={len(sampled_cat_info)}, background={len(sampled_background_info)}"
        )
        
        # 합치기 및 섞기
        all_selected_info = sampled_person_info + sampled_cat_info + sampled_background_info
        random.shuffle(all_selected_info)
        
        # 80% train, 20% val
        split_idx = int(len(all_selected_info) * 0.8)
        train_info = all_selected_info[:split_idx]
        val_info = all_selected_info[split_idx:]
        
        def copy_event_files(info_list: List[Dict], img_dir: Path, lbl_dir: Path) -> int:
            """이벤트 파일 복사 (성공한 개수 반환)"""
            count = 0
            for item in info_list:
                event = item['event']
                src_img = Path(event['image_path'])
                if not src_img.exists():
                    logger.warning(f"이미지 없음: {src_img}")
                    continue
                
                dst_img = img_dir / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 복사 (없어도 OK - null label일 수 있음)
                src_lbl = self.labels_dir / (src_img.stem + ".txt")
                if src_lbl.exists():
                    dst_lbl = lbl_dir / src_lbl.name
                    shutil.copy2(src_lbl, dst_lbl)
                
                count += 1
            return count
        
        train_count = copy_event_files(train_info, train_images, train_labels)
        val_count = copy_event_files(val_info, val_images, val_labels)
        
        # Summary 파일 생성
        summary_path = output_dir / "summary.md"

        def _parse_camera(event: Dict[str, Any]) -> str:
            camera = ''
            try:
                fd = event.get('frigate_data')
                if isinstance(fd, str):
                    fd = json.loads(fd)
                if isinstance(fd, dict):
                    camera = fd.get('camera', '') or ''
            except Exception:
                camera = ''
            return camera

        def _parse_event_dt(event: Dict[str, Any]) -> Optional[datetime]:
            # Many event_id values look like: "1767101046.219933-5isoy0" (timestamp + suffix)
            # Synthetic IDs can look like: "synthetic_1767250984.864521-xabk31_1767195076"
            # We extract the first numeric timestamp-like token we can find.
            for key in ('source_event_id', 'event_id'):
                raw = event.get(key)
                if raw is None:
                    continue

                # Fast path: already numeric
                if isinstance(raw, (int, float)):
                    try:
                        return datetime.fromtimestamp(float(raw))
                    except Exception:
                        pass

                s = str(raw)
                m = re.search(r'(\d+(?:\.\d+)?)', s)
                if not m:
                    continue
                try:
                    ts = float(m.group(1))
                    return datetime.fromtimestamp(ts)
                except Exception:
                    continue

            return None

        def _time_bucket_kor(dt: Optional[datetime]) -> str:
            if dt is None:
                return 'Unknown'
            hour = dt.hour
            if 0 <= hour < 6:
                return '새벽(00-06)'
            if 6 <= hour < 12:
                return '오전(06-12)'
            if 12 <= hour < 18:
                return '오후(12-18)'
            return '저녁(18-24)'

        def _norm_label(event: Dict[str, Any]) -> str:
            label = event.get('final_label', 'background') or 'background'
            if label not in ('person', 'cat', 'background'):
                return 'background'
            return label

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# YOLO Dataset Export Summary\n\n")
            f.write(f"- **Date**: {datetime.now().isoformat()}\n")
            f.write(f"- **Total Exported**: {len(all_selected_info)}\n")
            f.write(f"- **Train**: {train_count}\n")
            f.write(f"- **Val**: {val_count}\n\n")
            
            f.write("## Class Distribution\n")
            f.write(f"- **Person**: {len(sampled_person_info)}\n")
            f.write(f"- **Cat**: {len(sampled_cat_info)}\n")
            f.write(f"- **Background**: {len(sampled_background_info)}\n\n")

            # Camera / Time-of-day breakdowns
            camera_totals: Dict[str, int] = defaultdict(int)
            camera_by_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            bucket_totals: Dict[str, int] = defaultdict(int)
            bucket_by_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for item in all_selected_info:
                event = item['event']
                label = _norm_label(event)
                camera = _parse_camera(event) or 'Unknown'
                dt = _parse_event_dt(event)
                bucket = _time_bucket_kor(dt)

                camera_totals[camera] += 1
                camera_by_label[camera][label] += 1
                bucket_totals[bucket] += 1
                bucket_by_label[bucket][label] += 1

            f.write("## Camera Breakdown\n")
            f.write("| Camera | Total | Person | Cat | Background |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for camera, total in sorted(camera_totals.items(), key=lambda x: (-x[1], x[0])):
                p = camera_by_label[camera].get('person', 0)
                c = camera_by_label[camera].get('cat', 0)
                b = camera_by_label[camera].get('background', 0)
                f.write(f"| {camera} | {total} | {p} | {c} | {b} |\n")
            f.write("\n")

            f.write("## Time-of-Day Breakdown\n")
            f.write("| Time Bucket | Total | Person | Cat | Background |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            bucket_order = ['새벽(00-06)', '오전(06-12)', '오후(12-18)', '저녁(18-24)', 'Unknown']
            for bucket in bucket_order:
                if bucket not in bucket_totals:
                    continue
                total = bucket_totals[bucket]
                p = bucket_by_label[bucket].get('person', 0)
                c = bucket_by_label[bucket].get('cat', 0)
                b = bucket_by_label[bucket].get('background', 0)
                f.write(f"| {bucket} | {total} | {p} | {c} | {b} |\n")
            f.write("\n")
            
            f.write("## Selection Details\n")
            f.write("| Class | Event ID | Camera | Reason | Score | Set |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            # Helper to find set
            train_ids = {item['event']['event_id'] for item in train_info}
            
            # Sort by class then score
            all_selected_info.sort(key=lambda x: (x['event'].get('final_label', 'background') or 'background', -x['score']))
            
            for item in all_selected_info:
                event = item['event']
                label = event.get('final_label', 'background') or 'background'
                eid = event['event_id']

                camera = _parse_camera(event)

                reason = item['reason']
                score = f"{item['score']:.4f}"
                dataset_type = "Train" if eid in train_ids else "Val"
                
                f.write(f"| {label} | {eid} | {camera} | {reason} | {score} | {dataset_type} |\n")
        
        logger.info(f"Summary generated: {summary_path}")
        
        # YOLO 데이터셋 설정 파일 생성
        config = {
            "path": str(output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {
                0: "person",
                1: "cat"
            }
        }
        
        config_path = output_dir / "dataset.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"YOLO 데이터셋 내보내기 완료: {output_dir}")
        logger.info(f"  - Train: {train_count} images")
        logger.info(f"  - Val: {val_count} images")
        
        return output_dir
    
    async def recalculate_bound_boxes(self, limit: int = 100) -> Dict[str, int]:
        """
        기존 이벤트들의 bound_box 재계산 (마이그레이션용)
        
        Args:
            limit: 한 번에 처리할 이벤트 수
            
        Returns:
            {"success": 성공 개수, "failed": 실패 개수, "total": 전체 개수}
        """
        from PIL import Image
        
        events = await self.db.get_events_without_bound_box(limit)
        
        if not events:
            logger.info("bound_box 재계산이 필요한 이벤트 없음")
            return {"success": 0, "failed": 0, "total": 0}
        
        success_count = 0
        failed_count = 0
        
        for event in events:
            event_id = event['event_id']
            
            try:
                image_path = Path(event.get('image_path', ''))
                if not image_path.exists():
                    logger.warning(f"[{event_id}] 이미지 파일 없음: {image_path}")
                    failed_count += 1
                    continue
                
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
                            yolo_box = self._get_yolo_bound_box(
                                box, frame_size, camera, is_cropped=True
                            )
                            
                            if yolo_box:
                                bound_box = json.dumps(yolo_box)
                                is_bound_box = 1
                                
                                await self.db.update_event(event_id, {
                                    'bound_box': bound_box,
                                    'is_bound_box': is_bound_box,
                                    'updated_at': datetime.now().isoformat()
                                })
                                
                                success_count += 1
                                logger.info(f"[{event_id}] bound_box 재계산 성공")
                            else:
                                failed_count += 1
                                logger.warning(f"[{event_id}] bound_box 계산 실패")
                        else:
                            failed_count += 1
                            logger.warning(f"[{event_id}] box 데이터 없음")
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"[{event_id}] bound_box 재계산 중 오류: {e}")
                failed_count += 1
        
        result = {
            "success": success_count,
            "failed": failed_count,
            "total": len(events)
        }
        
        logger.info(f"bound_box 재계산 완료: {result}")
        return result
    
    async def create_synthetic_sample(self, source_event: Dict) -> Optional[str]:
        """
        라벨링 완료된 이벤트를 기반으로 synthetic negative sample 생성
        
        같은 카메라에서 ±1일 내 랜덤 시간의 프레임을 가져와 
        원본과 동일한 crop 영역으로 잘라 저장
        
        Args:
            source_event: 원본 이벤트 (라벨링 완료된 이벤트)
            
        Returns:
            생성된 synthetic 이벤트 ID 또는 None
        """
        import random
        from PIL import Image
        from datetime import timedelta
        from src.api.frigate_api import FrigateAPI
        
        try:
            # 10% 확률 체크
            if random.random() > self.synthetic_probability:
                return None
            
            # 원본 이벤트 데이터 추출
            source_event_id = source_event['event_id']
            frigate_data = source_event.get('frigate_data')
            
            if not isinstance(frigate_data, dict):
                logger.warning(f"[{source_event_id}] frigate_data 없음")
                return None
            
            camera = frigate_data.get('camera')
            if not camera:
                logger.warning(f"[{source_event_id}] camera 정보 없음")
                return None
            
            # 원본 이벤트의 timestamp (created_at 사용)
            created_at_str = source_event.get('created_at')
            if not created_at_str:
                return None
            
            source_time = datetime.fromisoformat(created_at_str)
            
            # ±1일 내 랜덤 시간 생성
            random_offset = timedelta(
                seconds=random.uniform(-86400, 86400)  # ±24시간
            )
            target_time = source_time + random_offset
            target_timestamp = target_time.timestamp()
            
            logger.info(f"[{source_event_id}] Synthetic 생성 시도: {camera} @ {target_time}")
            
            # Frigate에서 해당 시간의 스냅샷 가져오기
            frigate_api = FrigateAPI()
            
            # 임시 파일로 저장
            temp_filename = f"temp_synthetic_{source_event_id}_{int(target_timestamp)}.jpg"
            temp_path = self.synthetic_dir / temp_filename
            
            image_data = frigate_api.get_snapshot_at_time(
                camera, target_timestamp, temp_path
            )
            
            if not image_data:
                logger.warning(f"[{source_event_id}] 스냅샷 가져오기 실패")
                return None
            
            # Crop 영역 계산 (원본 이벤트와 동일)
            data_obj = frigate_data.get('data', {})
            box = data_obj.get('box')
            
            if not box or len(box) != 4:
                logger.warning(f"[{source_event_id}] box 데이터 없음")
                temp_path.unlink(missing_ok=True)
                return None
            
            # Camera별 원본 프레임 크기
            if camera == "main_entrance":
                fw, fh = 640, 360
            elif camera == "backyard":
                fw, fh = 640, 360
            elif camera == "garden":
                fw, fh = 960, 540
            else:
                logger.error(f"Unknown camera: {camera}")
                temp_path.unlink(missing_ok=True)
                return None
            
            # Crop 영역 계산
            x_min = int(box[0] * fw)
            y_min = int(box[1] * fh)
            w = int(box[2] * fw)
            h = int(box[3] * fh)
            
            size = max(300, int(1.1 * max(w, h)))
            size = (size // 4) * 4
            
            cx, cy = x_min + w // 2, y_min + h // 2
            x1 = cx - size // 2
            y1 = cy - size // 2
            
            # Offset 보정
            if x1 < 0:
                x1 = 0
            elif x1 > fw - size:
                x1 = fw - size
            
            if y1 < 0:
                y1 = 0
            elif y1 > fh - size:
                y1 = fh - size
            
            # 이미지 crop
            with Image.open(temp_path) as img:
                # 이미지 크기 확인
                if img.width != fw or img.height != fh:
                    logger.warning(
                        f"[{source_event_id}] 이미지 크기 불일치: "
                        f"예상 {fw}x{fh}, 실제 {img.width}x{img.height}"
                    )
                    # 크기가 다르면 비율로 조정
                    x1 = int(x1 * img.width / fw)
                    y1 = int(y1 * img.height / fh)
                    size = int(size * min(img.width / fw, img.height / fh))
                
                cropped = img.crop((x1, y1, x1 + size, y1 + size))
                
                # 최종 파일명 생성
                synthetic_id = f"synthetic_{source_event_id}_{int(target_timestamp)}"
                final_filename = f"{synthetic_id}.jpg"
                final_path = self.synthetic_dir / final_filename
                
                cropped.save(final_path, quality=95)
            
            # 임시 파일 삭제
            temp_path.unlink(missing_ok=True)
            
            # DB에 등록 (pending 상태로)
            synthetic_event_data = {
                "event_id": synthetic_id,
                "status": EventStatus.PENDING.value,
                "frigate_label": "synthetic",
                "frigate_data": {
                    "camera": camera,
                    "timestamp": target_timestamp,
                    "source_event_id": source_event_id,
                    "crop_region": [x1, y1, size]
                },
                "image_path": str(final_path),
                "snapshot_downloaded": 1,
                "snapshot_error": None,
                "gemini_result": None,
                "final_label": None,
                "bound_box": None,  # Synthetic은 라벨링 후 계산
                "is_bound_box": 0,
                "is_synthetic": 1,
                "source_event_id": source_event_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "error_message": None
            }
            
            await self.db.insert_event(synthetic_event_data)
            
            logger.info(
                f"[{source_event_id}] Synthetic 생성 완료: {synthetic_id} "
                f"(카메라: {camera}, 시간: {target_time})"
            )
            
            return synthetic_id
            
        except Exception as e:
            logger.error(f"[{source_event_id}] Synthetic 생성 실패: {e}", exc_info=True)
            return None


    async def regenerate_all_labels(self) -> Dict[str, int]:
        """
        모든 분류된 이벤트의 라벨 파일 재생성
        
        Returns:
            {"success": 성공 개수, "skipped": 스킵 개수, "failed": 실패 개수}
        """
        logger.info("라벨 재생성 시작")
        
        # 모든 분류된 이벤트 가져오기
        classified_events = await self.get_classified_events()
        mismatched_events = await self.get_mismatched_events()
        
        # 중복 제거
        all_events = {e['event_id']: e for e in classified_events + mismatched_events}
        
        stats = {"success": 0, "skipped": 0, "failed": 0}
        
        for event in all_events.values():
            event_id = event['event_id']
            label = event.get('final_label')
            
            try:
                # is_bound_box=False이고 null이 아니면 스킵
                if not event.get('is_bound_box') and label is not None:
                    logger.debug(f"[{event_id}] bound_box 없음, 스킵")
                    stats["skipped"] += 1
                    continue
                
                # 라벨 파일 생성
                await self._save_yolo_label(event_id, label, event)
                stats["success"] += 1
                logger.info(f"[{event_id}] 라벨 파일 생성 완료: {label}")
                
            except Exception as e:
                logger.error(f"[{event_id}] 라벨 재생성 실패: {e}", exc_info=True)
                stats["failed"] += 1
        
        logger.info(
            f"라벨 재생성 완료: 성공 {stats['success']}, "
            f"스킵 {stats['skipped']}, 실패 {stats['failed']}"
        )
        
        return stats

