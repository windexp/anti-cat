"""Gemini API 객체 인식 서비스"""
import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import re
from google import genai
from google.genai import types
from PIL import Image
import json
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 외부 라이브러리(google-genai)의 과도한 INFO 로그를 줄임
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


class GeminiService:
    """Gemini API를 사용한 이미지 객체 인식 서비스 (멀티 API 키 지원)"""
    
    # 분류 클래스
    CLASSES = ["person", "cat", "others"]
    
    def __init__(self, db=None):
        """Gemini API 초기화"""
        self.api_keys = settings.gemini_api_key_list
        # 기본 모델 리스트는 env 기반. DB에서 선택된 모델이 있으면 런타임에 덮어씀.
        # Gemini 2.0 모델 제거 (더 이상 지원되지 않음)
        self.models = [m for m in settings.gemini_model_list if not m.startswith('gemini-2.0')]
        self.current_key_index = 0
        self.current_model_index = 0
        self.db = db  # Database 인스턴스
        self.client = None
        self._last_request_time = 0
        self._min_request_interval = 5.0  # 최소 요청 간격 (초) - 동적으로 변경됨

        self._model_refresh_kv_key = "gemini_models_last_refresh_at"
        
        # 초기 클라이언트 생성
        if self.api_keys:
            self._init_client(0)
        
        logger.info(f"Gemini 서비스 초기화: {len(self.api_keys)}개 API 키, {len(self.models)}개 모델")
    
    def _init_client(self, key_index: int, model_index: int = 0):
        """특정 인덱스의 API 키와 모델로 클라이언트 초기화"""
        if 0 <= key_index < len(self.api_keys):
            self.current_key_index = key_index
            self.current_model_index = model_index
            self.client = genai.Client(api_key=self.api_keys[key_index])
            logger.warning(
                f"Gemini 설정: API 키 [{key_index}/{len(self.api_keys)}], "
                f"모델 [{model_index}/{len(self.models)}] {self.models[model_index]}"
            )
    
    def _get_current_model(self) -> str:
        """현재 사용 중인 모델 이름 반환"""
        if 0 <= self.current_model_index < len(self.models):
            return self.models[self.current_model_index]
        return self.models[0] if self.models else 'gemini-2.5-flash-lite'
    
    def _get_model_rate_limit(self, model_name: str) -> float:
        """모델별 최소 요청 간격 반환 (초)
        
        Args:
            model_name: 모델 이름
            
        Returns:
            최소 요청 간격 (초)
            - gemini-2.5-flash-lite: 6초 (분당 10회)
            - gemini-3-flash-preview: 12초 (분당 5회)
            - 기타: 5초 (기본값)
        """
        model_lower = model_name.lower()
        
        # 2.5 flash-lite: 분당 10회 제한
        if '2.5' in model_lower and 'flash-lite' in model_lower:
            return 6.0
        
        # 3 flash-preview: 분당 5회 제한
        if ('3-flash-preview' in model_lower or 
            'gemini-3' in model_lower and 'preview' in model_lower):
            return 12.0
        
        # 기본값
        return 5.0
    
    def switch_to_next_model(self) -> bool:
        """
        다음 모델로 전환 (같은 API 키 내에서)
        
        Returns:
            True: 전환 성공
            False: 더 이상 사용 가능한 모델 없음 (다음 API 키로 이동 필요)
        """
        next_model_index = self.current_model_index + 1
        
        if next_model_index < len(self.models):
            # 같은 키, 다음 모델로 전환
            self.current_model_index = next_model_index
            logger.info(
                f"다음 모델로 전환: [{next_model_index}/{len(self.models)}] {self.models[next_model_index]}"
            )
            return True
        else:
            # 모든 모델 소진
            logger.info(f"API 키 {self.current_key_index}의 모든 모델 소진")
            return False
    
    def switch_to_next_api_key(self) -> bool:
        """
        다음 API 키로 전환 (모델 인덱스는 0으로 리셋)
        
        Returns:
            True: 전환 성공
            False: 더 이상 사용 가능한 키 없음
        """
        next_key_index = self.current_key_index + 1
        
        if next_key_index < len(self.api_keys):
            # 다음 키로 전환, 모델은 처음부터
            self._init_client(next_key_index, model_index=0)
            logger.info(
                f"다음 API 키로 전환: [{next_key_index}/{len(self.api_keys)}], 모델 리셋"
            )
            return True
        else:
            # 모든 API 키 소진
            logger.error("모든 Gemini API 키 소진됨")
            return False
    
    def _extract_retry_delay(self, error_msg: str) -> Optional[float]:
        """429 에러에서 retry delay 추출"""
        import re
        # "Please retry in 57.822800477s" 형식 파싱
        match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_msg)
        if match:
            return float(match.group(1))
        return None

    def _is_model_not_found_error(self, error_msg: str) -> bool:
        """모델이 존재하지 않거나 generateContent 미지원인 경우인지 판단"""
        markers = [
            '404',
            'NOT_FOUND',
            'is not found',
            'not found for API version',
            'not supported for generateContent',
            'Call ListModels',
        ]
        return any(m in error_msg for m in markers)

    # === Model registry / selection (monthly refresh) ===
    def _strip_models_prefix(self, model_name: str) -> str:
        return model_name.split('models/', 1)[1] if model_name.startswith('models/') else model_name

    def _parse_family(self, model_name: str) -> Optional[str]:
        """gemini-x.y-* 형태에서 x.y 추출. (ex: gemini-2.5-... -> '2.5', gemini-3-... -> '3')"""
        name = self._strip_models_prefix(model_name)
        m = re.match(r'^gemini-(\d+(?:\.\d+)?)\b', name)
        return m.group(1) if m else None

    def _is_candidate_for_multimodal(self, model_name: str) -> bool:
        """멀티모달(이미지+텍스트) generateContent 후보인지 (이름 기반)"""
        name = model_name.lower()
        if 'gemini' not in name:
            return False
        exclude_markers = (
            'embedding',
            'native-audio',
            'tts',
            'robotics',
            'computer-use',
        )
        return not any(x in name for x in exclude_markers)

    def _candidate_rank(self, model_name: str) -> tuple:
        """family 내부 우선순위 (낮을수록 우선)."""
        n = self._strip_models_prefix(model_name).lower()
        unstable = ('preview' in n) or ('exp' in n)

        # 비용/속도 우선: flash-lite > flash > pro
        tier = 3
        if 'flash-lite' in n:
            tier = 0
        elif 'flash' in n:
            tier = 1
        elif 'pro' in n:
            tier = 2

        # 고정 버전 선호 (예: -001)
        fixed = 0 if re.search(r'-\d{3}\b', n) else 1

        return (unstable, tier, fixed, n)

    async def load_selected_models_from_db(self) -> bool:
        """DB에 저장된 family별 대표 모델을 로드하여 self.models에 반영"""
        if not self.db:
            return False

        try:
            selected = await self.db.get_selected_gemini_models()
            # DB에는 env-friendly 이름(접두사 제거)을 저장하는 것을 전제로 함.
            selected = [m.strip() for m in selected if m and m.strip()]
            # Gemini 2.0 모델 제거 (더 이상 지원되지 않음)
            selected = [m for m in selected if not m.startswith('gemini-2.0')]
            if selected:
                self.models = selected
                self.current_model_index = 0
                logger.info(f"DB 선택 모델 로드: {len(self.models)}개 ({', '.join(self.models)})")
                return True
            return False
        except Exception as e:
            logger.warning(f"DB 선택 모델 로드 실패: {e}")
            return False

    def set_models(self, models: List[str]) -> None:
        """런타임 모델 리스트를 교체하고 인덱스를 리셋"""
        # Gemini 2.0 모델 제거 (더 이상 지원되지 않음)
        self.models = [m.strip() for m in models if m and m.strip() and not m.startswith('gemini-2.0')]
        self.current_model_index = 0

    async def refresh_models_if_needed(self, refresh_interval_days: int = 30, force: bool = False) -> bool:
        """30일 간격으로 모델 레지스트리 갱신 및 selection 업데이트

        Args:
            refresh_interval_days: 마지막 갱신 이후 최소 경과 일수
            force: True면 마지막 갱신 시각과 무관하게 즉시 갱신
        """
        if not self.db or not self.client:
            return False

        # 마지막 갱신 시각 확인
        try:
            last_str = await self.db.get_kv(self._model_refresh_kv_key)
            if last_str:
                try:
                    last_dt = datetime.fromisoformat(last_str)
                except ValueError:
                    last_dt = None
                if (not force) and last_dt and datetime.now() - last_dt < timedelta(days=refresh_interval_days):
                    # 갱신 불필요: DB selection만 로드
                    await self.load_selected_models_from_db()
                    return False
        except Exception as e:
            logger.warning(f"모델 갱신 시각 조회 실패(계속 진행): {e}")

        logger.info(
            "Gemini 모델 레지스트리 갱신 시작 (ListModels + 신규 멀티모달 프로빙)"
            + (" [force]" if force else "")
        )
        now_iso = datetime.now().isoformat()

        # 현재 DB 레지스트리 로드 (first_seen 유지용)
        existing: Dict[str, Dict[str, any]] = {}
        try:
            rows = await self.db.list_gemini_models()
            existing = {r['model_name']: r for r in rows}
        except Exception:
            existing = {}

        # ListModels
        try:
            visible_models = [getattr(m, 'name', str(m)) for m in self.client.models.list()]
        except Exception as e:
            logger.error(f"ListModels 실패: {e}")
            return False

        # 레지스트리 upsert
        for model_name in visible_models:
            family = self._parse_family(model_name)
            lowered = self._strip_models_prefix(model_name).lower()
            is_preview = 1 if 'preview' in lowered else 0
            is_exp = 1 if 'exp' in lowered else 0

            prev = existing.get(model_name)
            first_seen = prev.get('first_seen_at') if prev else now_iso
            multimodal_status = prev.get('multimodal_status') if prev else 'unknown'
            last_probed_at = prev.get('last_probed_at') if prev else None
            last_probe_error = prev.get('last_probe_error') if prev else None

            await self.db.upsert_gemini_model(
                model_name=model_name,
                family=family,
                is_preview=is_preview,
                is_exp=is_exp,
                multimodal_status=multimodal_status or 'unknown',
                first_seen_at=first_seen,
                last_seen_at=now_iso,
                last_probed_at=last_probed_at,
                last_probe_error=last_probe_error,
            )

        # family별로 "대표 후보 1개"만 신규/unknown이면 최소 프로빙 (쿼터 보호)
        # - 이미 multimodal=yes 후보가 있으면 프로빙 생략
        try:
            current_rows = await self.db.list_gemini_models()
        except Exception as e:
            logger.error(f"gemini_models 조회 실패: {e}")
            return False

        by_family: Dict[str, List[Dict[str, any]]] = {}
        for r in current_rows:
            mn = r.get('model_name')
            if not mn or not self._is_candidate_for_multimodal(mn):
                continue
            fam = r.get('family') or self._parse_family(mn)
            if not fam:
                continue
            by_family.setdefault(fam, []).append(r)

        # tiny image + minimal tokens
        img = Image.new('RGB', (16, 16), (0, 0, 0))
        prompt = 'Reply with only: OK'
        config = types.GenerateContentConfig(max_output_tokens=16, temperature=0.0)

        for fam, rows in by_family.items():
            # 이미 멀티모달 YES가 하나라도 있으면 그 family는 프로빙 생략
            if any((r.get('multimodal_status') == 'yes') for r in rows):
                continue

            # best candidate 1개만 프로빙
            rows_sorted = sorted(rows, key=lambda r: self._candidate_rank(r['model_name']))
            candidate = rows_sorted[0]
            if candidate.get('multimodal_status') not in (None, '', 'unknown'):
                continue

            model_name = candidate['model_name']
            try:
                await asyncio.sleep(1)  # 속도/쿼터 보호
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt],
                    config=config,
                )
                _ = (getattr(resp, 'text', '') or '').strip()
                await self.db.upsert_gemini_model(
                    model_name=model_name,
                    family=candidate.get('family') or fam,
                    is_preview=int(candidate.get('is_preview') or 0),
                    is_exp=int(candidate.get('is_exp') or 0),
                    multimodal_status='yes',
                    first_seen_at=candidate.get('first_seen_at') or now_iso,
                    last_seen_at=now_iso,
                    last_probed_at=now_iso,
                    last_probe_error=None,
                )
                logger.info(f"멀티모달 프로빙 OK: {model_name}")
            except Exception as e:
                msg = str(e)
                # 429는 unknown 유지 (다음 달 재시도)
                status = 'unknown' if ('429' in msg or 'RESOURCE_EXHAUSTED' in msg) else 'no'
                await self.db.upsert_gemini_model(
                    model_name=model_name,
                    family=candidate.get('family') or fam,
                    is_preview=int(candidate.get('is_preview') or 0),
                    is_exp=int(candidate.get('is_exp') or 0),
                    multimodal_status=status,
                    first_seen_at=candidate.get('first_seen_at') or now_iso,
                    last_seen_at=now_iso,
                    last_probed_at=now_iso,
                    last_probe_error=msg[:500],
                )
                logger.warning(f"멀티모달 프로빙 실패({status}): {model_name} :: {msg[:120]}")

        # selection 업데이트: family별로 multimodal=yes 중 최우선 1개 선택
        try:
            updated_rows = await self.db.list_gemini_models()
        except Exception as e:
            logger.error(f"selection용 gemini_models 조회 실패: {e}")
            return False

        by_family2: Dict[str, List[Dict[str, any]]] = {}
        for r in updated_rows:
            if r.get('multimodal_status') != 'yes':
                continue
            mn = r.get('model_name')
            if not mn or not self._is_candidate_for_multimodal(mn):
                continue
            fam = r.get('family') or self._parse_family(mn)
            if not fam:
                continue
            by_family2.setdefault(fam, []).append(r)

        for fam, rows in by_family2.items():
            best = sorted(rows, key=lambda r: self._candidate_rank(r['model_name']))[0]
            selected_env_name = self._strip_models_prefix(best['model_name'])
            reason = f"multimodal=yes, rank={self._candidate_rank(best['model_name'])}"
            await self.db.upsert_gemini_model_selection(
                family=fam,
                selected_model_name=selected_env_name,
                selection_reason=reason,
            )

        # 마지막 갱신 시각 저장
        await self.db.set_kv(self._model_refresh_kv_key, now_iso)

        # 갱신 후 selection 로드
        await self.load_selected_models_from_db()
        logger.info("Gemini 모델 레지스트리 갱신 완료")
        return True
    
    async def _rate_limit(self):
        """API 요청 속도 제한 (모델별 동적 조정)"""
        current_model = self._get_current_model()
        min_interval = self._get_model_rate_limit(current_model)
        
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            logger.debug(f"Rate limit: {wait_time:.1f}초 대기 (모델: {current_model})")
            await asyncio.sleep(wait_time)
        self._last_request_time = time.time()
    
    async def analyze_image(
        self, 
        image_path: Path, 
        max_retries: int = None
    ) -> Optional[Dict[str, any]]:
        """
        이미지에서 객체를 인식하고 분류
        
        Args:
            image_path: 분석할 이미지 경로
            max_retries: 최대 재시도 횟수 (기본값: settings에서 가져옴)
            
        Returns:
            {
                'detected_objects': List[Dict],  # 감지된 객체 목록 (바운딩 박스 포함)
                'primary_class': str,            # person, cat, others 중 하나
                'confidence': str,               # 신뢰도 (high, medium, low)
                'description': str               # 상세 설명
            }
        """
        if max_retries is None:
            max_retries = settings.gemini_max_retries
        
        for attempt in range(max_retries):
            try:
                # 속도 제한 적용
                await self._rate_limit()
                
                # 이미지 로드
                image = Image.open(image_path)
                width, height = image.size
                
                # Gemini에게 질문할 프롬프트
                prompt = """Detect all prominent objects in the image. 
IMPORTANT: Only detect and report PERSON or CAT objects. Do not include any other object types in your response.

For each person or cat object, provide:
1. Object type (must be either "person" or "cat" only)
2. Bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000

Then classify the primary object in this image into one of these categories:
- person: if a person is clearly visible
- cat: if a cat is clearly visible  
- others: if neither person nor cat is clearly visible, or if other objects dominate the image

Respond in JSON format:
{
  "objects": [{"type": "person", "box_2d": [100, 200, 500, 600]}, {"type": "cat", "box_2d": [150, 250, 550, 650]}],
  "primary_class": "person",
  "confidence": "high"
}

Remember: objects array should ONLY contain person or cat. Do not include dogs, birds, vehicles, or any other object types."""
                
                # Gemini API 호출
                config = types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
                
                current_model = self._get_current_model()
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=[image, prompt],
                    config=config
                )
                
                # JSON 응답 파싱
                result_data = json.loads(response.text)
                
                # 바운딩 박스를 절대 좌표로 변환
                detected_objects = []
                for obj in result_data.get('objects', []):
                    box_2d = obj.get('box_2d', [])
                    if len(box_2d) == 4:
                        abs_y1 = int(box_2d[0]/1000 * height)
                        abs_x1 = int(box_2d[1]/1000 * width)
                        abs_y2 = int(box_2d[2]/1000 * height)
                        abs_x2 = int(box_2d[3]/1000 * width)
                        
                        detected_objects.append({
                            'type': obj.get('type', 'unknown'),
                            'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],  # [x1, y1, x2, y2]
                            'bbox_normalized': box_2d  # 원본 normalized 값 보존
                        })
                
                # primary_class 검증
                primary_class = result_data.get('primary_class', 'others')
                if primary_class not in self.CLASSES:
                    primary_class = 'others'
                
                result = {
                    'detected_objects': detected_objects,
                    'primary_class': primary_class,
                    'confidence': result_data.get('confidence', 'low'),
                    'description': result_data.get('description', '')
                }
                
                logger.info(
                    f"이미지 분석 완료: {image_path.name} -> "
                    f"{result['primary_class']} ({len(detected_objects)} objects)"
                )
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON 파싱 실패 ({attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = self._calculate_backoff(attempt)
                    logger.info(f"{wait_time:.1f}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                    continue
                    
            except Exception as e:
                error_msg = str(e)

                # 모델이 존재하지 않거나 generateContent 미지원이면 다음 모델로 즉시 전환
                if self._is_model_not_found_error(error_msg):
                    current_model = self._get_current_model()
                    logger.warning(
                        f"Gemini 모델 NOT_FOUND/미지원: API 키[{self.current_key_index}] "
                        f"모델[{self.current_model_index}:{current_model}] - 다음 모델로 전환 시도"
                    )

                    # 1) 같은 API 키에서 다음 모델 시도
                    if self.switch_to_next_model():
                        await asyncio.sleep(1)
                        continue

                    # 2) 모델 모두 소진 시 다음 API 키로 넘어가며 모델 리셋
                    if self.switch_to_next_api_key():
                        await asyncio.sleep(1)
                        continue

                    logger.error(
                        "Gemini 모델 NOT_FOUND/미지원으로 모든 API 키/모델 조합 실패. "
                        "GEMINI_MODELS 설정을 확인하세요."
                    )
                    return None
                
                # 재시도 가능한 에러인지 확인
                retryable_errors = [
                    '503', 'UNAVAILABLE', 'overloaded', 
                    '429', 'RESOURCE_EXHAUSTED', 'quota',
                    '500', 'INTERNAL', 'timeout', 'Timeout'
                ]
                
                is_retryable = any(err in error_msg for err in retryable_errors)
                
                if is_retryable and attempt < max_retries - 1:
                    # 429 에러인 경우 모델 -> API 키 순환
                    if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                        current_model = self._get_current_model()
                        logger.warning(
                            f"할당량 초과: API 키[{self.current_key_index}] "
                            f"모델[{self.current_model_index}:{current_model}]"
                        )
                        
                        # 1. 먼저 다음 모델 시도
                        if self.switch_to_next_model():
                            logger.info("다음 모델로 전환 성공. 즉시 재시도합니다.")
                            await asyncio.sleep(1)  # 1초만 대기
                            continue
                        
                        # 2. 모든 모델 소진 -> 다음 API 키로 전환
                        if self.switch_to_next_api_key():
                            logger.info("다음 API 키로 전환 성공. 즉시 재시도합니다.")
                            await asyncio.sleep(1)
                            continue
                        
                        # 3. 모든 API 키와 모델 소진
                        logger.error("모든 API 키 및 모델 할당량 소진. 내일 다시 시도하세요.")
                        return None
                    else:
                        wait_time = self._calculate_backoff(attempt)
                        logger.warning(
                            f"Gemini API 오류 ({attempt + 1}/{max_retries}), "
                            f"{wait_time:.1f}초 후 재시도: {error_msg[:100]}"
                        )
                        await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Gemini 이미지 분석 실패 ({image_path}): {e}")
                return None
        
        # 모든 재시도 실패
        logger.error(f"Gemini API 재시도 {max_retries}회 모두 실패")
        return None
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        지수 백오프 + 지터 계산
        
        Args:
            attempt: 현재 시도 횟수 (0부터 시작)
            
        Returns:
            대기 시간 (초)
        """
        base_wait = 2 ** attempt  # 1, 2, 4, 8, 16...
        jitter = random.uniform(0, 1)  # 랜덤 지터
        max_wait = 60  # 최대 대기 시간
        
        return min(base_wait + jitter, max_wait)
    
    def batch_analyze(
        self, 
        image_paths: List[Path],
        delay_between: float = 2.0
    ) -> List[Dict[str, any]]:
        """
        여러 이미지를 배치로 분석
        
        Args:
            image_paths: 이미지 경로 리스트
            delay_between: 요청 간 딜레이 (초)
            
        Returns:
            분석 결과 리스트
        """
        results = []
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"배치 분석 진행 중: {idx}/{total}")
            
            result = self.analyze_image(image_path)
            if result:
                results.append({
                    'image_path': str(image_path),
                    **result
                })
            
            # API 부하 방지를 위한 딜레이
            if idx < total:
                time.sleep(delay_between)
        
        logger.info(f"배치 분석 완료: {len(results)}/{total} 성공")
        return results
    
    def health_check(self) -> Dict[str, any]:
        """
        Gemini API 연결 상태 확인
        
        Returns:
            상태 정보
        """
        try:
            # 간단한 텍스트 요청으로 API 상태 확인
            current_model = self._get_current_model()
            response = self.client.models.generate_content(
                model=current_model,
                contents="Respond with just 'OK' if you can read this."
            )
            
            return {
                "status": "healthy",
                "model": current_model,
                "response": response.text.strip()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
