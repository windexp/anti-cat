"""환경 설정 관리 모듈"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path
from typing import List
import json


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Frigate 설정
    frigate_url: str
    frigate_api_key: str = ""  # Frigate는 기본적으로 API 키 불필요
    
    # Gemini API 설정 (JSON 배열 형식)
    gemini_api_keys: str  # '["key1", "key2"]' 형식
    gemini_api_key_list: List[str] = []  # 파싱된 리스트
    gemini_models: str = '["gemini-2.5-flash-lite"]'  # 모델 리스트 (JSON 배열) - 2.0은 더 이상 지원되지 않음
    gemini_model_list: List[str] = []  # 파싱된 모델 리스트
    
    @field_validator('gemini_api_key_list', mode='before')
    @classmethod
    def parse_api_keys(cls, v, info):
        """gemini_api_keys를 파싱하여 gemini_api_key_list 생성"""
        # 이미 파싱된 경우 그대로 반환
        if isinstance(v, list) and len(v) > 0:
            return v
        
        # gemini_api_keys에서 파싱
        if hasattr(info, 'data') and 'gemini_api_keys' in info.data:
            keys_str = info.data['gemini_api_keys']
            try:
                # JSON 파싱 시도
                parsed = json.loads(keys_str)
                if isinstance(parsed, list):
                    return [k.strip() for k in parsed if k.strip()]
            except (json.JSONDecodeError, ValueError):
                # JSON 실패 시 콤마 구분 파싱
                return [k.strip() for k in keys_str.split(',') if k.strip()]
        
        return []
    
    @field_validator('gemini_model_list', mode='before')
    @classmethod
    def parse_models(cls, v, info):
        """gemini_models를 파싱하여 gemini_model_list 생성"""
        # 이미 파싱된 경우 그대로 반환
        if isinstance(v, list) and len(v) > 0:
            return v
        
        # gemini_models에서 파싱
        if hasattr(info, 'data') and 'gemini_models' in info.data:
            models_str = info.data['gemini_models']
            try:
                # JSON 파싱 시도
                parsed = json.loads(models_str)
                if isinstance(parsed, list):
                    return [m.strip() for m in parsed if m.strip()]
            except (json.JSONDecodeError, ValueError):
                # JSON 실패 시 콤마 구분 파싱
                return [m.strip() for m in models_str.split(',') if m.strip()]
        
        return ['gemini-2.5-flash-lite']  # 기본값 (2.0은 더 이상 지원되지 않음)
    
    # 이벤트 폴링 설정
    polling_interval_seconds: int = 300  # 5분마다 폴링
    event_max_age_days: int = 3  # 최대 3일 전 이벤트까지 조회
    max_events_per_cycle: int = 5  # 한 사이클당 최대 처리 이벤트 수
    event_process_delay: float = 3.0  # 이벤트 처리 간 대기 시간 (초)
    gemini_retry_interval_seconds: int = 60  # Gemini 재시도 간격 (1분)
    gemini_max_retries: int = 5  # Gemini 최대 재시도 횟수
    
    # Synthetic data 생성 확률 (카메라별 설정)
    synthetic_probability_main_entrance: float = 0.05  # main entrance: 5%
    synthetic_probability_garden: float = 1.0  # garden: 100%
    synthetic_probability_default: float = 0.3  # 기타 카메라: 30%
    
    # 데이터 저장 경로
    data_dir: Path = Path("./data")
    images_dir: Path = Path("./data/images")
    labels_dir: Path = Path("./data/labels")
    mismatched_dir: Path = Path("./data/mismatched")  # 분류 불일치 데이터
    db_path: Path = Path("./data/events.db")  # SQLite 데이터베이스
    
    # 웹서버 설정
    web_host: str = "0.0.0.0"
    web_port: int = 8150
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # API 키 리스트 파싱 (값이 비어있으면 다시 파싱)
        if not self.gemini_api_key_list:
            try:
                parsed = json.loads(self.gemini_api_keys)
                if isinstance(parsed, list):
                    self.gemini_api_key_list = [k.strip() for k in parsed if k.strip()]
            except (json.JSONDecodeError, ValueError):
                self.gemini_api_key_list = [k.strip() for k in self.gemini_api_keys.split(',') if k.strip()]
        
        # 모델 리스트 파싱
        if not self.gemini_model_list:
            try:
                parsed = json.loads(self.gemini_models)
                if isinstance(parsed, list):
                    self.gemini_model_list = [m.strip() for m in parsed if m.strip()]
            except (json.JSONDecodeError, ValueError):
                self.gemini_model_list = [m.strip() for m in self.gemini_models.split(',') if m.strip()]
        
        if not self.gemini_model_list:
            self.gemini_model_list = ['gemini-2.5-flash-lite']  # 기본값 (2.0은 더 이상 지원되지 않음)
        
        # 디렉토리 생성
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.mismatched_dir.mkdir(exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()
