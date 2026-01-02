"""Frigate API 연동 모듈"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import requests
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrigateAPI:
    """Frigate API와 상호작용하는 클라이언트 (requests 기반)"""
    
    def __init__(self):
        self.base_url = settings.frigate_url.rstrip('/')
    
    def get_event_snapshot(
        self, 
        event_id: str, 
        save_path: Optional[Path] = None
    ) -> Optional[bytes]:
        """
        이벤트 스냅샷 이미지 다운로드
        
        Args:
            event_id: Frigate 이벤트 ID
            save_path: 이미지 저장 경로 (선택사항)
            
        Returns:
            이미지 바이트 데이터 또는 None
        """
        try:
            url = f"{self.base_url}/api/events/{event_id}/snapshot.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image_data = response.content
            
            # 파일로 저장
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"스냅샷 저장: {save_path}")
            
            return image_data
            
        except requests.HTTPError as e:
            logger.error(f"HTTP 오류 (이벤트 {event_id}): {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"스냅샷 다운로드 실패 (이벤트 {event_id}): {e}")
            return None
    
    def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        이벤트 상세 정보 가져오기
        
        Args:
            event_id: Frigate 이벤트 ID
            
        Returns:
            이벤트 정보 딕셔너리 또는 None
        """
        try:
            url = f"{self.base_url}/api/events/{event_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            event_data = response.json()
            logger.info(f"이벤트 정보 조회 성공: {event_id}")
            return event_data
            
        except requests.HTTPError as e:
            logger.error(f"HTTP 오류 (이벤트 {event_id}): {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"이벤트 정보 조회 실패 (이벤트 {event_id}): {e}")
            return None
    
    def get_latest_frame(
        self, 
        camera_name: str, 
        save_path: Optional[Path] = None
    ) -> Optional[bytes]:
        """
        카메라의 최신 프레임 가져오기
        
        Args:
            camera_name: 카메라 이름
            save_path: 이미지 저장 경로 (선택사항)
            
        Returns:
            이미지 바이트 데이터 또는 None
        """
        try:
            url = f"{self.base_url}/api/{camera_name}/latest.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image_data = response.content
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"최신 프레임 저장: {save_path}")
            
            return image_data
            
        except Exception as e:
            logger.error(f"최신 프레임 가져오기 실패 (카메라 {camera_name}): {e}")
            return None
    
    def get_event_list(
        self,
        limit: int = 10,
        camera: Optional[str] = None,
        label: Optional[str] = None,
        after: Optional[float] = None,
        before: Optional[float] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        이벤트 목록 조회
        
        Args:
            limit: 조회할 이벤트 수 (기본: 10)
            camera: 특정 카메라로 필터링 (선택사항)
            label: 특정 라벨로 필터링 (선택사항, 예: 'cat', 'person')
            after: 이 시간 이후 이벤트만 조회 (Unix timestamp)
            before: 이 시간 이전 이벤트만 조회 (Unix timestamp)
            
        Returns:
            이벤트 목록 (딕셔너리 리스트) 또는 None
        """
        try:
            url = f"{self.base_url}/api/events"
            params = {"limit": limit}
            
            if camera:
                params["camera"] = camera
            if label:
                params["label"] = label
            if after:
                params["after"] = after
            if before:
                params["before"] = before
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            events = response.json()
            logger.info(f"이벤트 목록 조회 성공: {len(events)}개")
            return events
            
        except requests.HTTPError as e:
            logger.error(f"HTTP 오류 (이벤트 목록 조회): {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"이벤트 목록 조회 실패: {e}")
            return None
    
    def get_snapshot_at_time(
        self,
        camera_name: str,
        timestamp: float,
        save_path: Optional[Path] = None
    ) -> Optional[bytes]:
        """
        특정 시간의 카메라 스냅샷 가져오기
        
        Args:
            camera_name: 카메라 이름
            timestamp: Unix timestamp
            save_path: 이미지 저장 경로 (선택사항)
            
        Returns:
            이미지 바이트 데이터 또는 None
        """
        try:
            # Frigate recordings snapshot API: /api/{camera}/recordings/{timestamp}/snapshot.jpg
            url = f"{self.base_url}/api/{camera_name}/recordings/{timestamp}/snapshot.jpg"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image_data = response.content
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"타임스탬프 스냅샷 저장: {save_path} (시간: {timestamp})")
            
            return image_data
            
        except Exception as e:
            logger.error(f"타임스탬프 스냅샷 가져오기 실패 (카메라 {camera_name}, 시간 {timestamp}): {e}")
            return None
