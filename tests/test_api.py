"""API 기능 테스트"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.frigate_api import FrigateAPI
from src.services.gemini_service import GeminiService
from src.services.dataset_manager import DatasetManager
from src.utils.config import settings

def test_event_processing(event_id: str):
    """이벤트 처리 테스트"""
    frigate = FrigateAPI()
    gemini = GeminiService()
    dataset = DatasetManager()
    
    # 이벤트 조회
    event_data = frigate.get_event_details(event_id)
    if not event_data:
        print("✗ 이벤트 조회 실패")
        return False
    
    print(f"✓ 이벤트: {event_data.get('label')} @ {event_data.get('camera')}")
    
    # 이미지 다운로드
    temp_path = Path(f"test_{event_id}.jpg")
    image_data = frigate.get_event_snapshot(event_id, temp_path)
    
    if not image_data:
        print("✗ 이미지 다운로드 실패")
        return False
    
    print(f"✓ 이미지 다운로드: {len(image_data)} bytes")
    
    # Gemini 분석
    result = gemini.analyze_image(temp_path)
    if result:
        print(f"✓ Gemini 분석: {result['primary_class']} ({result['confidence']})")
    
    # 저장
    filename = dataset.save_image_with_label(
        event_id, image_data,
        event_data.get('label', 'unknown'),
        result, None
    )
    
    if filename:
        print(f"✓ 저장 완료: {filename}")
    
    temp_path.unlink(missing_ok=True)
    return True

if __name__ == "__main__":
    if not settings.gemini_api_key:
        print("❌ GEMINI_API_KEY 필요")
        exit(1)
    
    event_id = input("Frigate 이벤트 ID: ").strip()
    if event_id:
        test_event_processing(event_id)
