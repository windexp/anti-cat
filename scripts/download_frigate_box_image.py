#!/usr/bin/env python3
"""
Frigate API로 바운딩 박스가 그려진 이미지 다운로드
"""
import sys
import requests
from pathlib import Path

def download_frigate_images(event_id, frigate_url="https://cam.windexp.online", output_dir="data/frigate_boxes"):
    """
    Frigate API로 이벤트의 썸네일(박스 포함)과 스냅샷(원본) 다운로드
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 썸네일 (바운딩 박스 포함)
    thumbnail_url = f"{frigate_url}/api/events/{event_id}/thumbnail.jpg"
    thumbnail_file = output_path / f"{event_id}_thumbnail.jpg"
    
    print(f"Downloading thumbnail (with box): {thumbnail_url}")
    response = requests.get(thumbnail_url)
    if response.status_code == 200:
        with open(thumbnail_file, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved: {thumbnail_file} ({len(response.content)} bytes)")
    else:
        print(f"✗ Failed: {response.status_code}")
    
    # 스냅샷 (원본, 박스 없음)
    snapshot_url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg"
    snapshot_file = output_path / f"{event_id}_snapshot.jpg"
    
    print(f"Downloading snapshot (original): {snapshot_url}")
    response = requests.get(snapshot_url)
    if response.status_code == 200:
        with open(snapshot_file, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved: {snapshot_file} ({len(response.content)} bytes)")
    else:
        print(f"✗ Failed: {response.status_code}")
    
    # 스냅샷 (바운딩 박스 + 리전 포함)
    snapshot_box_url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg?bbox=1"
    snapshot_box_file = output_path / f"{event_id}_snapshot_with_box.jpg"
    
    print(f"Downloading snapshot (with bbox & region): {snapshot_box_url}")
    response = requests.get(snapshot_box_url)
    if response.status_code == 200:
        with open(snapshot_box_file, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved: {snapshot_box_file} ({len(response.content)} bytes)")
    else:
        print(f"✗ Failed: {response.status_code}")
    
    return thumbnail_file, snapshot_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python download_frigate_box_image.py <event_id>")
        print("Example: python download_frigate_box_image.py 1767101564.720441-bbazyf")
        sys.exit(1)
    
    event_id = sys.argv[1]
    frigate_url = sys.argv[2] if len(sys.argv) > 2 else "https://cam.windexp.online"
    
    download_frigate_images(event_id, frigate_url)
