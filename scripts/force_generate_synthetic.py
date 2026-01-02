import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.dataset_manager import DatasetManager
from src.utils.config import settings
import requests
from PIL import Image
import random

def get_snapshot_from_recordings(camera, timestamp, save_path):
    """
    Frigate Recordings API를 사용하여 스냅샷 가져오기
    URL: /api/<camera_name>/recordings/<frame_time>/snapshot.jpg
    """
    base_url = settings.frigate_url.rstrip('/')
    url = f"{base_url}/api/{camera}/recordings/{timestamp}/snapshot.jpg"
    
    try:
        # print(f"Requesting: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # 저장
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return response.content
        else:
            print(f"  -> API Error: {response.status_code} for {url}")
            return None
    except Exception as e:
        print(f"  -> Request Failed: {e}")
        return None

async def main():
    manager = DatasetManager()
    await manager.initialize()
    
    # Force probability to 100%
    manager.synthetic_probability = 1.0
    
    # Get classified events
    events = await manager.get_classified_events()
    print(f"Found {len(events)} classified events.")
    
    # Filter events: snapshot_downloaded == 1 AND created_at within last 24 hours
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    
    filtered_events = []
    for event in events:
        # Check snapshot
        if event.get('snapshot_downloaded') != 1:
            continue
            
        # Check time
        created_at_str = event.get('created_at')
        if not created_at_str:
            continue
            
        try:
            created_at = datetime.fromisoformat(created_at_str)
            if created_at < one_day_ago:
                continue
        except ValueError:
            continue
            
        filtered_events.append(event)
    
    print(f"Filtered {len(filtered_events)} events (snapshot exists & within 24h).")
    
    # Sort by date (newest first)
    filtered_events.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Try to generate for the filtered events
    count = 0
    target_count = 5
    
    print(f"Attempting to generate synthetic samples using CORRECT API URL...")
    
    for event in filtered_events:
        if count >= target_count:
            break
            
        # Skip if not person or cat
        label = event.get('final_label')
        if label not in ('person', 'cat'):
            continue
            
        event_id = event['event_id']
        frigate_data = event.get('frigate_data')
        camera = frigate_data.get('camera')
        
        # Source time
        created_at = datetime.fromisoformat(event['created_at'])
        
        # Try up to 3 times to get a valid image
        for attempt in range(3):
            # Random time offset
            random_offset = timedelta(seconds=random.uniform(-86400, 86400))
            target_time = created_at + random_offset
            target_timestamp = target_time.timestamp()
            
            print(f"Generating for {event_id} ({label}) @ {target_time} (Attempt {attempt+1})")
            
            # Temp path
            temp_filename = f"temp_synthetic_{event_id}_{int(target_timestamp)}.jpg"
            temp_path = manager.synthetic_dir / temp_filename
            
            # Use the NEW function with correct URL
            image_data = get_snapshot_from_recordings(camera, target_timestamp, temp_path)
            
            if not image_data:
                continue
                
            # Validate Image (Check for black/blank frames)
            try:
                with Image.open(temp_path) as img:
                    extrema = img.getextrema()
                    is_invalid = False
                    
                    # Check for (1,1,1) or very dark images
                    if extrema:
                        if isinstance(extrema[0], tuple): # RGB
                            # Check if max value of all channels is very low (e.g. < 10)
                            # Or if min == max (solid color)
                            max_vals = [ch[1] for ch in extrema]
                            if all(v < 10 for v in max_vals):
                                is_invalid = True
                        else: # Grayscale
                            if extrema[1] < 10:
                                is_invalid = True
                    
                    if is_invalid:
                        print(f"  -> Invalid image (too dark/blank). Skipping.")
                        temp_path.unlink(missing_ok=True)
                        continue
                    
                    # If valid, proceed to crop (using DatasetManager logic manually here for test)
                    print(f"  -> Valid image found! Size: {img.size}")
                    
                    # --- CROP & SAVE LOGIC ---
                    # Expected detect resolution
                    if camera == "main_entrance":
                        fw, fh = 640, 360
                    elif camera == "backyard":
                        fw, fh = 640, 360
                    elif camera == "garden":
                        fw, fh = 960, 540
                    else:
                        fw, fh = 640, 360
                    
                    # Crop logic
                    data_obj = frigate_data.get('data', {})
                    box = data_obj.get('box')
                    
                    if box:
                        x_min = int(box[0] * fw)
                        y_min = int(box[1] * fh)
                        w = int(box[2] * fw)
                        h = int(box[3] * fh)
                        
                        size = max(300, int(1.1 * max(w, h)))
                        size = (size // 4) * 4
                        
                        cx, cy = x_min + w // 2, y_min + h // 2
                        x1 = cx - size // 2
                        y1 = cy - size // 2
                        
                        if x1 < 0: x1 = 0
                        elif x1 > fw - size: x1 = fw - size
                        
                        if y1 < 0: y1 = 0
                        elif y1 > fh - size: y1 = fh - size
                        
                        # Resize if needed
                        if img.width != fw or img.height != fh:
                            print(f"  -> Resizing from {img.width}x{img.height} to {fw}x{fh}")
                            img = img.resize((fw, fh), Image.LANCZOS)
                        
                        cropped = img.crop((x1, y1, x1 + size, y1 + size))
                        
                        # Save final synthetic image
                        synthetic_id = f"synthetic_{event_id}_{int(target_timestamp)}"
                        final_filename = f"{synthetic_id}.jpg"
                        final_path = manager.synthetic_dir / final_filename
                        
                        cropped.save(final_path, quality=95)
                        print(f"  -> Saved synthetic sample: {final_path}")
                        
                        # Register to DB (Simulated)
                        synthetic_event_data = {
                            "event_id": synthetic_id,
                            "status": "pending",
                            "frigate_label": "synthetic",
                            "frigate_data": {
                                "camera": camera,
                                "timestamp": target_timestamp,
                                "source_event_id": event_id,
                                "crop_region": [x1, y1, size]
                            },
                            "image_path": str(final_path),
                            "snapshot_downloaded": 1,
                            "snapshot_error": None,
                            "gemini_result": None,
                            "final_label": None,
                            "bound_box": None,
                            "is_bound_box": 0,
                            "is_synthetic": 1,
                            "source_event_id": event_id,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                            "error_message": None
                        }
                        await manager.db.insert_event(synthetic_event_data)
                        print(f"  -> Registered to DB: {synthetic_id}")

                    count += 1
                    temp_path.unlink(missing_ok=True) # Clean up for test
                    break # Success for this event
                    
            except Exception as e:
                print(f"  -> Image validation error: {e}")
                continue
        
    print(f"Done. Found {count} valid synthetic source images.")

if __name__ == "__main__":
    asyncio.run(main())
