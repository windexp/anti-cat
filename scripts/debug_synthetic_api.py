import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
from PIL import Image

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.dataset_manager import DatasetManager
from src.api.frigate_api import FrigateAPI

async def main():
    manager = DatasetManager()
    await manager.initialize()
    frigate_api = FrigateAPI()
    
    # Get classified events
    events = await manager.get_classified_events()
    
    # Filter events: snapshot_downloaded == 1 AND created_at within last 24 hours
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    
    filtered_events = []
    for event in events:
        if event.get('snapshot_downloaded') != 1:
            continue
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
    
    if not filtered_events:
        print("No suitable events found.")
        return

    # Pick the most recent one
    filtered_events.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    event = filtered_events[0]
    
    print(f"Testing with event: {event['event_id']}")
    
    frigate_data = event.get('frigate_data')
    camera = frigate_data.get('camera')
    created_at_str = event.get('created_at')
    source_time = datetime.fromisoformat(created_at_str)
    
    # Generate a random time offset
    random_offset = timedelta(seconds=random.uniform(-86400, 86400))
    target_time = source_time + random_offset
    target_timestamp = target_time.timestamp()
    
    print(f"Camera: {camera}")
    print(f"Source Time: {source_time}")
    print(f"Target Time: {target_time} (Timestamp: {target_timestamp})")
    
    # 1. Download full snapshot
    output_dir = Path("data/debug_synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = output_dir / "debug_full_source.jpg"
    
    print("Downloading full snapshot...")
    image_data = frigate_api.get_snapshot_at_time(camera, target_timestamp, full_path)
    
    if not image_data:
        print("Failed to download snapshot.")
        return
        
    print(f"Full snapshot saved to {full_path}")
    
    # Check image stats
    with Image.open(full_path) as img:
        print(f"Full Image Size: {img.width}x{img.height}")
        extrema = img.getextrema()
        print(f"Image Extrema (Min/Max pixel values): {extrema}")
        
        # Check if it's all black
        if extrema:
            # For RGB, extrema is a list of tuples
            if isinstance(extrema[0], tuple):
                is_black = all(max_val == 0 for min_val, max_val in extrema)
            else:
                is_black = extrema[1] == 0
                
            if is_black:
                print("WARNING: The downloaded image is completely BLACK.")
            else:
                print("The image contains data (not completely black).")

    # 2. Simulate Resize and Crop (using logic from DatasetManager)
    # Expected detect resolution
    if camera == "main_entrance":
        fw, fh = 640, 360
    elif camera == "backyard":
        fw, fh = 640, 360
    elif camera == "garden":
        fw, fh = 960, 540
    else:
        fw, fh = 640, 360 # Default
        
    print(f"Target Detect Resolution: {fw}x{fh}")
    
    # Crop logic from dataset_manager
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
        
        print(f"Crop Region: x={x1}, y={y1}, size={size}")
        
        with Image.open(full_path) as img:
            # Current logic in DatasetManager (without my fix)
            if img.width != fw or img.height != fh:
                print(f"Size mismatch! Image: {img.width}x{img.height}, Expected: {fw}x{fh}")
                print("Applying DatasetManager logic (scaling coordinates)...")
                
                # Logic from DatasetManager:
                # x1 = int(x1 * img.width / fw)
                # y1 = int(y1 * img.height / fh)
                # size = int(size * min(img.width / fw, img.height / fh))
                
                x1_scaled = int(x1 * img.width / fw)
                y1_scaled = int(y1 * img.height / fh)
                size_scaled = int(size * min(img.width / fw, img.height / fh))
                
                print(f"Scaled Crop Region: x={x1_scaled}, y={y1_scaled}, size={size_scaled}")
                
                cropped = img.crop((x1_scaled, y1_scaled, x1_scaled + size_scaled, y1_scaled + size_scaled))
                crop_path = output_dir / "debug_cropped_original_logic.jpg"
                cropped.save(crop_path)
                print(f"Cropped image (original logic) saved to {crop_path}")
                
                extrema = cropped.getextrema()
                print(f"Cropped Extrema (original logic): {extrema}")
                
                # Now try with RESIZE logic (my proposed fix)
                print("Applying RESIZE logic...")
                img_resized = img.resize((fw, fh), Image.LANCZOS)
                cropped_resized = img_resized.crop((x1, y1, x1 + size, y1 + size))
                crop_path_resized = output_dir / "debug_cropped_resize_logic.jpg"
                cropped_resized.save(crop_path_resized)
                print(f"Cropped image (resize logic) saved to {crop_path_resized}")
                
                extrema_resized = cropped_resized.getextrema()
                print(f"Cropped Extrema (resize logic): {extrema_resized}")

if __name__ == "__main__":
    asyncio.run(main())
