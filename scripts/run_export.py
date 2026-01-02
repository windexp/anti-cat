import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.dataset_manager import DatasetManager

async def main():
    manager = DatasetManager()
    print("Starting YOLO dataset export...")
    try:
        output_path = await manager.export_yolo_dataset()
        print(f"Export completed successfully. Output path: {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
