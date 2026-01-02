import asyncio
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.database import db

async def main():
    await db.initialize()
    
    # Get all events
    events = await db.get_all_events()
    
    print(f"Total events: {len(events)}")
    
    background_candidates = []
    for e in events:
        label = e.get('final_label')
        if label in ['background', None]:
            background_candidates.append(e)
            
    print(f"Background candidates (label is background or None): {len(background_candidates)}")
    
    for e in background_candidates:
        print(f"ID: {e['event_id']}")
        print(f"  Label: {e.get('final_label')}")
        print(f"  Status: {e.get('status')}")
        print(f"  is_bound_box: {e.get('is_bound_box')}")
        print(f"  has_snapshot: {e.get('has_snapshot')}")
        print("-" * 20)

if __name__ == "__main__":
    asyncio.run(main())
