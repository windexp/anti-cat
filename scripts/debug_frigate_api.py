import sys
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.api.frigate_api import FrigateAPI

def main(event_id):
    frigate_api = FrigateAPI()
    event_details = frigate_api.get_event_details(event_id)

    if event_details:
        print("Event Details:")
        for key, value in event_details.items():
            print(f"{key}: {value}")
    else:
        print("Failed to fetch event details or event not found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_frigate_api.py <event_id>")
        sys.exit(1)

    event_id = sys.argv[1]
    main(event_id)