import sys
from pathlib import Path

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.api.frigate_api import FrigateAPI

def main():
    frigate_api = FrigateAPI()
    try:
        response = frigate_api.get_event_list()
        if response:
            print("Available Events:")
            for i, event in enumerate(response[:10]):  # Limit to first 10 events
                print(f"{i + 1}. ID: {event.get('id')}, Label: {event.get('label')}, Confidence: {event.get('confidence')}")
        else:
            print("No events found or failed to fetch events.")
    except Exception as e:
        print(f"Error fetching events: {e}")

    id=response[0].get('id')
    event_details = frigate_api.get_event_details(id)
    if event_details:
        print("\nEvent Details for ID:", id)
        for key, value in event_details.items():
            print(f"{key}: {value}")
    else:
        print("Failed to fetch event details or event not found.")
if __name__ == "__main__":
    main()