import json
import time
from pathlib import Path

history_file = Path(r'c:\Users\PRAKRUTI M SHETTI\Desktop\Number_plate\artifacts_scratch\history.json')
last_count = 0

print("Monitoring artifacts_scratch/history.json for updates...")
print("-" * 70)

while True:
    try:
        if not history_file.exists():
            print(f"Waiting for history file to be created...")
            time.sleep(5)
            continue
        with open(history_file, 'r') as f:
            data = json.load(f)
        current_count = len(data)
        if current_count > last_count:
            latest = data[-1]
            print(f"[{time.strftime('%H:%M:%S')}] Round {latest.get('round')} | Update {latest.get('server_update')} | Client {latest.get('client_id')} | Loss {latest.get('loss')}")
            last_count = current_count
        time.sleep(3)
    except KeyboardInterrupt:
        print('\nMonitoring stopped.')
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(3)
