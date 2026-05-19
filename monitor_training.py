import json
import time
from pathlib import Path

history_file = Path(r'c:\Users\PRAKRUTI M SHETTI\Desktop\Number_plate\artifacts\history.json')
last_count = 0
check_interval = 3

print("Monitoring training progress...")
print("-" * 70)

while True:
    try:
        if not history_file.exists():
            print(f"Waiting for history file to be created... ({check_interval}s)")
            time.sleep(check_interval)
            continue
            
        with open(history_file, 'r') as f:
            data = json.load(f)
        current_count = len(data)
        
        if current_count > last_count:
            latest = data[-1]
            current_round = latest['round']
            print(f"[{time.strftime('%H:%M:%S')}] Round {current_round:2d}, Update {latest['server_update']:2d}, Client {latest['client_id']}, Loss: {latest['loss']:.4f}")
            last_count = current_count
        
        # Check if we've completed rounds 11 and 12 (should have 48+8=56 entries)
        if current_count >= 56:
            max_round = max(d['round'] for d in data)
            if max_round >= 12:
                print("-" * 70)
                print(f"✓ Training completed through Round {max_round}!")
                print(f"Total entries: {current_count}")
                print(f"Loss trajectory: {data[0]['loss']:.4f} → {data[-1]['loss']:.4f}")
                break
        
        time.sleep(check_interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(check_interval)
