import json
from pathlib import Path

history_file = Path('artifacts/history.json')
if history_file.exists():
    data = json.loads(history_file.read_text())
    last = data[-1]
    print(f"Total entries: {len(data)}")
    print(f"Last: Round {last['round']}, Server Update {last['server_update']}, Loss: {last['loss']:.4f}")
else:
    print("History file not found")
