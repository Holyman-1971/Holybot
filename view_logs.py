import os
import json
from datetime import datetime

LOG_FILE = "./logs/chat_logs.jsonl"

def inspect_logs():
    if not os.path.exists(LOG_FILE):
        print(f"❌ Log file not found at '{LOG_FILE}'. Have you sent a message with logging enabled?")
        return

    print(f"=== Reading Logs from {LOG_FILE} ===\n")
    
    total_entries = 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total_entries += 1
                
                # Format timestamp
                raw_time = entry.get("timestamp", "")
                dt = datetime.fromisoformat(raw_time) if raw_time else "Unknown Time"
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else raw_time

                print(f"--- Entry #{i} [{formatted_time}] ---")
                print(f"Session ID : {entry.get('session_id')}")
                print(f"Mode       : {entry.get('mode')}")
                print(f"User Prompt: {entry.get('user_prompt')}")
                print(f"Response   : {entry.get('bot_response')[:100]}...\n")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse line {i}: {e}")

    print(f"=== Total Logged Interactions: {total_entries} ===")

if __name__ == "__main__":
    inspect_logs()