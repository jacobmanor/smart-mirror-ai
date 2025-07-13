import csv
from datetime import datetime
import os

def log_outfit(image_file, label, gpt_response, log_path="outfit_log.csv"):
    row = [datetime.now().isoformat(), image_file, label, gpt_response]
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_file", "label", "gpt_response"])
        writer.writerow(row)
