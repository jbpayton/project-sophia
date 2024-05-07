import os
import time
from datetime import datetime, timedelta


class ConversationFileLogger:
    def __init__(self, directory="logs"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_log_file_path(self, date_str):
        return os.path.join(self.directory, f"{date_str}.txt")

    def log_message(self, role, message_to_log):
        # Write the message to the log file
        date_str = time.strftime("%Y-%m-%d", time.localtime())
        with open(self.get_log_file_path(date_str), 'a', encoding="utf-8") as f:
            f.write(role + ":" + message_to_log + '\n')

    def load_last_n_lines(self, n):
        lines_to_return = []
        current_date = datetime.now()
        while n > 0 and current_date > datetime(2000, 1, 1):  # Assuming logs won't be from before the year 2000
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = self.get_log_file_path(date_str)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    if len(lines) <= n:
                        lines_to_return = lines + lines_to_return
                        n -= len(lines)
                    else:
                        lines_to_return = lines[-n:] + lines_to_return
                        n = 0
            current_date -= timedelta(days=1)

        return lines_to_return

    def append_last_lines_to_messages(self, n, messages):
        lines = self.load_last_n_lines(n)
        for line in lines:
            if line.startswith("user:"):
                messages.append({"role": "user", "content": line[5:]})
            elif line.startswith("system:"):
                messages.append({"role": "system", "content": line[8:]})
            elif line.startswith("assistant:"):
                messages.append({"role": "assistant", "content": line[10:]})
            else:
                messages.append({"role": "unknown", "content": line})
