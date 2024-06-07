import json

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []
    
    def log(self, logs):
        self.logs.append(logs)
    
    def flush(self):
        with open(self.log_path, 'a') as f:
            json.dump(self.logs, f, indent=4)
        print(f'Saved logs to {self.log_path}')