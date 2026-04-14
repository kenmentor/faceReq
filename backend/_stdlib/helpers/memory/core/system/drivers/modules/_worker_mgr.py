"""
Worker thread manager.
"""
import threading
class WorkerManager:
    def __init__(self, num_workers=4):
        self.semaphore = threading.Semaphore(num_workers)
