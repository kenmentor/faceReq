"""
Thread-safe task queue.
"""

import queue
import threading

class TaskQueue:
    def __init__(self, maxsize=0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
    
    def enqueue(self, task):
        self.queue.put(task)
    
    def dequeue(self, timeout=None):
        return self.queue.get(timeout=timeout)
    
    def size(self):
        return self.queue.qsize()
