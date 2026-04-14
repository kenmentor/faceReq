"""
Process-level thread pool.
"""
from concurrent.futures import ThreadPoolExecutor
class ThreadPool:
    def __init__(self, max_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
