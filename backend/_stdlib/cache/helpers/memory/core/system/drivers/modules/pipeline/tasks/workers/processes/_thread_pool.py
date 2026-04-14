"""
Process-level thread pool executor.
"""

from concurrent.futures import ThreadPoolExecutor

class ThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)
    
    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)
