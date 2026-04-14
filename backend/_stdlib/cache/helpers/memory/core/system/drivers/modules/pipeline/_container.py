"""
Process container for isolation.
"""

class ProcessContainer:
    def __init__(self):
        self.namespace = {}
    
    def execute(self, func, *args):
        return func(*args)
    
    def cleanup(self):
        self.namespace.clear()
