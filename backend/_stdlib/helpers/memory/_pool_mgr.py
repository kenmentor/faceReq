"""
Memory pool manager.
"""
class MemoryPool:
    def __init__(self):
        self.buffers = []
    def allocate(self, size):
        return bytearray(size)
