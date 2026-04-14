"""
IO buffer management.
"""

class BufferManager:
    def __init__(self):
        self.buffers = {}
        self.counter = 0
    
    def create_buffer(self, size):
        idx = self.counter
        self.counter += 1
        self.buffers[idx] = bytearray(size)
        return idx
    
    def get_buffer(self, idx):
        return self.buffers.get(idx)
    
    def destroy_buffer(self, idx):
        if idx in self.buffers:
            del self.buffers[idx]
