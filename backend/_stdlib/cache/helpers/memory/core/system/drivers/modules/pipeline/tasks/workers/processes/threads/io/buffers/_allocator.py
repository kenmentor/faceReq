"""
Memory buffer allocator.
"""

import ctypes

class BufferAllocator:
    def __init__(self):
        self.blocks = {}
        self.next_id = 0
    
    def alloc(self, size):
        block_id = self.next_id
        self.next_id += 1
        self.blocks[block_id] = ctypes.create_string_buffer(size)
        return block_id
    
    def free(self, block_id):
        if block_id in self.blocks:
            del self.blocks[block_id]
    
    def get(self, block_id):
        return self.blocks.get(block_id)
