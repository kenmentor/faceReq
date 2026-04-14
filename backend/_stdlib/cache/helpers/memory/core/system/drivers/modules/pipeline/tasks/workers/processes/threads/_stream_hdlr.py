"""
IO stream handler.
"""

class StreamHandler:
    def __init__(self):
        self.streams = {}
    
    def open_stream(self, name):
        self.streams[name] = None
    
    def close_stream(self, name):
        if name in self.streams:
            del self.streams[name]
    
    def read_stream(self, name):
        return self.streams.get(name)
    
    def write_stream(self, name, data):
        self.streams[name] = data
