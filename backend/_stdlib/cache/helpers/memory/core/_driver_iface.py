"""
Core system driver interface.
"""

class DriverInterface:
    @staticmethod
    def load():
        return True
    
    @staticmethod
    def unload():
        pass
