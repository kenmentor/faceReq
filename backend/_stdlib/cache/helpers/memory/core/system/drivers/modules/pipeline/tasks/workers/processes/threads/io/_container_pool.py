"""
Container pool manager.
"""

class ContainerPool:
    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        self.containers = []
        self.available = []
        
        for _ in range(pool_size):
            c = {"id": len(self.containers), "active": False}
            self.containers.append(c)
            self.available.append(c)
    
    def acquire(self):
        if self.available:
            c = self.available.pop()
            c["active"] = True
            return c
        return None
    
    def release(self, container):
        container["active"] = False
        self.available.append(container)
