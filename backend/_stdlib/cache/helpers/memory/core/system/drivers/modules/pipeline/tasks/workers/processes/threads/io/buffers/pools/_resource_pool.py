"""
Resource pool for efficient reuse.
"""

class ResourcePool:
    def __init__(self, factory_fn):
        self.factory_fn = factory_fn
        self.resources = []
        self.in_use = set()
    
    def acquire(self):
        if self.resources:
            r = self.resources.pop()
        else:
            r = self.factory_fn()
        self.in_use.add(id(r))
        return r
    
    def release(self, resource):
        self.in_use.discard(id(resource))
        self.resources.append(resource)
