"""
Container instance factory.
"""

import uuid

class ContainerFactory:
    @staticmethod
    def create():
        return {
            "id": str(uuid.uuid4()),
            "state": "created",
            "metadata": {}
        }
    
    @staticmethod
    def destroy(container):
        container["state"] = "destroyed"
