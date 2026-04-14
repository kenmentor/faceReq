"""
Pipeline task scheduler.
"""

class TaskScheduler:
    def __init__(self):
        self.tasks = []
    
    def schedule(self, task):
        self.tasks.append(task)
    
    def run(self):
        for task in self.tasks:
            task.execute()
