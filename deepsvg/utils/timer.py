import time


class Timer:
    def __init__(self):
        self.time = time.time()

    def reset(self):
        self.time = time.time()

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.time
        self.reset()
        return elapsed_time
