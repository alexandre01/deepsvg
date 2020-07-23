class TrainVars:
    def __init__(self):
        pass

    def values(self):
        for key in dir(self):
            if not key.startswith("__") and not callable(getattr(self, key)):
                yield key, getattr(self, key)

    def to_dict(self):
        return {key: val for key, val in self.values()}

    def load_dict(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)
