from .defaults_fonts import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = True
        self.dim_z = 128


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_uni = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                           97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

        self.learning_rate = 2e-4 * num_gpus
        self.batch_size = 60 * num_gpus

        self.val_every = 2000
