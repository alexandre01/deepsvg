import torch.optim as optim
from deepsvg.schedulers.warmup import GradualWarmupScheduler


class _Config:
    """
    Training config.
    """
    def __init__(self, num_gpus=1):

        self.num_gpus = num_gpus                              #

        self.dataloader_module = "deepsvg.svgtensor_dataset"  #
        self.collate_fn = None                                #
        self.data_dir = "./dataset/icons_tensor/"             #
        self.meta_filepath = "./dataset/icons_meta.csv"       #
        self.loader_num_workers = 0                           #

        self.pretrained_path = None                           #

        self.model_cfg = None                                 #

        self.num_epochs = None                                #
        self.num_steps = None                                 #
        self.learning_rate = 1e-3                             #
        self.batch_size = 100                                 #
        self.warmup_steps = 500                               #


        # Dataset
        self.train_ratio = 1.0                                #
        self.nb_augmentations = 1                             #

        self.max_num_groups = 15                              #
        self.max_seq_len = 30                                 #
        self.max_total_len = None                             #

        self.filter_uni = None                                #
        self.filter_category = None                           #
        self.filter_platform = None                           #

        self.filter_labels = None                             #

        self.grad_clip = None                                 #

        self.log_every = 20                                   #
        self.val_every = 1000                                 #
        self.ckpt_every = 1000                                #

        self.stats_to_print = {
            "train": ["lr", "time"]
        }

        self.model_args = []                                  #
        self.optimizer_starts = [0]                           #

    # Overridable methods
    def make_model(self):
        raise NotImplementedError

    def make_losses(self):
        raise NotImplementedError

    def make_optimizers(self, model):
        return [optim.AdamW(model.parameters(), self.learning_rate)]

    def make_schedulers(self, optimizers, epoch_size):
        return [None] * len(optimizers)

    def make_warmup_schedulers(self, optimizers, scheduler_lrs):
        return [GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=self.warmup_steps, after_scheduler=scheduler_lr)
                for optimizer, scheduler_lr in zip(optimizers, scheduler_lrs)]

    def get_params(self, step, epoch):
        return {}

    def get_weights(self, step, epoch):
        return {}

    def set_train_vars(self, train_vars, dataloader):
        pass

    def visualize(self, model, output, train_vars, step, epoch, summary_writer, visualization_dir):
        pass

    # Utility methods
    def values(self):
        for key in dir(self):
            if not key.startswith("__") and not callable(getattr(self, key)):
                yield key, getattr(self, key)

    def to_dict(self):
        return {key: val for key, val in self.values()}

    def load_dict(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

    def print_params(self):
        for key, val in self.values():
            print(f"  {key} = {val}")
