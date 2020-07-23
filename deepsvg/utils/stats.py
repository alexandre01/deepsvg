from collections import defaultdict
from collections import deque
import datetime
import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Stats:
    def __init__(self, num_steps=None, num_epochs=None, steps_per_epoch=None, stats_to_print=None):
        self.step = self.epoch = 0

        if num_steps is not None:
            self.num_steps = num_steps
        else:
            self.num_steps = num_epochs * steps_per_epoch

        self.stats = {
            "train": defaultdict(SmoothedValue),
        }
        self.stats_to_print = {k: set(v) for k, v in stats_to_print.items()}

    def to_dict(self):
        return self.__dict__

    def load_dict(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

    def update(self, split, step, epoch, dict):
        self.step = step
        self.epoch = epoch

        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.stats[split][k].update(v)

    def update_stats_to_print(self, split, stats_to_print):
        self.stats_to_print[split].update(stats_to_print)

    def get_summary(self, split):

        if split == "train":
            completion_pct = self.step / self.num_steps * 100
            eta_seconds = self.stats[split].get("time").global_avg * (self.num_steps - self.step)
            eta_string = datetime.timedelta(seconds=int(eta_seconds))

            s = "[{}/{}, {:.1f}%] eta: {}, ".format(self.step, self.num_steps, completion_pct, eta_string)
        else:
            s = f"[Validation, epoch {self.epoch + 1}] "

        return s + ", ".join(f"{stat}: {self.stats[split].get(stat).median:.4f}" for stat in self.stats_to_print[split])

    def write_tensorboard(self, summary_writer, split):
        summary_writer.add_scalar(f"{split}/epoch", self.epoch + 1, self.step)

        for stat in self.stats_to_print[split]:
            summary_writer.add_scalar(f"{split}/{stat}", self.stats[split].get(stat).median, self.step)

    def is_best(self):
        return True
