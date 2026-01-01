import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir, exp_name):
        self.log_path = os.path.join(log_dir, exp_name)
        self.writer = SummaryWriter(self.log_path)
        print(f"Logging to {self.log_path}")

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
