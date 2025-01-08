# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from torch.utils.tensorboard import SummaryWriter


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, ckpt_path):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.ckpt_path = ckpt_path
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "runs"))

        logging.info(
            f"Training Metrics: 1px_disp_0...5, 3px_disp_0...5, 5px_disp_0...5, epe_disp_0...5"
        )
    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(
            f"Training Metrics ({self.total_steps}): {training_str + metrics_str}"
        )

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "runs"))
        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0
            self.running_loss[task_key] += metrics[key]

    def update(self):
        self.total_steps += 1
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            print(self.running_loss)
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_path, "runs"))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
