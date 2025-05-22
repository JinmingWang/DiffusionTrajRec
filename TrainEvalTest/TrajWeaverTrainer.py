# Nothing different from the default Trainer

import torch
import torch.nn as nn
from Datasets import *
from JimmyTorch.Datasets import MultiThreadLoader, DEVICE, plotTraj
from JimmyTorch.Training import *
from JimmyTorch.Models import JimmyModel
from datetime import datetime
import os
from rich import print as rprint
from typing import Dict, Any
import inspect
import matplotlib.pyplot as plt


class TrajWeaverTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 train_set: StatePropDidiDataset,
                 eval_set: StatePropDidiDataset,
                 model: JimmyModel,
                 lr_scheduler: JimmyLRScheduler | torch.optim.lr_scheduler._LRScheduler,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int,
                 moving_avg: int,
                 eval_interval: int) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param train_set: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: A `JimmyModel` object. The model must implement a `forwardBackward` function that returns a dictionary of loss and output, and must have a `loss_names` attribute that lists the keys of the loss dictionary.
        :param optimizer: A PyTorch optimizer used to update the model parameters.
        :param lr_scheduler: A learning rate scheduler. It can be a `JimmyLRScheduler` or a PyTorch learning rate scheduler. The scheduler must implement an `update` method that takes the current loss as an argument.
        :param log_dir: A string specifying the directory where the training logs will be saved.
        :param save_dir: A string specifying the directory where the model checkpoints will be saved.
        :param n_epochs: An integer specifying the number of training epochs.
        :param moving_avg: An integer specifying the window size for calculating the moving average of the loss. Default is 100.
        :param eval_interval: An integer specifying the interval (in Iterations) at which to evaluate the model on the validation set.
        """

        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model

        if not hasattr(lr_scheduler, 'update'):
            # get number of arguments of lr_scheduler.step()
            num_args = len(inspect.signature(lr_scheduler.step).parameters)
            if num_args == 1:
                lr_scheduler.update = lambda metric: lr_scheduler.step()
            else:
                lr_scheduler.update = lambda metric: lr_scheduler.step(metric)

        self.lr_scheduler = lr_scheduler
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.moving_avg = moving_avg
        self.eval_interval = eval_interval  # !!! In iterations, not epochs


    def start(self) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        pm_log_tags = self.model.train_loss_names + ["LR"]
        tm_log_tags = self.model.train_loss_names + self.model.eval_loss_names + ["LR"]
        # Initialize progress manager and tensorboard manager
        pm = ProgressManager(self.train_set.total_iters, self.n_epochs, 5, 2, custom_fields=pm_log_tags)
        tm = TensorBoardManager(self.log_dir, tags=tm_log_tags, value_types=["scalar"] * len(tm_log_tags))
        tm.register("Visualization", "figure")
        ma_losses = {name: MovingAvg(self.moving_avg) for name in self.model.train_loss_names}

        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            # loader = MultiThreadLoader(self.train_set, 3)
            loader = self.train_set
            for i, data_dict in enumerate(loader):
                # forward, backward, optimization
                loss_dict, output_dict = self.model.trainStep(data_dict)

                self.train_set.updateState(output_dict["state_features"])

                # Compute moving average of losses
                for loss_name in self.model.train_loss_names:
                    ma_losses[loss_name].update(loss_dict[loss_name])
                    loss_dict[loss_name] = ma_losses[loss_name].get()

                # Update progress manager
                pm.update(epoch, i, LR=self.model.lr, **loss_dict)

                if pm.overall_progress % self.eval_interval == 0:
                    # Update learning rate scheduler
                    self.lr_scheduler.update(loss_dict["Train_MSE"])

                    # Update tensorboard
                    tm.log(pm.overall_progress, LR=self.model.lr, **loss_dict)

                    eval_losses = self.evaluate(self.eval_set, pm=pm, tm=tm)

                    # 更新tensorboard
                    tm.log(pm.overall_progress, **eval_losses)

                    # 根据eval_losses["MAE"]来判断最好的模型
                    eval_loss = eval_losses["Eval_MSE"]
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.model.saveTo(os.path.join(self.save_dir, "best.pth"))
                    self.model.saveTo(os.path.join(self.save_dir, "last.pth"))

        pm.close()


    def evaluate(self,
                 dataset: DidiDataset,
                 compute_avg: bool=True,
                 pm: ProgressManager = None,
                 tm: TensorBoardManager = None) -> Dict[str, Any]:
        n_batches = dataset.n_batches
        eval_losses = {name: torch.zeros(n_batches).to(DEVICE) for name in self.model.eval_loss_names}
        self.model.eval()

        for i, data_dict in enumerate(dataset):
            loss_dict, output_dict = self.model.evalStep(data_dict)

            for name in self.model.eval_loss_names:
                eval_losses[name][i] = loss_dict[name]

        if tm is not None:
            tm.log(pm.overall_progress, Visualization=output_dict["fig"])

        self.model.train()

        if compute_avg:
            return {name: torch.mean(eval_losses[name]).item() for name in self.model.eval_loss_names}

        return {name: eval_losses[name].cpu().numpy() for name in self.model.eval_loss_names}