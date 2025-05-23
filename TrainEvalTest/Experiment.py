from torch.optim.lr_scheduler import ReduceLROnPlateau

from TrainEvalTest.Trainer import Trainer
from Datasets import DidiXianNovDataset, DidiChengduNovDataset
from JimmyTorch.Datasets import DEVICE
from JimmyTorch.Training import *
from JimmyTorch.Models import JimmyModel
from JimmyTorch.DynamicConfig import DynamicConfig
import torch
from typing import *
from datetime import datetime
import os
from rich import print as rprint
import pandas as pd


class Experiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str):
        self.comments = comments

        self.optimizer_cfg = DynamicConfig(torch.optim.AdamW,
                                           lr=2e-4,
                                           amsgrad=True)

        self.model_cfg = DynamicConfig(JimmyModel,
                                       optimizer_cls=torch.optim.AdamW,
                                       optimizer_args={"lr": 2e-4, "amsgrad": True},
                                       mixed_precision=True,
                                       compile_model=False,
                                       clip_grad=0.0)

        self.train_set_cfg = DynamicConfig(DidiXianNovDataset,
                                           dataset_root="/home/jimmy/Data/Didi/",
                                           pad_to_len=512,
                                           min_erase_rate=0.1,
                                           max_erase_rate=1.0,
                                           batch_size=64,
                                           drop_last=True,
                                           shuffle=True,
                                           set_name="train",
                                           compute_guess = False, )

        self.eval_set_cfg = DynamicConfig(DidiXianNovDataset,
                                           dataset_root="/home/jimmy/Data/Didi/",
                                           pad_to_len=512,
                                           min_erase_rate=0.5,
                                           max_erase_rate=0.6,
                                           batch_size=64,
                                           drop_last=True,
                                           shuffle=False,
                                           set_name="eval",
                                           compute_guess=False, )

        self.test_set_cfg = DynamicConfig(DidiXianNovDataset,
                                          dataset_root="/home/jimmy/Data/Didi/",
                                          pad_to_len=512,
                                          min_erase_rate=0.5,
                                          max_erase_rate=0.6,
                                          batch_size=1,
                                          drop_last=True,
                                          shuffle=False,
                                          set_name="test",
                                          compute_guess=False, )


        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg = DynamicConfig(ReduceLROnPlateau,
                                            mode="min",
                                            factor=0.5,
                                            patience=10,
                                            threshold=0.01,
                                            min_lr=1e-6,
                                            verbose=False)

        # Other constants for the experiment.
        self.constants = {
            "n_epochs": 200,
            "moving_avg": 1000,
            "eval_interval": 1,
        }

        self.trainer_type = Trainer


    def __str__(self):
        return (f"Experiment{{\n"
                f"\ttrainset={self.train_set_cfg}\n"
                f"\tevalset={self.eval_set_cfg}\n"
                f"\ttestset={self.test_set_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")


    def __repr__(self):
        return self.__str__()


    def start(self, checkpoint: str = None) -> Trainer:
        """
        Start the experiment with the given comments.
        :param comments: Comments to be added to the Experiment.
        :return: A `JimmyTrainer` object with almost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        train_set = self.train_set_cfg.build()
        eval_set = self.eval_set_cfg.build()

        self.model_cfg.optimizer_cls = self.optimizer_cfg.cls
        self.model_cfg.optimizer_args = {"lr": self.optimizer_cfg.lr,
                                         "amsgrad": self.optimizer_cfg.amsgrad}
        model = self.model_cfg.build().to(DEVICE)
        model.initialize()

        if checkpoint is not None:
            model.loadFrom(checkpoint)

        self.lr_scheduler_cfg.optimizer = model.optimizer
        lr_scheduler = self.lr_scheduler_cfg.build()

        trainer_kwargs = {"train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # Create Experiment directories
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_name = trainer_kwargs["train_set"].__class__.__name__
        model_name = model.__class__.__name__
        save_dir = f"Runs/{dataset_name}/{model_name}/{now_str}/"
        log_dir = save_dir

        # Create directories if they do not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(log_dir, "model_arch.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(log_dir, "comments.txt"), "w") as f:
            f.write(f"{self.comments}\n{self.__str__()}")

        rprint(f"[blue]Save directory: {save_dir}.[/blue]")
        rprint(f"[blue]Log directory: {log_dir}.[/blue]")

        trainer_kwargs["log_dir"] = log_dir
        trainer_kwargs["save_dir"] = save_dir

        trainer = self.trainer_type(**trainer_kwargs)
        trainer.start()

        rprint(f"[blue]Training done. Start testing.[/blue]")
        test_set = self.test_set_cfg.build()
        test_losses = trainer.evaluate(test_set, compute_avg=False)

        test_report = pd.DataFrame.from_dict(test_losses)
        test_report.to_csv(os.path.join(log_dir, "test_report.csv"))

        rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer


