from Datasets import DidiChengduNovDataset, DidiXianNovDataset
from TrainEvalTest.Experiment import Experiment
from Models import *
from Diffusion import *
from JimmyTorch.Datasets import DEVICE


if __name__ == '__main__':
    experiment = Experiment("测试不用SP的TrajWeaver")

    ddm = DDIM(0.0001, 0.05, 300, scale_mode="quadratic", skip_step=10, device=DEVICE)

    # ddm = DDPM([512, 2], 0.0001, 0.05, 500)

    experiment.model_cfg.cls = TrajWeaverNoSP
    experiment.model_cfg.add(
        ddm=ddm,
        d_in=8,
        d_out=2,
        d_list=[64, 128, 128, 128, 128],
        d_embed=64,
    )
    experiment.model_cfg.compile_model = False
    experiment.model_cfg.mixed_precision = False

    experiment.train_set_cfg.cls = DidiXianNovDataset
    experiment.train_set_cfg.set_name = "eval"
    experiment.train_set_cfg.compute_guess = True
    experiment.train_set_cfg.batch_size = 256

    experiment.eval_set_cfg.cls = DidiXianNovDataset
    experiment.eval_set_cfg.compute_guess = True
    experiment.eval_set_cfg.batch_size = 256
    experiment.test_set_cfg.cls = DidiXianNovDataset
    experiment.test_set_cfg.compute_guess = True

    experiment.constants["n_epochs"] = 300
    experiment.lr_scheduler_cfg.patience = 15

    experiment.start()