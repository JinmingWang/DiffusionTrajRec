from Datasets import *
from TrainEvalTest.Experiment import Experiment
from Models import *
from Diffusion import *
from JimmyTorch.Datasets import DEVICE
from TrainEvalTest.TrajWeaverTrainer import TrajWeaverTrainer


def trainTrajWeaver():
    experiment = Experiment("第一次训练西安数据集")

    # --- Define model ---
    ddm = DDIM(0.0001, 0.05, 300, scale_mode="quadratic", skip_step=10, device=DEVICE)
    experiment.model_cfg.cls = TrajWeaver
    experiment.model_cfg.add(
        ddm=ddm,
        d_in=8,
        d_out=2,
        d_state=64,
        d_list=[64, 128, 128, 128, 128],
        d_embed=64,
        d_time=128,
        l_traj=512,
        n_heads=8
    )
    experiment.model_cfg.compile_model = True
    experiment.model_cfg.mixed_precision = False

    # --- Define Train Set ---
    experiment.train_set_cfg.cls = StatePropDidiXianDataset
    experiment.train_set_cfg.add(
        n_iters=30_0000,
        ddm=ddm,
        state_shapes=TrajWeaver.getStateShapes(512, 64, 4),
        t_distribution="uniform",
    )
    experiment.train_set_cfg.batch_size = 256
    del experiment.train_set_cfg.compute_guess

    # --- Define Eval and Test Set ---
    experiment.eval_set_cfg.cls = DidiXianNovDataset
    experiment.eval_set_cfg.batch_size = 256
    experiment.eval_set_cfg.compute_guess = True
    experiment.test_set_cfg.cls = DidiXianNovDataset
    experiment.test_set_cfg.batch_size = 256
    experiment.test_set_cfg.compute_guess = True


    # Use new trainer
    experiment.trainer_type = TrajWeaverTrainer
    experiment.constants["n_epochs"] = 1
    experiment.constants["eval_interval"] = 10
    experiment.constants["moving_avg"] = 1000

    experiment.start()


if __name__ == '__main__':
    trainTrajWeaver()

