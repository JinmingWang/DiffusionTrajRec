from Datasets import StatePropDidiXianDataset, StatePropDidiChengduDataset
from TrainEvalTest.Experiment import Experiment
from Models import *
from Diffusion import *
from JimmyTorch.Datasets import DEVICE
from TrainEvalTest.TrajWeaverTrainer import TrajWeaverTrainer


def trainTrajWeaver():
    experiment = Experiment("第一次训练西安数据集")

    ddm = DDIM(0.0001, 0.05, 300, scale_mode="quadratic", skip_step=10, device=DEVICE)

    # ddm = DDPM([512, 2], 0.0001, 0.05, 500)

    experiment.model_cfg.cls = TrajWeaver
    experiment.model_cfg.add(
        ddm=ddm,
        d_in=8,
        d_out=2,
        d_state=64,
        d_list=[64, 96, 128, 128, 128],
        d_embed=64,
        l_traj=512,
        n_heads=4
    )
    experiment.model_cfg.compile_model = False
    experiment.model_cfg.mixed_precision = False

    experiment.dataset_cfg.cls = StatePropDidiXianDataset
    experiment.dataset_cfg.add(
        n_iters=30_0000,
        ddm=ddm,
        state_shapes=TrajWeaver.getStateShapes(512, 64, 4),
        t_distribution="uniform",
    )
    experiment.dataset_cfg.batch_size = 256
    del experiment.dataset_cfg.compute_guess

    # TrajWeaverTrainer Uses Each Data sample many times (300//10 = 30 times in this case)
    experiment.trainer_type = TrajWeaverTrainer

    # The number of epochs should be small because of the reuse of data samples
    experiment.constants["n_epochs"] = 1
    # As a result, it updates lr_scheduler, logs, save model and do evaluation not after each epoch,
    # But every time it finishes eval_interval iterations
    experiment.constants["eval_interval"] = 1000    # Approximately 300 data samples
    experiment.constants["moving_avg"] = 1000

    experiment.start()


if __name__ == '__main__':
    trainTrajWeaver()

