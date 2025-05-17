from Experiment import Experiment
from Models import *
from Diffusion import *
from JimmyTorch.Datasets import DEVICE
from JimmyTorch.DynamicConfig import DynamicConfig
from JimmyTorch.Training import loadModels



def trainRoadMAE():
    experiment = Experiment("训练小一些的RoadMAE，encoder和decoder数量用2和4取代4和8.")
    experiment.model_cfg.cls = RoadMAE
    experiment.model_cfg.add(
        mask_ratio=0.5,
        patch_len=8,
        encoder_blocks=2,
        decoder_blocks=4,
        heads=4,
        hidden_dim=256,
        encode_dim=128,
        max_len=512,
    )

    experiment.model_cfg.compile_model = True
    experiment.model_cfg.mixed_precision = False

    experiment.dataset_cfg.compute_guess = False
    experiment.dataset_cfg.batch_size = 256

    experiment.constants["n_epochs"] = 250
    experiment.lr_scheduler_cfg.patience = 15

    trainer = experiment.start()
    return trainer.model


def trainGeoUNet(road_mae):
    road_mae.mask_ratio = 0.0
    road_mae.eval()
    # road_mae = RoadMAE(0.0, 8, 4, 8, 4, 256, 128, 512).to(DEVICE)
    # loadModels("Runs/DidiXianNovDataset/RoadMAE/250514_062607/best.pth", model=road_mae)

    experiment = Experiment("之前的模型还是太准了，这次使用更小的RoadMAE，同时扩散步骤从500改为300")

    ddm = DDIM(0.0001, 0.05, 300, scale_mode="quadratic", skip_step=10, device=DEVICE)

    # ddm = DDPM([512, 2], 0.0001, 0.05, 500)

    experiment.model_cfg.cls = GeoUNet
    experiment.model_cfg.add(
        road_mae=road_mae,
        ddm=ddm,
        D_hidden=128,
        D_condition=128,
        D_time=64,
        num_heads=4,
        num_res=2
    )
    experiment.model_cfg.compile_model = True
    experiment.model_cfg.mixed_precision = False

    experiment.dataset_cfg.compute_guess = False
    experiment.dataset_cfg.batch_size = 256

    experiment.constants["n_epochs"] = 250
    experiment.lr_scheduler_cfg.patience = 15

    experiment.start()


if __name__ == '__main__':
    road_mae = trainRoadMAE()
    trainGeoUNet(road_mae)

