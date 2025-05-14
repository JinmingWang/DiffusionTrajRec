from Experiment import Experiment
from Models import *
from Diffusion import DDIM
from JimmyTorch.Datasets import DEVICE

if __name__ == '__main__':
    # experiment = Experiment("First Try To Train RoadMAE")
    # experiment.model_cfg["class"] = RoadMAE
    # experiment.model_cfg["args"].update({
    #     "mask_ratio": 0.5,
    #     "patch_len": 8,
    #     "encoder_blocks": 4,
    #     "decoder_blocks": 8,
    #     "heads": 4,
    #     "hidden_dim": 256,
    #     "encode_dim": 128,
    #     "max_len": 512,
    # })
    #
    # experiment.dataset_cfg["args"]["batch_size"] = 256
    #
    # experiment.start()

    experiment = Experiment("First Try To Train GeuUNet")

    road_mae = RoadMAE(0.0, 8, 4, 8, 4, 256, 128, 512).to(DEVICE)

    ddm = DDIM([512, 2], 0.0001, 0.05, 500, scale_mode="quadratic", skip_step=10)

    experiment.model_cfg["class"] = GeoUNet
    experiment.model_cfg["args"].update({
        "road_mae": road_mae,
        "ddm": ddm,
        "D_hidden": 128,
        "D_condition": 128,
        "D_time": 64,
        "num_heads": 4,
        "max_T": 500,
        "num_res": 2
    })

    experiment.dataset_cfg["args"]["batch_size"] = 128

    experiment.start()

