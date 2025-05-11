from Experiment import Experiment
from Models import *

if __name__ == '__main__':
    experiment = Experiment("First Try To Train RoadMAE")
    experiment.model_cfg["class"] = RoadMAE
    experiment.model_cfg["args"].update({
        "mask_ratio": 0.5,
        "patch_len": 8,
        "encoder_blocks": 4,
        "decoder_blocks": 8,
        "heads": 4,
        "hidden_dim": 256,
        "encode_dim": 128,
        "max_len": 512,
    })

    experiment.dataset_cfg["args"]["batch_size"] = 256

    experiment.start()

