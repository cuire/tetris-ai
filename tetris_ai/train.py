import os

import torch
from pytorch_lightning import Trainer

from tetris_ai.ai.trainer import DQNLightning

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())


def train():
    model = DQNLightning()

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=1,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
