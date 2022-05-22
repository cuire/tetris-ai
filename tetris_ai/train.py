import argparse
import os

import torch
from pytorch_lightning import Trainer

from tetris_ai.ai.module import DQNLightning

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())


def train(args):
    model = DQNLightning.from_argparse_args(args)

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DQNLightning.add_model_specific_args(parser)
    parser.add_argument("--max_epochs", type=int, default=1)
    args = parser.parse_args()
    train(args)
