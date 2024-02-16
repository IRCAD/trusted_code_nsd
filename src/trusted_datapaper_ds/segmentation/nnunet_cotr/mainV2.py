"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from TrainerV2 import Trainer
from utils.tools import Log

LOG = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    log = Log(LOG)

    log.info("Config:", OmegaConf.to_yaml(cfg))
    log.info("Working directory:", os.getcwd())
    log.debug("Debug level message", None)

    log.start("Trainer initialization")
    trainer = Trainer(cfg, log)
    log.end("Trainer initialization")

    if not cfg.training.only_val:
        log.start("Training")
        trainer.run_training()
        log.end("Training")

    log.start("Eval")
    trainer.run_eval()
    log.end("Eval")


if __name__ == "__main__":
    main()
