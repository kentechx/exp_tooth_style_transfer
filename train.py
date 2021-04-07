import torch, torch.nn as nn, torchvision, torch.nn.functional as F, os, random, numpy as np
from torchvision.utils import save_image
from argparse import ArgumentParser
from models.pl_models import LitVAE
from data.teeth import ImageDataModule

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def get_parser():
    import argparse, yaml
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument("--config", type=str, default="config/default.yaml", help="path to config file")

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


if __name__ == "__main__":
    args = get_parser()
    pl.seed_everything(args.seed)

    dm = ImageDataModule(args.data_dir, args.batch_size, num_workers=args.train_workers)

    model = LitVAE(**vars(args))
    if args.pretrained_weights:
        model = model.load_from_checkpoint(args.pretrained_weights)

    checkpoint_callback = ModelCheckpoint(monitor='loss', save_last=True, save_top_k=20)

    logger = pl_loggers.TestTubeLogger('lightning_logs', name=args.model_name)
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, callbacks=[checkpoint_callback], logger=logger)
    # trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=3)
    # trainer.tune(model, dm)
    trainer.fit(model, dm)
