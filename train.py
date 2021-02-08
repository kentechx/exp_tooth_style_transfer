import torch, torch.nn as nn, torchvision, torch.nn.functional as F, os, random, numpy as np
from torchvision.utils import save_image
from argparse import ArgumentParser
from models.pl_models import LigAdaIN
from data.teeth import ImageDataModule
from utils.config import get_parser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def init(args):
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    args = get_parser()
    init(args)

    dm = ImageDataModule(args.data_dir, args.batch_size, num_workers=args.train_workers)

    model = LigAdaIN(**vars(args))
    if args.pretrained_weights:
        model = model.load_from_checkpoint(args.pretrained_weights)

    checkpoint_callback = ModelCheckpoint(monitor='recon_loss', save_last=True, save_top_k=20)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=3)
    # trainer.tune(model, dm)
    trainer.fit(model, dm)
