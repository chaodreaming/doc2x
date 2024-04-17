import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from custom_dataloads import Latex_Dataset
from models.models import Simple_Latex_OCR
from utils import get_processor_model
from utils import load_setting, dir_check, seed_torch, get_Transfer

seed_torch(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="configs/im2latex.yaml",
                        help="Experiment settings")
    parser.add_argument("--name", type=str, default="test",
                        help="Train experiment version")

    parser.add_argument("--num_workers", "-nw", type=int, default=10,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="Batch size for training and validate")
    parser.add_argument("--resume", default=None,
                        help="ckpt path")
    parser.add_argument("--Transfer", type=str, default="",
                        help="pretrain")
    parser.add_argument("--device", default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--auto", default=False, action='store_true',
                        help="no create new dir")
    args = parser.parse_args()

    cfg = load_setting(args.setting)

    cfg.update(vars(args))
    processor, init_model = get_processor_model(cfg.model_dir, cfg.max_seq_len)
    train_set = Latex_Dataset(cfg, cfg.train_data, processor)

    train_dataloaders = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        drop_last=True
    )

    val_set = Latex_Dataset(cfg, cfg.val_data, processor)

    val_dataloaders = DataLoader(
        val_set,
        batch_size=4,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=cfg.pin_memory,
    )
    cfg["steps_per_epoch"] = len(train_dataloaders)
    print(cfg)
    if cfg.auto:
        dirpath = dir_check(cfg.save_path, cfg.name, increase=False)
        if not os.path.exists(dirpath):
            dirpath = dir_check(cfg.save_path, cfg.name)
    else:
        dirpath = dir_check(cfg.save_path, cfg.name)
    cfg['dirpath'] = dirpath
    model = Simple_Latex_OCR(cfg, init_model=init_model, processor=processor)
    if cfg.Transfer:
        model = get_Transfer(model, cfg)

    wandb_logger = WandbLogger(project=cfg.experiment, name=cfg.name)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="bleu",
        dirpath=dirpath,
        filename="{epoch:02d}-{bleu:f}",
        save_top_k=3,
        mode="max",
    )
    save_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="{epoch}-{step}",
        save_top_k=1,
        save_weights_only=False,
        save_last=True,
        every_n_train_steps=1000,
        mode="max"
    )
    early_stop = EarlyStopping(monitor="bleu", patience=5, verbose=True, mode="max")

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(

        devices="auto",
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        # gradient_clip_val=100,
        # gradient_clip_algorithm='norm',
        callbacks=[
            ckpt_callback,
            lr_callback,
            early_stop,
            save_callback,
            # ModelPruning("l1_unstructured", amount=0.5)
        ],

    )

    trainer.fit(model, ckpt_path=cfg.resume if cfg.resume else None, train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders)
    print(f"save path is :{cfg.save_path}/version_{cfg.name}")

# #conda env list
# # conda activate  /opt/miniconda3/envs/doc2x
