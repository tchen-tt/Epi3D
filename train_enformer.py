import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from enformer_pytorch import Enformer

from data_pipeline.data import EnformerSignalDataset
from models.enformer import EnformerFineTunerPL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Enformer on ChIP-seq tracks")
    p.add_argument("--train_bed", required=True)
    p.add_argument("--val_bed", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--bigwig_files", nargs="+", required=True)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--num_chip_tracks", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss_fn", default="poisson", choices=["poisson", "mse"])
    p.add_argument("--gpus", nargs="+", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_ds = EnformerSignalDataset(args.train_bed, args.fasta, args.bigwig_files)
    val_ds = EnformerSignalDataset(args.val_bed, args.fasta, args.bigwig_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=True)
    model = EnformerFineTunerPL(
        enformer_model=enformer,
        num_chip_tracks=args.num_chip_tracks,
        learning_rate=args.lr,
        loss_fn=args.loss_fn,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="enformer-finetuned-{epoch:02d}-{val_loss_epoch:.4f}",
            monitor="val_loss_epoch",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val_loss_epoch", patience=3, mode="min"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
