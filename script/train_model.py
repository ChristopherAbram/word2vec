import logging

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningArgumentParser, SaveConfigCallback

from word2vec.data import WikiText2DataModule
from word2vec.train import Word2Vec_SkipGram


def main() -> None:
    parser = LightningArgumentParser(description="Word2Vec training script")
    parser.add_class_arguments(WikiText2DataModule, nested_key="data")
    parser.add_class_arguments(Word2Vec_SkipGram, nested_key="model", skip={"vocab_size"})
    parser.add_class_arguments(Trainer, nested_key="trainer", skip={"callbacks", "logger"})
    args = parser.parse_args()

    # Create data module
    dm = WikiText2DataModule(**args.data.as_dict())
    dm.prepare_data()
    dm.setup(stage="fit")

    # Create model module
    model = Word2Vec_SkipGram(vocab_size=dm.vocab_size, embed_size=args.model.embed_size)

    # Create training callbacks
    # Save model on min val_loss
    default_root_dir = ".model"
    model_ckpt_clb = ModelCheckpoint(
        dirpath=default_root_dir, filename="best_model", monitor="val_loss", save_top_k=1, mode="min"
    )
    # Save training config before training
    save_cfg_clb = SaveConfigCallback(parser, config=args)

    # Define custom logger for training session
    logger = TensorBoardLogger("tensorboard_logs/", default_hp_metric=False)

    # Create trainer and run training
    trainer = Trainer(**args.trainer.as_dict(), callbacks=[model_ckpt_clb, save_cfg_clb], logger=logger)

    logging.info("Start training")
    trainer.fit(model, datamodule=dm)
    logging.info("Training completed")


if __name__ == "__main__":
    main()
