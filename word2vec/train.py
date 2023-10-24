from typing import Tuple

import lightning.pytorch as pl
import torch

from . import model


class Word2Vec_SkipGram(pl.LightningModule):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.model = model.SkipGramModel(vocab_size, embed_size)
        self.criterion = model.NegativeSamplingLoss()
        self.save_hyperparameters(ignore=["model", "criterion"])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, batch_idx, step_type="train")
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _ = self._shared_step(batch, batch_idx, step_type="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _ = self._shared_step(batch, batch_idx, step_type="test")

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, step_type: str
    ) -> Tuple[torch.Tensor, float]:
        # Unpack batch into input, output and noise batches
        x_batch, y_batch, noise_batch = batch
        # Forward network
        x_embed_batch = self.model.in_forward(x_batch)
        y_embed_batch = self.model.out_forward(y_batch)
        noise_embed_batch = self.model.out_forward(noise_batch)
        # Forward loss function
        loss = self.criterion(x_embed_batch, y_embed_batch, noise_embed_batch)

        acc = 0.0
        self.log_dict({f"{step_type}_acc": acc, f"{step_type}_loss": loss}, on_epoch=True, on_step=True)
        return loss, acc

    # def predict_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    # ) -> torch.Tensor:
    #     x_batch, y_batch = batch
    #     y_pred_batch = self.model(x_batch)
    #     return y_pred_batch

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)
