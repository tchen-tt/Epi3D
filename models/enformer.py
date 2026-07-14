import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
from torchmetrics.regression import PearsonCorrCoef
from enformer_pytorch import from_pretrained


class EnformerFinetune(nn.Module):
    def __init__(self, pretrained_enformer: nn.Module, num_chip_tracks: int = 5):
        super().__init__()
        self.enformer = pretrained_enformer
        self.num_chip_tracks = num_chip_tracks

        for param in self.enformer.parameters():
            param.requires_grad = False

        output = nn.ModuleDict({
            'human': nn.Sequential(
                nn.Linear(3072, num_chip_tracks, bias=True),
                nn.Softplus(beta=1.0, threshold=20.0)),
            'mouse': nn.Sequential(
                nn.Linear(3072, num_chip_tracks, bias=True),
                nn.Softplus(beta=1.0, threshold=20.0))
        })
        if not hasattr(self.enformer, "_heads"):
            raise ValueError("Enformer does not have _heads attribute")
        self.enformer._heads = output

    def forward(self, sequence: torch.Tensor):
        return self.enformer(sequence)


class EnformerFineTunerPL(pl.LightningModule):
    def __init__(self, enformer_model: Optional[nn.Module] = None, num_chip_tracks: int = 5,
                 learning_rate: float = 1e-4, loss_fn: str = "poisson"):
        super().__init__()
        if enformer_model is None:
            enformer_model = from_pretrained('EleutherAI/enformer-official-rough')
        self.model = EnformerFinetune(enformer_model, num_chip_tracks)
        self.learning_rate = learning_rate
        self.num_chip_tracks = num_chip_tracks

        if loss_fn == "poisson":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False)
        elif loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

        self.train_pearson = PearsonCorrCoef(num_outputs=num_chip_tracks)
        self.val_pearson = PearsonCorrCoef(num_outputs=num_chip_tracks)
        self.save_hyperparameters(ignore=["enformer_model"])

    def forward(self, sequence: torch.Tensor, organism: str = "human"):
        return self.model(sequence)[organism]

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.loss_fn(predictions, targets)
        self.train_pearson.update(predictions.view(-1, self.num_chip_tracks),
                                  targets.view(-1, self.num_chip_tracks))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_pearson_epoch', self.train_pearson.compute().mean(), prog_bar=True)
        self.train_pearson.reset()

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.loss_fn(predictions, targets)
        self.val_pearson.update(predictions.view(-1, self.num_chip_tracks),
                                targets.view(-1, self.num_chip_tracks))
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        pearson = self.val_pearson.compute()
        self.log('val_pearson_epoch', pearson.mean(), prog_bar=True)
        self.log_dict({f'val_pearson_{i}': pearson[i].item()
                       for i in range(self.num_chip_tracks)})
        self.val_pearson.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class ChIPPredictionModel(nn.Module):
    def __init__(self, num_chip_tracks: int = 100,
                 target_length: int = 896, dropout_rate: float = 0.1):
        super().__init__()
        self.enformer = from_pretrained('EleutherAI/enformer-official-rough',
                                        target_length=target_length,
                                        dropout_rate=dropout_rate,
                                        use_tf_gamma=True)
        output = nn.ModuleDict({
            'human': nn.Sequential(
                nn.Linear(3072, num_chip_tracks, bias=True),
                nn.Softplus(beta=1.0, threshold=20.0)),
            'mouse': nn.Sequential(
                nn.Linear(3072, num_chip_tracks, bias=True),
                nn.Softplus(beta=1.0, threshold=20.0))
        })
        self.enformer._heads = output

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.enformer(sequence)
