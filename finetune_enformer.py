import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef


class EnformerFinetune(nn.Module):
    def __init__(self, pretrained_enformer: nn.Module, num_chip_tracks: int = 5):
        super(EnformerFinetune, self).__init__()
        self.enformer = pretrained_enformer
        self.num_chip_tracks = num_chip_tracks

        for param in self.enformer.parameters():
            param.requires_grad = False

        # # Add new output heads for ChIP tracks
        # if hasattr(self.enformer, "add_heads") and callable(getattr(self.enformer, "add_heads")):
        #     self.enformer = self.enformer.add_heads(self.num_chip_tracks)
        # else:
        #     # If add_heads is not available, add a new head manually
        #     # Assumes the enformer has an attribute '_heads' as a ModuleDict with 'human' and 'mouse' keys  
        #     output = nn.ModuleDict({
        #         'human': nn.Sequential(
        #             nn.Linear(in_features=3072, out_features=num_chip_tracks, bias=True),
        #             nn.Softplus(beta=1.0, threshold=20.0)),
        #         'mouse': nn.Sequential(
        #             nn.Linear(in_features=3072, out_features=num_chip_tracks, bias=True),
        #             nn.Softplus(beta=1.0, threshold=20.0))
        #     })
        #     if hasattr(self.enformer, "_heads"):
        #         self.enformer._heads = output
        output = nn.ModuleDict({
                'human': nn.Sequential(
                    nn.Linear(in_features=3072, out_features=num_chip_tracks, bias=True),
                    nn.Softplus(beta=1.0, threshold=20.0)),
                'mouse': nn.Sequential(
                    nn.Linear(in_features=3072, out_features=num_chip_tracks, bias=True),
                    nn.Softplus(beta=1.0, threshold=20.0))
            })
        if hasattr(self.enformer, "_heads"):
            self.enformer._heads = output
        else:
            raise ValueError("Enformer does not have _heads attribute")

    def forward(self, sequence: torch.Tensor):
        return self.enformer(sequence)





# class EnformerFinetune(nn.Module):
#     def __init__(self, pretrained_enformer: nn.Module, num_chip_tracks: int = 5,
#                  trunk_output_dim: int = 1536, head_input_dim: int = 3072):
#         super().__init__()
        
#         # 1. 将Enformer的主干和头部分离
#         #    这仍然需要对模型结构有一些了解
#         if hasattr(pretrained_enformer, 'trunk'): # 假设主干叫 'trunk'
#             self.enformer_trunk = pretrained_enformer.trunk
#         else:
#             # 如果没有明确的trunk，就把它当作一个整体，但要移除最后的头
#             # 这通常通过创建一个不包含最后几层的新 nn.Sequential 来实现
#             # 这里我们简化，假设可以直接访问主干
#             # 这是一个需要根据你使用的Enformer库来适配的地方
#             self.enformer_trunk = nn.Sequential(*list(pretrained_enformer.children())[:-1])

#         # 2. 冻结主干参数
#         for param in self.enformer_trunk.parameters():
#             param.requires_grad = False
            
#         # 3. 创建你自己的、完全独立的输出头
#         #    注意: Enformer的输出是 (batch, seq_len, features)
#         #    这里的seq_len是896, features是1536*2=3072
#         #    线性层作用在最后一个维度上
#         self.new_head = nn.Sequential(
#             nn.Linear(in_features=head_input_dim, out_features=num_chip_tracks),
#             nn.Softplus()
#         )

#     def forward(self, sequence: torch.Tensor, organism: str = 'human'):
#         # 1. 通过主干网络提取特征
#         #    enformer的输出形状通常是 {'human': (B, 896, 1536), 'mouse': ...}
#         #    或者是一个元组 (human_trunk_out, mouse_trunk_out)
#         #    这里需要根据你使用的库来适配
#         trunk_output = self.enformer_trunk(sequence)
        
#         # 假设trunk_output是字典
#         organism_features = trunk_output[organism] # Shape: (B, 896, 1536)

#         # Enformer在送入head前会有一个 target_length x channels 的reshape
#         # 这里我们需要模拟这个行为，或者直接使用 head_input_dim
#         # 假设 enformer 在送入head前，特征维度是3072
#         # 这里的逻辑需要精确匹配原始模型
        
#         # 2. 将特征通过新的输出头
#         predictions = self.new_head(organism_features) # Shape: (B, 896, num_chip_tracks)

#         return predictions
    

class EnformerFineTunerPL(pl.LightningModule):
    def __init__(self, enformer_model: nn.Module, num_chip_tracks: int = 5, learning_rate: float = 1e-4, loss_fn: str = "poisson"):
        super(EnformerFineTunerPL, self).__init__()
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
        self.test_pearson = PearsonCorrCoef(num_outputs=num_chip_tracks)

        self.save_hyperparameters()


    # todo: 处理跨物种的问题
    def forward(self, sequence: torch.Tensor, organism: str = "human"):
        return self.model(sequence)[organism]
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.loss_fn(predictions, targets)

        self.train_pearson.update(predictions.view(-1, self.num_chip_tracks),
                                 targets.view(-1, self.num_chip_tracks))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
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

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self.log('val_pearson_epoch', self.val_pearson.compute().mean(), prog_bar=True)
        self.log_dict({f'val_pearson_{i}': self.val_pearson.compute()[i].item() for i in range(self.num_chip_tracks)}, logger=True)
        self.val_pearson.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
