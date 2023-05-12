import lightning as L
import timm
import torch
import torch.nn.functional as F


class Model(L.LightningModule):

    def __init__(
        self,
        lr: float = 0.1,
    ):
        super().__init__()

        self.lr = lr

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=True,
            num_classes=1,
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch):
        x, y = batch
        y = torch.stack(tuple(y.values()), dim=1).float()

        y_hat = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.stack(tuple(y.values()), dim=1).float()

        y_hat = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]
