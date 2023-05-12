import click
import lightning as L

import ai.constants as C
from ai.data import DummyDataModule
from ai.models import Model


@click.command(name="main")
@click.option("--lr",
              help="Learning rate",
              prompt="Learning rate",
              type=float,
              default=C.LR)
@click.option("--batch_size",
              help="Batch size",
              prompt="Batch size",
              type=int,
              default=C.BATCH_SIZE)
@click.option("--epochs",
              help="Number of epochs",
              prompt="Number of epochs",
              type=int,
              default=C.MAX_EPOCHS)
def main(lr: float, batch_size: int, epochs: int):
    data_module = DummyDataModule(data_dir=C.PATH,
                                  batch_size=batch_size,
                                  num_workers=C.NUM_WORKERS,
                                  pin_memory=C.PIN_MEMORY,
                                  shuffle=C.SHUFFLE)

    model = Model(lr=lr)
    # To compile the model for maximum performance, uncomment the following lines:
    # compiled = torch.compile(model, fullgraph=True, mode="max-autotune")
    # compiled(torch.randn(32, 3, 32, 32))  # warmup

    trainer = L.Trainer(
        devices=1,
        accelerator="auto",
        benchmark=True,
        log_every_n_steps=1,
        max_epochs=epochs,
        precision=C.PRECISION,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    # set seed for reproducibility
    L.seed_everything(42069)
    main()
