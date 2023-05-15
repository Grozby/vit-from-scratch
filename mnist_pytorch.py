import logging
from datetime import datetime
from typing import Tuple

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from pytorch.vision_transformer.vit import ViT


def get_train_val_loaders(
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_set = torchvision.datasets.MNIST(
        root="./dataset/MNIST",
        train=True,
        transform=transform,
        download=True,
    )
    validation_set = torchvision.datasets.MNIST(
        root="./dataset/MNIST",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=128,
    )
    return train_loader, validation_loader


class Trainer:

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.logger = logging.getLogger("Trainer")
        self._current_epoch = 0
        self._timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._tensorboard_writer = SummaryWriter(f'mnist/{self._timestamp}')
        self.epochs = epochs

    def train_one_epoch(self):
        self._current_epoch += 1
        self.model.train(True)

        total_loss = 0.
        current_loss = 0.

        for i, data in enumerate((pbar := tqdm(self.train_loader))):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Gather data and report
            current_loss += loss.item()
            total_loss += loss.item()
            pbar.set_description(desc=f"Batch: {i + 1: <8} | "
                                 f"Loss: {total_loss / (i + 1):.4f}")
            if i % 1000 == 999:
                last_loss = current_loss / 1000  # loss per batch
                self._tensorboard_writer.add_scalar(
                    'Loss/train',
                    last_loss,
                    self._current_epoch * len(self.train_loader) + i + 1,
                )
                total_loss += current_loss
                current_loss = 0.

        if current_loss != 0:
            total_loss += current_loss
        return total_loss / (i + 1)

    def validation_one_epoch(self) -> float:
        self.model.train(False)
        total_loss = 0.0
        for i, data in enumerate((pbar := tqdm(self.validation_loader))):
            inputs, labels = data
            outputs = self.model(inputs)
            total_loss += self.loss_fn(outputs, labels)
            pbar.set_description(desc=f"Batch: {i + 1: <8} | "
                                 f"Loss: {total_loss / (i + 1):.4f}")

        return total_loss / (i + 1)

    def main(self):
        best_validation_loss = 0.
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}")
            train_loss = self.train_one_epoch()
            validation_loss = self.validation_one_epoch()
            # Log the running loss averaged per batch
            # for both training and validation
            self._tensorboard_writer.add_scalars(
                'Training vs. Validation Loss',
                {
                    'Training': train_loss,
                    'Validation': validation_loss
                },
                epoch + 1,
            )
            self._tensorboard_writer.flush()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    model.state_dict(),
                    f"./models/{self._timestamp}_epoch_{epoch + 1}",
                )


if __name__ == "__main__":
    train_loader, validation_loader = get_train_val_loaders()
    trainer = Trainer(
        model=(model := ViT(
            model_dim=768,
            feed_forward_hidden_dim=768 * 4,
            number_attention_heads=12,
            number_stacks=12,
            channels=1,
            image_shape=(28, 28),
            patch_size=14,
            number_classes=10,
        )),
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    trainer.main()