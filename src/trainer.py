"""Module for the Model Trainer, including training and validation loops."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Closes training if validation loss doesn't improve after a set period."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initializes EarlyStopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Checks if training should stop.

        Args:
            val_loss: The current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Orchestrator for model training and validation.

    Handles optimization, scheduling, gradient clipping, and metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ) -> None:
        """Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer (e.g., Adam).
            criterion: The loss function (e.g., QuantileLoss).
            scheduler: Optional learning rate scheduler.
            device: Device to run training on ('cpu' or 'cuda').
            grad_clip: Maximum norm for gradient clipping.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Runs one training epoch.

        Args:
            train_loader: DataLoader for the training set.

        Returns:
            The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Runs validation.

        Args:
            val_loader: DataLoader for the validation set.

        Returns:
            The average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss = self.criterion(preds, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Fits the model to the data.

        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            epochs: Maximum number of epochs.
            early_stopping_patience: Patience for early stopping.

        Returns:
            A dictionary containing training and validation loss history.
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

        return self.history
