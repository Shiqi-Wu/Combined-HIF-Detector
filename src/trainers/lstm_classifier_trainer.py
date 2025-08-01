import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
import sys
import logging
from datetime import datetime
import argparse

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module
from models.fault_lstm_classifier import LSTMClassifier


def setup_logging(config):
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    log_path = os.path.join(config["logging"]["log_dir"], config["logging"]["log_file"])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if config["logging"].get("use_wandb", False) and WANDB_AVAILABLE:
        wandb.init(
            project=config["logging"].get("project", "lstm-training"),
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logging.info("WandB initialized")
    elif config["logging"].get("use_wandb", False):
        logging.warning("WandB not installed. Falling back to local logging.")


class ClassifierTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["train"].get("grad_accum", 1),
            mixed_precision="no"
        )

        if config["train"].get("seed"):
            set_seed(config["train"]["seed"])

        self.model = model.to(torch.float64)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["train"]["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config["train"].get("lr_factor", 0.1),
            patience=config["train"].get("lr_patience", 10),
            verbose=self.accelerator.is_main_process
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch"],
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch"],
            shuffle=False,
            num_workers=0
        )

        

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader
        )

    def train(self):
        best_val_acc = 0
        for epoch in range(self.config["train"]["epochs"]):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            self.scheduler.step(val_loss)

            if self.accelerator.is_main_process:
                logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4e}, Train Acc: {train_acc:.4e}, "
                             f"Val Loss: {val_loss:.4e}, Val Acc: {val_acc:.4e}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
                if WANDB_AVAILABLE and self.config["logging"].get("use_wandb", False):
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "lr": self.optimizer.param_groups[0]['lr']
                    })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.accelerator.is_main_process:
                    torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self.config["train"]["save_path"])
        return best_val_acc

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total = 0, 0, 0

        for x, _, p in self.train_loader:
            x = x.to(torch.float64)
            p_indices = torch.argmax(p, dim=1)

            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, p_indices)
                self.accelerator.backward(loss)
                self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == p_indices).sum().item()
            total += p_indices.size(0)

        return total_loss / len(self.train_loader), total_correct / total

    def eval_epoch(self):
        self.model.eval()
        total_loss, total_correct, total = 0, 0, 0

        with torch.no_grad():
            for x, u, p in self.val_loader:
                x = x.to(torch.float64)
                p_indices = torch.argmax(p, dim=1)
                outputs = self.model(x)
                loss = self.criterion(outputs, p_indices)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == p_indices).sum().item()
                total += p_indices.size(0)

        return total_loss / len(self.val_loader), total_correct / total


def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    setup_logging(config)
    data_cfg = config["data"]

    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"],
        config=data_cfg
    )

    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    model = LSTMClassifier(config=config["model"])
    trainer = ClassifierTrainer(model, scaled_train, scaled_val, config)
    best_acc = trainer.train()

    if trainer.accelerator.is_main_process:
        logging.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
        wandb.finish()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description="Train LSTM Classifier")
    parser.add_argument("--config", type=str, default="config_lstm_classifier.json", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)