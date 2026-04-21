import os
import itertools
import copy
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.models.baseline.mlp_classifier import GestureMLP
from src.dataset.gesture_dataset import create_dataloaders


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_experiment(config, save_dir="checkpoints"):
    # 학습 자체 랜덤성 고정
    set_seed(config["train_seed"])

    run = wandb.init(
        entity="yqazxcv39372046-yong-in-tae-kwon-do-college",
        project="jamjambeat-gesture-mlp",
        config=config,
        reinit=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_dataloaders(
        csv_path=config["csv_path"],
        feature_columns=config["feature_columns"],
        label_col=config["label_col"],
        num_classes=config["num_classes"],
        batch_size=config["batch_size"],
        validation_split_ratio=config["validation_split_ratio"],
        test_split_ratio=config["test_split_ratio"],
        test_split_seed=config["test_split_seed"],
        train_val_split_seed=config["train_val_split_seed"],
        balance_train=config["balance_train"],
        majority_class=config["majority_class"],
        majority_ratio=config["majority_ratio"],
    )

    model = GestureMLP(
        input_dim=config["input_size"],
        num_classes=config["num_classes"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        use_batchnorm=config["use_batchnorm"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if config["optimizer_name"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer_name"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer_name']}")

    scheduler = None
    if config["scheduler_name"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"],
        )
    elif config["scheduler_name"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["gamma"],
            patience=2,
        )

    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []

    # 모델 구조 저장
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(config["num_epochs"]):
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_now = inputs.size(0)
            running_train_loss += loss.item() * batch_size_now
            running_train_acc += calculate_accuracy(outputs, labels) * batch_size_now

        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = running_train_acc / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                batch_size_now = inputs.size(0)
                running_val_loss += loss.item() * batch_size_now
                running_val_acc += calculate_accuracy(outputs, labels) * batch_size_now

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = running_val_acc / len(val_dataset)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch [{epoch+1}/{config['num_epochs']}] | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        if scheduler is not None:
            if config["scheduler_name"] == "ReduceLROnPlateau":
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        improved = early_stopping.step(epoch_val_loss)
        if improved:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            save_path = os.path.join(save_dir, config["save_name"])
            torch.save(best_model_wts, save_path)

        if early_stopping.should_stop:
            print("  --> Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    model.eval()
    running_test_loss = 0.0
    running_test_acc = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size_now = inputs.size(0)
            running_test_loss += loss.item() * batch_size_now
            running_test_acc += calculate_accuracy(outputs, labels) * batch_size_now

    epoch_test_loss = running_test_loss / len(test_dataset)
    epoch_test_acc = running_test_acc / len(test_dataset)

    result = {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_loss": epoch_test_loss,
        "test_acc": epoch_test_acc,
        "history": history,
        "config": config,
        "model_state_dict": best_model_wts,
    }

    wandb.log(
        {
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc,
        }
    )

    wandb.finish()
    return result


def manual_grid_search(param_grid, save_dir="checkpoints"):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    print(f"Total experiments: {len(all_combinations)}")
    best_result = None

    for i, combination in enumerate(all_combinations, start=1):
        config = dict(zip(keys, combination))
        print(f"\n\n[Experiment {i}/{len(all_combinations)}]")

        result = train_one_experiment(config, save_dir=save_dir)

        if best_result is None or result["best_val_loss"] < best_result["best_val_loss"]:
            best_result = result
            print(f"*** New best result found! Val Loss: {best_result['best_val_loss']:.4f}")

    return best_result


def parse_hidden_dims(hidden_dims_str):
    return [int(x.strip()) for x in hidden_dims_str.split(",")]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="Train GestureMLP with optional grid search")

    parser.add_argument("--use-grid-search", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="best_model.pth")

    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="gesture")

    parser.add_argument("--input-size", type=int, default=63)
    parser.add_argument("--num-classes", type=int, default=7)

    parser.add_argument("--test-split-seed", type=int, default=42)
    parser.add_argument("--train-val-split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)

    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batchnorm", type=str2bool, default=False)

    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--validation-split-ratio", type=float, default=0.2)
    parser.add_argument("--test-split-ratio", type=float, default=0.1)

    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="none",
        choices=["none", "StepLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument("--balance-train", type=str2bool, default=True)
    parser.add_argument("--majority-class", type=int, default=0)
    parser.add_argument("--majority-ratio", type=float, default=1.5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    scheduler_name = None if args.scheduler_name == "none" else args.scheduler_name
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    feature_columns = []
    for i in range(0, 21):
        feature_columns.extend([f"x{i}", f"y{i}", f"z{i}"])

    if args.use_grid_search:
        param_grid = {
            "csv_path": [args.csv_path],
            "feature_columns": [feature_columns],
            "label_col": [args.label_col],
            "input_size": [args.input_size],
            "num_classes": [args.num_classes],
            "hidden_dims": [[64, 32], [128, 64], [128, 128]],
            "dropout": [0.0, 0.003, 0.005, 0.01],
            "use_batchnorm": [False, True],
            "optimizer_name": [args.optimizer_name],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [0.0, 1e-5, 1e-4],
            "scheduler_name": [None],
            "step_size": [args.step_size],
            "gamma": [args.gamma],
            "batch_size": [32],
            "num_epochs": [args.num_epochs],
            "validation_split_ratio": [args.validation_split_ratio],
            "test_split_ratio": [args.test_split_ratio],
            "test_split_seed": [args.test_split_seed],
            "train_val_split_seed": [7, 42, 123],
            "train_seed": [args.train_seed],
            "save_name": [args.save_name],
            "early_stopping_patience": [args.early_stopping_patience],
            "early_stopping_min_delta": [args.early_stopping_min_delta],
            "balance_train": [args.balance_train],
            "majority_class": [args.majority_class],
            "majority_ratio": [args.majority_ratio],
        }

        best_result = manual_grid_search(param_grid, save_dir=args.save_dir)
        print(best_result)

    else:
        config = {
            "csv_path": args.csv_path,
            "feature_columns": feature_columns,
            "label_col": args.label_col,
            "input_size": args.input_size,
            "num_classes": args.num_classes,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "use_batchnorm": args.use_batchnorm,
            "optimizer_name": args.optimizer_name,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler_name": scheduler_name,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "validation_split_ratio": args.validation_split_ratio,
            "test_split_ratio": args.test_split_ratio,
            "test_split_seed": args.test_split_seed,
            "train_val_split_seed": args.train_val_split_seed,
            "train_seed": args.train_seed,
            "save_name": args.save_name,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "balance_train": args.balance_train,
            "majority_class": args.majority_class,
            "majority_ratio": args.majority_ratio,
        }

        result = train_one_experiment(config, save_dir=args.save_dir)
        print(result)


if __name__ == "__main__":
    main()