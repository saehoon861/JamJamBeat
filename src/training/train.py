import os
import itertools
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.models.baseline.mlp_classifier import GestureMLP


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


def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def create_dataloaders(
    X,
    y,
    batch_size=32,
    validation_split_ratio=0.2,
    test_split_ratio=0.1,
    random_state=42,
):
    # 1차 분할: train+val / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_split_ratio,
        random_state=random_state,
        stratify=y,
    )

    # 2차 분할: train / val
    # 전체 기준 validation_split_ratio가 유지되도록 보정
    val_ratio_adjusted = validation_split_ratio / (1.0 - test_split_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_val,
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def train_one_experiment(X, y, config, save_dir="checkpoints"):
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
        X=X,
        y=y,
        batch_size=config["batch_size"],
        validation_split_ratio=config["validation_split_ratio"],
        test_split_ratio=config["test_split_ratio"],
        random_state=config["random_state"],
    )

    model = GestureMLP(
        input_dim=config["input_size"],
        num_classes=config["num_classes"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        use_batchnorm=config["use_batchnorm"],
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
    elif config["scheduler_name"] is None:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config['scheduler_name']}")

    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc_at_best_loss = 0.0
    history = []

    for epoch in range(config["num_epochs"]):
        # -----------------
        # Train
        # -----------------
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

        # -----------------
        # Validation
        # -----------------
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

        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if scheduler is not None:
            if config["scheduler_name"] == "ReduceLROnPlateau":
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        improved = early_stopping.step(epoch_val_loss)
        if improved:
            best_val_loss = epoch_val_loss
            best_val_acc_at_best_loss = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            save_path = os.path.join(save_dir, config["save_name"])
            torch.save(best_model_wts, save_path)
            print(f"  --> Best model saved to {save_path}")

        if early_stopping.should_stop:
            print("  --> Early stopping triggered.")
            break

    # -----------------
    # Load best model
    # -----------------
    model.load_state_dict(best_model_wts)

    # -----------------
    # Final Test Evaluation
    # -----------------
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

    print(f"Final Test Loss: {epoch_test_loss:.4f} | Final Test Acc: {epoch_test_acc:.4f}")

    result = {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc_at_best_loss,
        "test_loss": epoch_test_loss,
        "test_acc": epoch_test_acc,
        "history": history,
        "config": config,
        "model_state_dict": best_model_wts,
    }

    return result


def manual_grid_search(X, y, param_grid, save_dir="checkpoints"):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    print(f"Total combinations: {len(all_combinations)}")

    best_result = None

    for i, combination in enumerate(all_combinations, start=1):
        config = dict(zip(keys, combination))
        print("\n" + "=" * 80)
        print(f"[Grid Search] Running combination {i}/{len(all_combinations)}")
        print(config)

        result = train_one_experiment(X, y, config, save_dir=save_dir)

        if best_result is None or result["best_val_loss"] < best_result["best_val_loss"]:
            best_result = result
            print("  --> New best result found.")

    print("\n" + "=" * 80)
    print("Grid Search Finished")
    print(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
    print(f"Best Val Acc : {best_result['best_val_acc']:.4f}")
    print(f"Test Loss    : {best_result['test_loss']:.4f}")
    print(f"Test Acc     : {best_result['test_acc']:.4f}")
    print("Best Config:")
    print(best_result["config"])

    return best_result


def parse_hidden_dims(hidden_dims_str):
    return [int(x.strip()) for x in hidden_dims_str.split(",")]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="Train GestureMLP with optional grid search.")

    # 실행 관련
    parser.add_argument("--use-grid-search", type=str2bool, default=False)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="best_mlp.pt")

    # 데이터 관련
    parser.add_argument("--input-size", type=int, default=63)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)

    # 모델 관련
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batchnorm", type=str2bool, default=False)

    # 학습 관련
    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--validation-split-ratio", type=float, default=0.2)
    parser.add_argument("--test-split-ratio", type=float, default=0.1)

    # 스케줄러 관련
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="none",
        choices=["none", "StepLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)

    # Early stopping 관련
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    scheduler_name = None if args.scheduler_name == "none" else args.scheduler_name
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    # 현재는 더미 데이터
    X = torch.randn(args.num_samples, args.input_size)
    y = torch.randint(0, args.num_classes, (args.num_samples,))

    if args.use_grid_search:
        param_grid = {
            "input_size": [args.input_size],
            "num_classes": [args.num_classes],
            "hidden_dims": [[128, 64], [256, 128], [128, 128, 64]],
            "dropout": [0.0, 0.3],
            "use_batchnorm": [False, True],
            "optimizer_name": [args.optimizer_name],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [0.0, 1e-4],
            "batch_size": [16, 32],
            "num_epochs": [args.num_epochs],
            "validation_split_ratio": [args.validation_split_ratio],
            "test_split_ratio": [args.test_split_ratio],
            "scheduler_name": [None, "StepLR", "ReduceLROnPlateau"],
            "step_size": [args.step_size],
            "gamma": [args.gamma],
            "early_stopping_patience": [args.early_stopping_patience],
            "early_stopping_min_delta": [args.early_stopping_min_delta],
            "random_state": [args.random_state],
            "save_name": [args.save_name],
        }

        manual_grid_search(X, y, param_grid, save_dir=args.save_dir)

    else:
        config = {
            "input_size": args.input_size,
            "num_classes": args.num_classes,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "use_batchnorm": args.use_batchnorm,
            "optimizer_name": args.optimizer_name,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "validation_split_ratio": args.validation_split_ratio,
            "test_split_ratio": args.test_split_ratio,
            "scheduler_name": scheduler_name,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "random_state": args.random_state,
            "save_name": args.save_name,
        }

        result = train_one_experiment(X, y, config, save_dir=args.save_dir)

        print("\n" + "=" * 80)
        print("Training Finished")
        print(f"Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"Best Val Acc : {result['best_val_acc']:.4f}")
        print(f"Test Loss    : {result['test_loss']:.4f}")
        print(f"Test Acc     : {result['test_acc']:.4f}")


if __name__ == "__main__":
    main()