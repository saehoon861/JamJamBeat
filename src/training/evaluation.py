import argparse
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

from src.models.baseline.mlp_classifier import GestureMLP
from src.dataset.gesture_dataset import create_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_hidden_dims(hidden_dims_str: str) -> List[int]:
    return [int(x.strip()) for x in hidden_dims_str.split(",")]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def evaluate_model(config):
    set_seed(config["train_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns = []
    for i in range(21):
        feature_columns.extend([f"x{i}", f"y{i}", f"z{i}"])

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_dataloaders(
        csv_path=config["csv_path"],
        feature_columns=feature_columns,
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
        use_batchnorm=config["use_batchnorm"],
    ).to(device)

    checkpoint = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []

    running_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            batch_size_now = inputs.size(0)
            running_test_loss += loss.item() * batch_size_now
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    test_loss = running_test_loss / len(test_dataset)
    test_acc = correct / total

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        zero_division=0,
    )

    print("\n===== Evaluation Result =====")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test acc : {test_acc:.6f}")

    print("\n===== Confusion Matrix =====")
    print(cm)

    print("\n===== Classification Report =====")
    print(report)

    if config["save_confusion_matrix_path"]:
        os.makedirs(os.path.dirname(config["save_confusion_matrix_path"]), exist_ok=True)
        np.savetxt(
            config["save_confusion_matrix_path"],
            cm,
            fmt="%d",
            delimiter=",",
        )
        print(f"\nConfusion matrix saved to: {config['save_confusion_matrix_path']}")

    if config["save_predictions_path"]:
        os.makedirs(os.path.dirname(config["save_predictions_path"]), exist_ok=True)
        with open(config["save_predictions_path"], "w", encoding="utf-8") as f:
            f.write("y_true,y_pred\n")
            for y_true, y_pred in zip(all_labels, all_preds):
                f.write(f"{y_true},{y_pred}\n")
        print(f"Predictions saved to: {config['save_predictions_path']}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate GestureMLP and print confusion matrix")

    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--label-col", type=str, default="gesture")

    parser.add_argument("--input-size", type=int, default=63)
    parser.add_argument("--num-classes", type=int, default=7)

    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batchnorm", type=str2bool, default=False)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split-ratio", type=float, default=0.2)
    parser.add_argument("--test-split-ratio", type=float, default=0.1)
    

    parser.add_argument("--test-split-seed", type=int, default=42)
    parser.add_argument("--train-val-split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)

    parser.add_argument("--balance-train", type=str2bool, default=True)
    parser.add_argument("--majority-class", type=int, default=0)
    parser.add_argument("--majority-ratio", type=float, default=1.5)

    parser.add_argument(
        "--save-confusion-matrix-path",
        type=str,
        default="outputs/confusion_matrix.csv",
    )
    parser.add_argument(
        "--save-predictions-path",
        type=str,
        default="outputs/test_predictions.csv",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = {
        "csv_path": args.csv_path,
        "model_path": args.model_path,
        "label_col": args.label_col,
        "input_size": args.input_size,
        "num_classes": args.num_classes,
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "dropout": args.dropout,
        "use_batchnorm": args.use_batchnorm,
        "batch_size": args.batch_size,
        "validation_split_ratio": args.validation_split_ratio,
        "test_split_ratio": args.test_split_ratio,
        "test_split_seed": args.test_split_seed,
        "train_val_split_seed": args.train_val_split_seed,
        "train_seed": args.train_seed,
        "balance_train": args.balance_train,
        "majority_class": args.majority_class,
        "majority_ratio": args.majority_ratio,
        "save_confusion_matrix_path": args.save_confusion_matrix_path,
        "save_predictions_path": args.save_predictions_path,
    }

    evaluate_model(config)


if __name__ == "__main__":
    main()