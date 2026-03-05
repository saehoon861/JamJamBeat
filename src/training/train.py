import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

# Import the model definition from src/models
from src.models.baseline.mlp_classifier import SimpleMLPClassifier

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = 66  # Example: 22 landmarks * 3 coordinates (x, y, z)
    hidden_size = 128
    num_classes = 10 # Example: Number of distinct gestures
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001
    validation_split_ratio = 0.2

    # 1. Generate Dummy Data (replace with actual data loading later)
    num_samples = 1000
    dummy_features = torch.randn(num_samples, input_size)
    dummy_labels = torch.randint(0, num_classes, (num_samples,))

    # 2. Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        dummy_features, dummy_labels, test_size=validation_split_ratio, random_state=42, stratify=dummy_labels
    )

    # Convert to TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # 3. Initialize the model, loss function, and optimizer
    model = SimpleMLPClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Validation Loss: {epoch_val_loss:.4f}")

        # Basic model saving (optional, uncomment if you want to save the best model)
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     os.makedirs('checkpoints', exist_ok=True)
        #     model_save_path = os.path.join('checkpoints', 'best_mlp_model.pth')
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"  --> Saved best model with Validation Loss: {best_val_loss:.4f}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_model()