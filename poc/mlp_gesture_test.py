import torch
import torch.nn as nn
import torch.optim as optim

# Import the model definition from src/models
from src.models.baseline.mlp_classifier import SimpleMLPClassifier

def main():
    # Define hyperparameters (these are placeholders and should be tuned)
    input_size = 66  # Example: 22 landmarks * 3 coordinates (x, y, z)
    hidden_size = 128
    num_classes = 10 # Example: Number of distinct gestures

    # Initialize the model
    model = SimpleMLPClassifier(input_size, hidden_size, num_classes)
    
    # Define a loss function and optimizer (even for a quick test, to ensure they can be initialized)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- PoC Model Test ---")
    print("Model Architecture:")
    print(model)

    # --- Test with dummy data ---
    # Dummy input: batch_size x input_size
    dummy_input = torch.randn(16, input_size) 
    # Dummy target: batch_size (class indices)
    dummy_target = torch.randint(0, num_classes, (16,)) 

    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Dummy target shape: {dummy_target.shape}")

    # Forward pass
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    # Calculate loss (example of one training step)
    loss = criterion(output, dummy_target)
    print(f"Dummy loss (before optimization): {loss.item():.4f}")

    # Backward pass and optimization step (example of one training step)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("One optimization step performed with dummy data (parameters updated).")
    
    print("\nPoC Test Complete. Model initialized and processed dummy data.")

if __name__ == "__main__":
    main()
