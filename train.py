import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from model import SimpleCNN
from data_loader import get_data_loaders

def train_model(epochs=5, learning_rate=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize data loaders
    train_loader, test_loader, classes = get_data_loaders(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    print("Starting Training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{epochs}] Summary: Train Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    total_time = time.time() - start_time
    print(f"Finished Training. Total time: {total_time:.2f}s")

    # Plot results
    plot_results(train_losses, test_accuracies)

    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def plot_results(losses, accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results.png')
    print("Results plot saved as results.png")
    # plt.show() # Can't show in headless environment

if __name__ == "__main__":
    # For demonstration, we'll run a few epochs
    # In a real scenario, you'd run for 20-50 epochs
    train_model(epochs=3)
