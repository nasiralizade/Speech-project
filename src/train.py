import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import AudioClassifier, get_baseline_model
import os
import torch.serialization
import numpy.core.multiarray
import numpy

# Allowlist NumPy globals for safe loading
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, numpy.ndarray])


def train_model(model, train_loader, val_loader, num_epochs=10, lr=2e-5, device='cuda', model_name='model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == y).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_correct += (outputs.argmax(dim=1) == y).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        print(
            f'Epoch {epoch + 1}: Train Loss={train_loss / len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/best_{model_name}.pt')
            print(f'Best Model: Acc={best_val_acc:.4f}')


if __name__ == '__main__':
    # Load preprocessed data
    data = torch.load('data/preprocessed/preprocessed.pt',weights_only=False)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize models
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(num_classes=50).to(device)
    baseline_model = get_baseline_model(num_classes=50).to(device)

    # Train fine-tuned model
    os.makedirs('models', exist_ok=True)
    num_epochs = 100
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=2e-5, device=device, model_name='model')

    # Train baseline model
    train_model(baseline_model, train_loader, val_loader, num_epochs=num_epochs, lr=2e-5, device=device, model_name='baseline_model')