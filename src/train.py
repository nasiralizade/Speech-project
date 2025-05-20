import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import AudioClassifier, get_baseline_model
import os
import numpy.core.multiarray
import numpy
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def train_model(model, train_loader, val_loader, num_epochs=10, lr=2e-5, device='cuda', model_name='model', patience=9):
    criterion = nn.CrossEntropyLoss()

    # Configure freezing and differential learning rates for fine-tuned model
    if model_name == 'model':  # Fine-tuned model (gpt2-small)
        # Freeze all GPT-2 layers except the last 2 blocks
        if hasattr(model, 'gpt2') or hasattr(model, 'transformer'):
            gpt2 = model.gpt2 if hasattr(model, 'gpt2') else model.transformer
            for i, block in enumerate(gpt2.h):
                if i < len(gpt2.h) - 2:  # Freeze blocks 0 to 9 (0-based indexing)
                    for p in block.parameters():
                        p.requires_grad = False
                else:  # Unfreeze blocks 10 and 11
                    for p in block.parameters():
                        p.requires_grad = True
        # Differential learning rates
        head_lr = 1e-4
        wd = 1e-2
        groups = [
            {'params': [p for p in
                        (model.gpt2.parameters() if hasattr(model, 'gpt2') else model.transformer.parameters()) if
                        p.requires_grad], 'lr': lr, 'weight_decay': wd},
            {'params': model.cnn_projection.parameters(), 'lr': head_lr, 'weight_decay': wd},
            {'params': model.fc.parameters(), 'lr': head_lr, 'weight_decay': wd},
            {'params': model.cnn.parameters(), 'lr': head_lr, 'weight_decay': wd},
        ]
    else:  # Baseline model (randomly initialized, no freezing)
        groups = [{'params': model.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2}]

    optimizer = optim.AdamW(groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # Validate labels
    for X, y in train_loader:
        if y.min() < 0 or y.max() >= 50:
            raise ValueError(f"Invalid train labels: min={y.min()}, max={y.max()}")

    best_val_acc = 0
    epochs_no_improve = 0

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
                val_loss = criterion(outputs, y)

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_acc)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss / len(train_loader):.4f}, '
              f'Train Acc={train_acc:.4f},Val loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]["lr"]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/best_{model_name}.pt')
            print(f'Best Model Updated: Acc={best_val_acc:.4f}')
            print(f'⎺' * 50)
        else:
            epochs_no_improve += 1
            print(f'Early Stopping Counter: {epochs_no_improve}/{patience}')
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == '__main__':
    # Load preprocessed data
    data = torch.load('data/preprocessed/preprocessed.pt')
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
    num_epochs = 20
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=2e-5, device=device, model_name='model',
                patience=20)

    # Train baseline model
    print("Start training baseline_model\n")
    train_model(baseline_model, train_loader, val_loader, num_epochs=num_epochs, lr=1e-5, device=device,
                model_name='baseline_model', patience=20)

    # just testing the gpt2-medium model, to see if it performs better
    mode_medium = AudioClassifier(num_classes=50, gpt2_model='gpt2-medium').to(device)
    print("Start training medium_model\n")
    train_model(mode_medium, train_loader, val_loader, num_epochs=num_epochs, lr=2e-5, device=device,
                model_name='medium_model', patience=20)