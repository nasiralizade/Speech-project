import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import AudioClassifier, get_baseline_model
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def train_model(model, train_loader, val_loader,
                num_epochs=20, lr=2e-5, device='cuda',
                model_name='model', patience=500):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Freeze GPT-2
    if hasattr(model, 'gpt2'):
        for p in model.gpt2.parameters():
            p.requires_grad = False

    # head hyperparams
    head_lr = 1e-3
    wd = 1e-2
    unfreeze_schedule = {5: 0, 10: 1, 15: 2}  # epoch → block idx

    def create_optimizer():
        groups = []
        if hasattr(model, 'gpt2'):
            groups.append({'params': model.gpt2.parameters(),
                           'lr': lr, 'weight_decay': wd})
        if hasattr(model, 'cnn_projection'):
            groups.append({'params': model.cnn_projection.parameters(),
                           'lr': head_lr, 'weight_decay': wd})
        if hasattr(model, 'fc'):
            groups.append({'params': model.fc.parameters(),
                           'lr': head_lr, 'weight_decay': wd})
        return optim.AdamW(groups)

    optimizer = create_optimizer()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        # Gradual unfreeze
        if epoch_num in unfreeze_schedule and hasattr(model, 'gpt2'):
            idx = unfreeze_schedule[epoch_num]
            # correct path to GPT2 blocks:
            for p in model.gpt2.h[idx].parameters():
                p.requires_grad = True
            optimizer = create_optimizer()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5)
            print(f"→ Unfroze GPT-2 layer {idx}")

        # train
        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                val_correct += (preds == y).sum().item()
        val_acc = val_correct / len(val_loader.dataset)

        scheduler.step(val_acc)
        print(f"Epoch {epoch_num}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(),
                       f"models/best_{model_name}.pt")
            print(f"Best Model Updated: Acc={best_val_acc:.4f}")
            print("─" * 50)
        else:
            epochs_no_improve += 1
            print(f"Early Stopping Counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == '__main__':
    # load data
    data = torch.load('data/preprocessed/preprocessed.pt')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val     = data['X_val'],   data['y_val']

    train_ds = TensorDataset(torch.FloatTensor(X_train),
                             torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),
                             torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)

    device = ('mps' if torch.backends.mps.is_available()
              else ('cuda' if torch.cuda.is_available()
                    else 'cpu'))
    model = AudioClassifier(num_classes=50).to(device)
    baseline = get_baseline_model(num_classes=50).to(device)

    os.makedirs('models', exist_ok=True)
    # fine-tuned model
    train_model(model, train_loader, val_loader,
                num_epochs=20, lr=2e-5, device=device,
                model_name='model', patience=500)
    # baseline
    print("Start training baseline_model\n")
    train_model(baseline, train_loader, val_loader,
                num_epochs=20, lr=2e-5, device=device,
                model_name='baseline_model', patience=500)
