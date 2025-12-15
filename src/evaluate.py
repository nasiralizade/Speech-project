import torch
from torch.utils.data import DataLoader, TensorDataset
from model import AudioClassifier, get_baseline_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_labels=50, name='confusion_matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'notebooks/confusion_matrix_{name}.png')
    plt.close()


if __name__ == '__main__':
    # Load preprocessed data
    data = torch.load('data/preprocessed/preprocessed.pt')
    X_test, y_test = data['X_test'], data['y_test']
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load models
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    model = AudioClassifier(num_classes=50).to(device)
    model.load_state_dict(torch.load('models/best_model.pt'))
    baseline_model = get_baseline_model(num_classes=50).to(device)
    baseline_model.load_state_dict(torch.load('models/best_baseline_model.pt'))  # Load baseline weights

    # Evaluate
    acc, prec, rec, f1, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f'Fine-tuned Model: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}')

    baseline_acc, baseline_prec, baseline_rec, baseline_f1,b_y_true, b_y_pred = evaluate_model(baseline_model, test_loader, device)
    print(f'Baseline Model: Acc={baseline_acc:.4f}, Prec={baseline_prec:.4f}, Rec={baseline_rec:.4f}, F1={baseline_f1:.4f}')

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, name='fine_tuned')
    plot_confusion_matrix(b_y_true, b_y_pred, name='baseline')
