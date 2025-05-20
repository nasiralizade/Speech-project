import torch
from torch.utils.data import TensorDataset, DataLoader

from evaluate import evaluate_model, plot_confusion_matrix
from src.model import AudioClassifier
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Load preprocessed data
    data = torch.load('data/preprocessed/preprocessed.pt')
    X_test, y_test = data['X_test'], data['y_test']
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load models
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    model_before = AudioClassifier(num_classes=50).to(device)
    acc_fb, prec_fb, rec_fb, f1_fb, y_true_fb, y_pred_fb = evaluate_model(model_before, test_loader, device)
    base_model_before = AudioClassifier(num_classes=50,gpt2_model=None).to(device)
    acc_bf, prec_bf, rec_bf, f1_bf, y_true_bf, y_pred_bf = evaluate_model(base_model_before, test_loader, device)

    model= AudioClassifier(num_classes=50).to(device)
    model.load_state_dict(torch.load('models/best_model.pt'))
    acc_a, prec_a, rec_a, f1_a, y_true_a, y_pred_a = evaluate_model(model, test_loader, device)

    base_model_m = AudioClassifier(num_classes=50,gpt2_model=None).to(device)
    base_model_m.load_state_dict(torch.load('models/best_baseline_model.pt'))
    acc_b_m, prec_b_m, rec_b_m, f1_b_m, y_true_b_m, y_pred_b_m = evaluate_model(base_model_m, test_loader, device)

    # Data for plotting
    labels = ['Fine-tuned (Before)', 'Fine-tuned (After)', 'Base (Before)', 'Base (After)']
    accuracies = [acc_fb * 100, acc_a * 100, acc_bf * 100, acc_b_m * 100]
    colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']  # Blue for fine-tuned, orange for base

    # Create the bar plot
    plt.figure(figsize=(12, 10))
    bars = plt.bar(labels, accuracies, color=colors)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{acc:.2f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Before and After Training')
    plt.ylim(0, 100)
    plt.savefig('notebooks/accuracy_before_after_all_models.png')
    plt.close()


