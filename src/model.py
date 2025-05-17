import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=50, gpt2_model='gpt2'):
        super().__init__()
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Linear layer to project CNN output to GPT-2 embedding size
        self.cnn_projection = nn.Linear(128 * 25 * 10, 768)  # Adjust based on CNN output size
        # GPT-2
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model) if gpt2_model else GPT2Model(GPT2Config())
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model) if gpt2_model else GPT2Config()
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(self.gpt2_config.n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 2, max_len, n_mfcc), e.g., (batch, 2, 100, 40)
        cnn_out = self.cnn(x)  # (batch, 128 * 25 * 10)
        cnn_out = self.cnn_projection(cnn_out)  # (batch, n_embd)
        cnn_out = cnn_out.unsqueeze(1)  # (batch, 1, n_embd)
        gpt_out = self.gpt2(inputs_embeds=cnn_out).last_hidden_state  # (batch, 1, n_embd)
        logits = self.fc(gpt_out[:, -1, :])  # (batch, num_classes)
        return logits


def get_baseline_model(num_classes=50):
    config = GPT2Config(n_layer=6, n_head=8, n_embd=512)
    model = AudioClassifier(num_classes, gpt2_model=None)
    model.gpt2 = GPT2Model(config)  # Randomly initialized GPT-2
    model.gpt2_config = config  # Update config to match
    model.cnn_projection = nn.Linear(128 * 25 * 10, 512)  # Match baseline n_embd
    model.fc = nn.Sequential(  # Update classification head
        nn.Linear(config.n_embd, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model