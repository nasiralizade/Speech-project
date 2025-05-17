# --- model.py ---
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=50, gpt2_model='gpt2'):
        super().__init__()

        if gpt2_model:
            self.gpt2_config = GPT2Config.from_pretrained(gpt2_model)
            self.gpt2 = GPT2Model.from_pretrained(gpt2_model)
        else:
            self.gpt2_config = GPT2Config(n_layer=6, n_head=8, n_embd=512)
            self.gpt2 = GPT2Model(self.gpt2_config)

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # cnn_out shape: (batch, 128, 25, 10) → tokens = 25 time steps, 128×10 dims each
        self.cnn_projection = nn.Linear(128 * 10, self.gpt2_config.n_embd)

        self.fc = nn.Sequential(
            nn.Linear(self.gpt2_config.n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = self.cnn(x)  # (B, 128, 25, 10)
        cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()  # (B, 25, 128, 10)
        tokens = cnn_out.view(batch_size, 25, -1)  # (B, 25, 1280)
        embeddings = self.cnn_projection(tokens)  # (B, 25, n_embd)
        gpt_out = self.gpt2(inputs_embeds=embeddings).last_hidden_state  # (B, 25, n_embd)
        pooled = gpt_out.mean(dim=1)  # mean pool over sequence
        return self.fc(pooled)



def get_baseline_model(num_classes=50, num_tokens=4):
    model = AudioClassifier(num_classes=num_classes, gpt2_model=None)
    cnn_output_dim = 128 * 25 * 10
    assert cnn_output_dim % num_tokens == 0
    token_dim = cnn_output_dim // num_tokens

    model.cnn_projection = nn.Linear(128 * 10, 512)
    return model