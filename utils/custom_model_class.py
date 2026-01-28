import torch.nn as nn

class CustomSTTModel(nn.Module):
    def __init__(self):
        super(CustomSTTModel, self).__init__()
        # This structure is a placeholder. Replace with your real architecture.
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 30)  # Assume 30 output characters

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
