import torch


class LogisticModel(torch.nn.Module):
    def __init__(self, n_features):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
