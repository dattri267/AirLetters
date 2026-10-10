"""Small trajectory classifiers. Input (B, T, C) -> logits (B, n_classes).
Every model starts with a BatchNorm over channels, so raw/normalized inputs are standardized on the fly."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, T, C, n_classes, width=512, dropout=0.3):
        super().__init__()
        self.inorm = nn.BatchNorm1d(C)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * C, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, n_classes),
        )

    def forward(self, x):
        x = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        return self.net(x)


def _block(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                         nn.Conv1d(cout, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU())


class CNN1D(nn.Module):
    def __init__(self, T, C, n_classes, width=64, dropout=0.3, k=5):
        super().__init__()
        self.inorm = nn.BatchNorm1d(C)
        self.b1, self.b2, self.b3 = _block(C, width, k), _block(width, 2 * width, k), _block(2 * width, 4 * width, k)
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * width, n_classes))

    def forward(self, x):
        x = self.inorm(x.transpose(1, 2))
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)
        x = torch.cat([x.mean(-1), x.amax(-1)], 1)
        return self.head(x)


class BiGRU(nn.Module):
    def __init__(self, T, C, n_classes, width=128, dropout=0.3, layers=2):
        super().__init__()
        self.inorm = nn.BatchNorm1d(C)
        self.rnn = nn.GRU(C, width, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        x = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


def build(name, T, C, n_classes, width=None, dropout=0.3):
    cls = {"mlp": MLP, "cnn1d": CNN1D, "bigru": BiGRU}[name]
    kw = dict(dropout=dropout)
    if width:
        kw["width"] = width
    return cls(T, C, n_classes, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
