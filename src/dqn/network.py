from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes) -> None:
        super().__init__()

        sizes = [obs_dim, *hidden_sizes, n_actions]
        layers = []

        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))

            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
