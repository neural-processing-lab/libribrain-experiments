from torch import nn
import torch


class AverageGroups(nn.Module):
    def __init__(self, n_groups):
        """
        Assumes input channels are grouped as torch.stack([group1, group2, ...])
        Expects input of shape (B, C, L).
        """
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x):
        return x.view(x.size(0), self.n_groups, x.size(1)//self.n_groups, x.size(2)).mean(dim=1)


def test_average_groups():
    x = torch.randn(10, 100, 57)
    avg_groups = AverageGroups(10)
    assert avg_groups(x).shape == (10, 10, 57)

    x_groups = [torch.randn(10, 10, 57) for _ in range(10)]
    x_concat = torch.cat(x_groups, dim=1)
    assert torch.all(avg_groups(x_concat) ==
                     torch.stack(x_groups, dim=1).mean(dim=1))


if __name__ == "__main__":
    test_average_groups()
    print("test successful")
