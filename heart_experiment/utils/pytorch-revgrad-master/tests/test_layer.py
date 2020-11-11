import copy
import torch
from pytorch_revgrad import RevGrad


def test_gradients_inverted():
    network = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
    revnetwork = torch.nn.Sequential(copy.deepcopy(network), RevGrad())

    inp = torch.randn(8, 5)
    outp = torch.randn(8, 1)

    criterion = torch.nn.MSELoss()

    criterion(network(inp), outp).backward()
    criterion(revnetwork(inp), outp).backward()

    assert all(
        (p1.grad == -p2.grad).all()
        for p1, p2 in zip(network.parameters(), revnetwork.parameters())
    )
