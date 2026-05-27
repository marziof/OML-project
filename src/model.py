import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


### For MNIST ### 

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train(model, optimizer, loader, loss_fn, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total



### For CIFAR10 ###

class SimCLR(nn.Module):

    def __init__(self, proj_dim=128, hidden=2048):
        super().__init__()
        enc = models.resnet18(weights=None)
        enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        enc.maxpool = nn.Identity()
        enc.fc = nn.Identity()
        self.encoder = enc

        self.projector = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim) # (bs, proj_dim)
        )

    def normalize(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, x):
        h = self.encoder(x)
        z = self.normalize(self.projector(h))
        return h, z

# model = SimCLR(proj_dim=128).to(device)
# model = torch.compile(model)


@torch.compile
def nt_xent(z1, z2, tau=0.5):
    B, d = z1.shape
    z = torch.cat([z1, z2], dim=0)              # (2B, d)
    sim = (z @ z.t()) / tau                     # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)

