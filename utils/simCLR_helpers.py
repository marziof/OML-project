import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models

# -----------------------
# SIMCLR AUGMENTATION
# -----------------------
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
    


def collate_fn(batch):
    x1, x2, y = [], [], []
    for (v1, v2), label in batch:
        x1.append(v1)
        x2.append(v2)
        y.append(label)
    return torch.stack(x1), torch.stack(x2), torch.tensor(y)


## EVAL

@torch.no_grad()
# def extract_features(model, loader, device):
#     model.eval()
#     feats, labels = [], []

#     for (x1, x2), y in loader:
#         x = x1.to(device)
#         h, _ = model(x)
#         h = F.normalize(h, dim=1)

#         feats.append(h.cpu())
#         labels.append(y)

#     return torch.cat(feats), torch.cat(labels)

def extract_features(model, loader, device, frac=1.0):
    model.eval()
    feats, labels = [], []
    max_batches = max(1, int(len(loader) * frac))
    with torch.no_grad():
        for i, ((x1, x2), y) in enumerate(loader):
            if i >= max_batches:
                break
            x1 = x1.to(device)
            h, _ = model(x1)
            feats.append(h.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

# -----------------------
# kNN EVALUATION
# -----------------------
@torch.no_grad()
def knn(feat_train, y_train, feat_test, y_test=None, k=20, T=0.07, num_classes=10):
    feat_train = F.normalize(feat_train, dim=1)
    feat_test  = F.normalize(feat_test,  dim=1)

    sims = feat_test @ feat_train.t()
    vals, idx = sims.topk(int(k), dim=1)
    neighbors = y_train[idx]

    weights = (vals / T).softmax(dim=1)

    scores = torch.zeros(feat_test.size(0), num_classes)
    scores.scatter_add_(1, neighbors, weights)

    pred = scores.argmax(dim=1)
    if y_test is not None:
        return (pred == y_test).float().mean().item()
    return pred


# -----------------------
# FEATURE EXTRACTION
# -----------------------
# @torch.no_grad()
# def extract_features(loader):
#     model.eval()
#     feats, labels = [], []

#     for (x1, x2), y in loader:
#         x = x1.to(device)
#         h, _ = model(x)
#         h = F.normalize(h, dim=1)

#         feats.append(h.cpu())
#         labels.append(y)

#     return torch.cat(feats), torch.cat(labels)