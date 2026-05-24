import copy
import torch
import torch.nn.functional as F

class ElasticOptim:
    def __init__(self, workers, master, optimizers, val_loader,
        alpha=0.1,
        beta=5.0,
        tau=10,
        device="cuda"
    ):
        self.workers = workers
        self.master = master
        self.optimizers = optimizers
        self.val_loader = val_loader

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.device = device

        self.step_count = 0

        
    @torch.no_grad()
    def compute_val_losses(self):
        losses = [0.0 for _ in self.workers]

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)

            for i, model in enumerate(self.workers):
                out = model(x)
                loss = F.cross_entropy(out, y)
                losses[i] += loss.item()

        return losses
    
    def compute_weights(self, losses):
        loss_tensor = torch.tensor(losses, device=self.device)
        weights = torch.softmax(-self.beta * loss_tensor, dim=0)
        return weights
    

    def local_step(self, batches):
        for i, model in enumerate(self.workers):
            x, y = batches[i]
            x, y = x.to(self.device), y.to(self.device)

            self.optimizers[i].zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            self.optimizers[i].step()

    @torch.no_grad()
    def elastic_sync(self, weights):
        # snapshot workers
        worker_params = [
            [p.data.clone() for p in model.parameters()]
            for model in self.workers
        ]

        master_params = [p.data for p in self.master.parameters()]

        # update master
        for j, p_m in enumerate(master_params):
            delta = 0.0

            for i in range(len(self.workers)):
                delta += weights[i] * (worker_params[i][j] - p_m)

            p_m += self.alpha * delta

        # pull workers toward master
        for i, model in enumerate(self.workers):
            for j, p_w in enumerate(model.parameters()):
                p_w.data -= self.alpha * weights[i] * (p_w.data - master_params[j])


    def step(self, batches):
        self.local_step(batches)

        self.step_count += 1

        if self.step_count % self.tau == 0:
            losses = self.compute_val_losses()
            weights = self.compute_weights(losses)
            self.elastic_sync(weights)