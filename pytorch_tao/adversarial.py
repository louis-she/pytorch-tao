import torch


class Adversarial:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def save(self):
        pass

    def attack(self):
        pass

    def restore(self):
        pass


class AWP(Adversarial):
    def __init__(self, model, adv_lr=0.2, adv_eps=0.005):
        super().__init__(model)
        self.adv_param = "weight"
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup_eps = {}
        self.backup = {}

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad_norm = torch.norm(param.grad)
                weight_norm = torch.norm(param.data.detach())
                if grad_norm != 0 and not torch.isnan(grad_norm):
                    r_at = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
