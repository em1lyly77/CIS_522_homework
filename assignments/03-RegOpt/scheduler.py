from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Define a custom learning rate scheduler that first decays and then enters cosine annealing
    """

    def __init__(self, optimizer, gamma, T_max, eta_min=0, last_epoch=-1):
        # gamma
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        # self.num_epochs = num_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        updates learning rate accordingly: exponentially at first, then cosine annealing
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]

        # ... Your Code Here ...

        # self.base_lrs all are 0.001 (initial lr)
        # exponential decay
        # return [base_lr * math.exp(self.gamma) for base_lr in self.base_lrs]
        if self.last_epoch < 2500:
            return [i for i in self.base_lrs]
            # return [
            #     base_lr * math.exp(-self.gamma * self.last_epoch)
            #     for base_lr in self.base_lrs
            # ]

        else:
            return [
                1e-4
                + (base_lr - self.eta_min)
                * 1
                / 2
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                for base_lr in self.base_lrs
            ]

        # cosine annealing
        # return [(1 + math.cos())]
